from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from paperpilot.models import Chunk


def point_id(chunk_id: str) -> str:
    # qdrant wants an int or uuid; uuid5 gives the same id for the same chunk every time
    return str(uuid5(NAMESPACE_URL, chunk_id))


class VectorStore:
    """Qdrant in local mode. Only one process can open the folder at a time."""

    def __init__(self, path: Path, collection: str):
        self.client = QdrantClient(path=str(path))
        self.collection = collection

    def recreate(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(id=point_id(c.chunk_id), vector=v, payload=c.model_dump())
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), 256):
            self.client.upsert(self.collection, points=points[i : i + 256])

    def search(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        result = self.client.query_points(self.collection, query=vector, limit=k)
        return [(Chunk.model_validate(p.payload), p.score) for p in result.points]

    def close(self) -> None:
        self.client.close()
