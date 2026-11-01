import logging

from paperpilot.config import Settings
from paperpilot.models import Chunk, RetrievedChunk
from paperpilot.retrieval.fusion import rrf

log = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self, settings: Settings, chunks: dict[str, Chunk], embedder, store, bm25, reranker=None
    ):
        self.settings = settings
        self.chunks = chunks
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.reranker = reranker

    def retrieve(
        self, query: str, *, use_bm25: bool | None = None, use_reranker: bool | None = None
    ) -> list[RetrievedChunk]:
        s = self.settings
        use_bm25 = s.use_bm25 if use_bm25 is None else use_bm25
        use_reranker = s.use_reranker if use_reranker is None else use_reranker

        vector_hits = self.store.search(self.embedder.embed_query(query), s.vector_top_k)
        if use_bm25:
            bm25_ids = [cid for cid, _ in self.bm25.search(query, s.bm25_top_k)]
            ranked = rrf([[c.chunk_id for c, _ in vector_hits], bm25_ids], k=s.rrf_k)
        else:
            ranked = [(c.chunk_id, score) for c, score in vector_hits]

        if use_reranker:
            if self.reranker is None:
                raise ValueError("reranking is on but no reranker was loaded")
            candidates = [cid for cid, _ in ranked[: s.rerank_candidates]]
            passages = [self.chunks[cid].indexed_text for cid in candidates]
            scores = self.reranker.score(query, passages)
            ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: -x[1])

        results = [
            RetrievedChunk(chunk=self.chunks[cid], score=score)
            for cid, score in ranked[: s.final_top_k]
        ]
        log.info("%r -> %s", query, [(r.chunk.chunk_id, round(r.score, 3)) for r in results])
        return results

    def close(self) -> None:
        self.store.close()


def load_retriever(settings: Settings) -> Retriever:
    # heavy imports here, so tests that use fakes stay fast
    from paperpilot.retrieval.bm25 import BM25Index
    from paperpilot.retrieval.embedder import Embedder
    from paperpilot.retrieval.reranker import Reranker
    from paperpilot.retrieval.vector_store import VectorStore

    lines = (settings.processed_dir / "chunks.jsonl").read_text("utf-8").splitlines()
    chunks = {c.chunk_id: c for c in map(Chunk.model_validate_json, lines)}
    embedder = Embedder(
        settings.embedding_model, settings.embedding_query_prefix, settings.embedding_max_tokens
    )
    store = VectorStore(settings.qdrant_path, settings.qdrant_collection)
    bm25 = BM25Index.load(settings.bm25_path)
    reranker = Reranker(settings.reranker_model) if settings.use_reranker else None
    return Retriever(settings, chunks, embedder, store, bm25, reranker)
