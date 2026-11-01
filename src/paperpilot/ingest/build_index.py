"""Rebuild the Qdrant collection and the BM25 index from chunks.jsonl."""

import logging

from paperpilot.config import get_settings
from paperpilot.models import Chunk
from paperpilot.retrieval.bm25 import BM25Index
from paperpilot.retrieval.embedder import Embedder
from paperpilot.retrieval.vector_store import VectorStore

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = get_settings()
    lines = (s.processed_dir / "chunks.jsonl").read_text("utf-8").splitlines()
    chunks = [Chunk.model_validate_json(line) for line in lines]
    s.index_dir.mkdir(parents=True, exist_ok=True)

    embedder = Embedder(s.embedding_model, s.embedding_query_prefix, s.embedding_max_tokens)
    vectors = embedder.embed_chunks(chunks)

    store = VectorStore(s.qdrant_path, s.qdrant_collection)
    store.recreate(embedder.dim)
    store.add(chunks, vectors)
    store.close()
    log.info("qdrant: %d vectors (dim %d)", len(vectors), embedder.dim)

    BM25Index(chunks).save(s.bm25_path)
    log.info("bm25: %d chunks -> %s", len(chunks), s.bm25_path)


if __name__ == "__main__":
    main()
