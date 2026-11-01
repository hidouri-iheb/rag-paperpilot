from sentence_transformers import SentenceTransformer

from paperpilot.models import Chunk


class Embedder:
    def __init__(self, model_name: str, query_prefix: str, max_tokens: int):
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_tokens
        self.query_prefix = query_prefix

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        texts = [c.indexed_text for c in chunks]  # no prefix on passages
        vectors = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode(self.query_prefix + query, normalize_embeddings=True).tolist()
