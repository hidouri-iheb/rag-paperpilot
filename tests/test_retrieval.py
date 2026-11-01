from paperpilot.config import Settings
from paperpilot.models import Chunk
from paperpilot.retrieval.fusion import rrf
from paperpilot.retrieval.retriever import Retriever


def test_rrf_hand_example():
    fused = rrf([["a", "b", "c"], ["b", "c", "d"]], k=60)
    assert [cid for cid, _ in fused] == ["b", "c", "a", "d"]
    assert abs(fused[0][1] - (1 / 62 + 1 / 61)) < 1e-12


def make_chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        arxiv_id="x",
        title="T",
        year=2020,
        section=None,
        page=1,
        text=f"text {cid}",
        header="T (2020) › p.1",
    )


CHUNKS = {cid: make_chunk(cid) for cid in "abcde"}


class FakeEmbedder:
    def embed_query(self, query):
        return [0.0]


class FakeStore:
    def search(self, vector, k):
        return [(CHUNKS[c], 0.9 - i * 0.1) for i, c in enumerate("abc")][:k]


class FakeBM25:
    def search(self, query, k):
        return [(c, 10.0 - i) for i, c in enumerate("dec")][:k]


class FakeReranker:
    def score(self, query, passages):
        return [0.99 if p.endswith("text e") else 0.1 for p in passages]


def make_retriever() -> Retriever:
    s = Settings(_env_file=None, final_top_k=3, vector_top_k=3, bm25_top_k=3, rerank_candidates=5)
    return Retriever(s, CHUNKS, FakeEmbedder(), FakeStore(), FakeBM25(), FakeReranker())


def ids(results):
    return [r.chunk.chunk_id for r in results]


def test_vector_only():
    results = make_retriever().retrieve("q", use_bm25=False, use_reranker=False)
    assert ids(results) == ["a", "b", "c"]


def test_hybrid_puts_chunk_found_by_both_first():
    results = make_retriever().retrieve("q", use_bm25=True, use_reranker=False)
    assert ids(results)[0] == "c"


def test_reranker_reorders_candidates():
    results = make_retriever().retrieve("q", use_bm25=True, use_reranker=True)
    assert ids(results)[0] == "e"
    assert all(0.0 <= r.score <= 1.0 for r in results)
