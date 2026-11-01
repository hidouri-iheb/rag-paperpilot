from paperpilot.models import Chunk
from paperpilot.retrieval.bm25 import BM25Index, tokenize


def chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        arxiv_id="x",
        title="T",
        year=2020,
        section=None,
        page=1,
        text=text,
        header="T (2020) › p.1",
    )


def test_tokenizer_keeps_hyphenated_terms_and_drops_stopwords():
    assert tokenize("The self-attention of GPT-3") == ["self-attention", "gpt-3"]


def test_bm25_ranks_matching_chunk_first():
    index = BM25Index(
        [
            chunk("a", "reinforcement learning with policy gradients"),
            chunk("b", "dense passage retrieval for question answering"),
            chunk("c", "image segmentation with convolutional networks"),
        ]
    )
    assert index.search("passage retrieval", k=1)[0][0] == "b"
