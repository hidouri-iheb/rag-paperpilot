from paperpilot.generation.verify import verify
from paperpilot.models import Answer, Chunk, Citation, RetrievedChunk

TEXT = (
    "The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers."
)
CHUNK = RetrievedChunk(
    chunk=Chunk(
        chunk_id="1706.03762:p3:c0",
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        year=2017,
        section="3.1 Encoder and Decoder Stacks",
        page=3,
        text=TEXT,
        header="Attention Is All You Need (2017) › p.3",
    ),
    score=0.9,
)


def run(quote: str, chunk_id: str = "1706.03762:p3:c0", sufficient: bool = True):
    answer = Answer(
        answer="The encoder has six layers.",
        citations=[Citation(chunk_id=chunk_id, quote=quote)],
        sufficient_evidence=sufficient,
    )
    return verify(answer, [CHUNK], threshold=90, max_words=30)


def test_exact_quote_is_kept():
    result = run("stack of N = 6 identical layers")
    assert not result.refused
    assert result.citations[0].page == 3
    assert result.citations[0].url == "https://arxiv.org/pdf/1706.03762#page=3"


def test_near_match_with_curly_quotes_is_kept():
    result = run("\u201cstack of N=6 identical layers\u201d")
    assert not result.refused and len(result.citations) == 1


def test_fake_quote_is_caught():
    result = run("the encoder uses twelve recurrent LSTM layers")
    assert result.refused and result.citations == []
    assert result.dropped_citations == 1


def test_wrong_chunk_id_is_caught():
    result = run("stack of N = 6 identical layers", chunk_id="9999.99999:p1:c0")
    assert result.refused


def test_insufficient_evidence_means_refusal():
    assert run("stack of N = 6 identical layers", sufficient=False).refused
