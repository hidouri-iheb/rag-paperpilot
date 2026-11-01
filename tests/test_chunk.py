from paperpilot.ingest.chunk import make_chunks
from paperpilot.models import Page, Paper

PAPER = Paper(
    arxiv_id="1234.5678", title="Test Paper", authors=["A"], year=2020, abstract="", area="rl"
)


def para(word: str, i: int) -> str:
    return " ".join([f"{word} sentence {i} about retrieval."] * 6)


def page(n: int, word: str) -> Page:
    text = "\n\n".join(para(word, i) for i in range(12))
    return Page(arxiv_id=PAPER.arxiv_id, page=n, text=text)


def test_ids_unique_and_sizes_within_limit():
    chunks = make_chunks(page(1, "alpha"), PAPER, size=300, overlap=50, max_size=400)
    ids = [c.chunk_id for c in chunks]
    assert len(chunks) > 1 and len(ids) == len(set(ids))
    assert all(0 < len(c.text) <= 400 for c in chunks)


def test_chunks_never_cross_pages():
    chunks = make_chunks(page(1, "alpha"), PAPER, 300, 50, 400)
    chunks += make_chunks(page(2, "beta"), PAPER, 300, 50, 400)
    for c in chunks:
        assert ("beta" if c.page == 1 else "alpha") not in c.text


def test_section_and_header():
    text = "# 2 Method\n\n" + para("x", 0)
    p = Page(arxiv_id=PAPER.arxiv_id, page=3, text=text, section="1 Intro")
    c = make_chunks(p, PAPER, 300, 50, 400)[0]
    assert c.section == "2 Method"
    assert c.header == "Test Paper (2020) › 2 Method › p.3"
