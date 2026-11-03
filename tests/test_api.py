from fastapi.testclient import TestClient

from paperpilot.api.main import create_app
from paperpilot.generation.llm import QuotaExhausted
from paperpilot.models import FinalAnswer, Paper, VerifiedCitation

PAPER = Paper(
    arxiv_id="1706.03762",
    title="Attention Is All You Need",
    authors=["Ashish Vaswani", "Noam Shazeer"],
    year=2017,
    abstract="...",
    area="transformers",
)


class FakePipeline:
    def __init__(self, error: Exception | None = None):
        self.papers = {PAPER.arxiv_id: PAPER}
        self.error = error

    def ask(self, question, allow_general=None):
        if self.error:
            raise self.error
        citation = VerifiedCitation(
            chunk_id="1706.03762:p3:c0",
            arxiv_id="1706.03762",
            title=PAPER.title,
            year=2017,
            page=3,
            quote="stack of N = 6 identical layers",
        )
        return FinalAnswer(answer="Six layers.", citations=[citation], refused=False), "direct"

    def close(self):
        pass


def client(error: Exception | None = None) -> TestClient:
    return TestClient(create_app(lambda: FakePipeline(error)))


def test_health():
    with client() as c:
        assert c.get("/health").json() == {"status": "ok", "papers": 1}


def test_papers():
    with client() as c:
        assert c.get("/papers").json()[0]["title"] == "Attention Is All You Need"


def test_ask_returns_verified_citation_with_authors_and_link():
    with client() as c:
        body = c.post("/ask", json={"question": "How many layers?"}).json()
    assert body["path"] == "direct" and not body["refused"]
    citation = body["citations"][0]
    assert citation["url"] == "https://arxiv.org/pdf/1706.03762#page=3"
    assert citation["authors"] == ["Ashish Vaswani", "Noam Shazeer"]


def test_too_short_question_is_422():
    with client() as c:
        assert c.post("/ask", json={"question": "a"}).status_code == 422


def test_quota_exhausted_is_friendly_503():
    with client(QuotaExhausted("Gemini free quota exhausted, try again later")) as c:
        response = c.post("/ask", json={"question": "How many layers?"})
    assert response.status_code == 503
    assert "quota" in response.json()["detail"]
