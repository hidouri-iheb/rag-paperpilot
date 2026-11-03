from typing import Literal

from pydantic import BaseModel, computed_field


class Paper(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    year: int
    abstract: str
    area: str


class Page(BaseModel):
    arxiv_id: str
    page: int  # 1-indexed, like a pdf viewer
    text: str
    section: str | None = None  # section active when the page starts


class Chunk(BaseModel):
    chunk_id: str  # "{arxiv_id}:p{page}:c{idx}", unique
    arxiv_id: str
    title: str
    year: int
    section: str | None
    page: int
    text: str
    header: str  # "Title (Year) › Section › p.N"

    @property
    def indexed_text(self) -> str:
        """What gets embedded and keyword-indexed."""
        return f"{self.header}\n{self.text}"


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class Citation(BaseModel):
    chunk_id: str
    quote: str  # exact short quote copied from the chunk


class Answer(BaseModel):
    """What we ask Gemini to return."""

    answer: str
    citations: list[Citation]
    sufficient_evidence: bool


class VerifiedCitation(BaseModel):
    chunk_id: str
    arxiv_id: str
    title: str
    year: int
    page: int
    quote: str

    @computed_field
    @property
    def url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_id}#page={self.page}"


class FinalAnswer(BaseModel):
    """What the user sees: only citations that passed verification."""

    answer: str | None
    citations: list[VerifiedCitation]
    refused: bool
    reason: str | None = None
    dropped_citations: int = 0
    source: Literal["library", "general"] = "library"
