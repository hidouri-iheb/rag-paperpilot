"""FastAPI backend. Owns the models and the index; the UI only talks to it over HTTP.

Run: uv run uvicorn paperpilot.api.main:app
"""

import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from paperpilot.config import get_settings
from paperpilot.generation.llm import LLM, LLMUnavailable
from paperpilot.graph.crag import build_crag, run_crag
from paperpilot.models import FinalAnswer, Paper, VerifiedCitation
from paperpilot.retrieval.retriever import load_retriever


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    allow_general: bool | None = None  # None: use the server's default setting


class CitationOut(VerifiedCitation):
    authors: list[str]


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    path: Literal["direct", "corrected", "general", "refused"]
    source: Literal["library", "general"]
    refused: bool
    duration_ms: float


class Pipeline:
    """Everything expensive to load, created once at startup."""

    def __init__(self):
        s = get_settings()
        needed = [s.qdrant_path, s.bm25_path, s.processed_dir / "chunks.jsonl"]
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            raise RuntimeError(
                "Library not built, missing: "
                + ", ".join(missing)
                + ". Build it with: uv run python -m paperpilot.ingest.build_library"
            )
        self.retriever = load_retriever(s)
        self.llm = LLM.from_settings(s)
        self.crag = build_crag(self.retriever, self.llm, s)
        lines = (s.processed_dir / "papers.jsonl").read_text("utf-8").splitlines()
        self.papers = {p.arxiv_id: p for p in map(Paper.model_validate_json, lines)}

    def ask(self, question: str, allow_general: bool | None = None) -> tuple[FinalAnswer, str]:
        return run_crag(self.crag, question, allow_general)

    def close(self) -> None:
        self.retriever.close()
        self.llm.close()


def create_app(pipeline_factory: Callable[[], Pipeline] = Pipeline) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline_factory()
        try:
            yield
        finally:
            app.state.pipeline.close()

    app = FastAPI(title="PaperPilot", version="0.2.0", lifespan=lifespan)
    lock = threading.Lock()  # qdrant local mode and the cpu reranker: one request at a time

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "papers": len(app.state.pipeline.papers)}

    @app.get("/papers", response_model=list[Paper])
    def papers() -> list[Paper]:
        return sorted(app.state.pipeline.papers.values(), key=lambda p: (p.area, p.year))

    # plain def on purpose: fastapi runs it in a thread pool, so the event loop never blocks
    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        start = time.perf_counter()
        pipeline = app.state.pipeline
        try:
            with lock:
                final, path = pipeline.ask(request.question.strip(), request.allow_general)
        except LLMUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        citations = [
            CitationOut(
                **c.model_dump(exclude={"url"}),
                authors=pipeline.papers[c.arxiv_id].authors
                if c.arxiv_id in pipeline.papers
                else [],
            )
            for c in final.citations
        ]
        return AskResponse(
            answer=final.answer or "",
            citations=citations,
            path=path,
            source=final.source,
            refused=final.refused,
            duration_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    return app


app = create_app()
