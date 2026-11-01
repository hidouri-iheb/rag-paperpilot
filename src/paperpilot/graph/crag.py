"""Corrective RAG as an explicit LangGraph.

retrieve -> grade -> generate -> verify            path: direct (or corrected after a rewrite)
              |-> rewrite -> retrieve -> grade      at most max_retries rewrites
              |-> refuse                            path: refused
"""

import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from paperpilot.config import Settings
from paperpilot.generation.generator import draft_answer
from paperpilot.generation.llm import LLM
from paperpilot.generation.prompts import GRADE_PROMPT, REWRITE_PROMPT, format_context
from paperpilot.generation.verify import REFUSAL, verify
from paperpilot.models import Answer, FinalAnswer, RetrievedChunk

log = logging.getLogger(__name__)


class ChunkGrades(BaseModel):
    relevant_chunk_ids: list[str]


class RewrittenQuery(BaseModel):
    query: str


class CragState(TypedDict, total=False):
    question: str  # what the user asked, used for grading and answering
    query: str  # what we search with, may get rewritten
    chunks: list[RetrievedChunk]
    verdict: str  # "relevant" or "weak"
    retries: int
    draft: Answer
    final: FinalAnswer
    path: str  # "direct", "corrected" or "refused"


def build_crag(retriever, llm: LLM, s: Settings):
    def retrieve(state: CragState) -> CragState:
        return {"chunks": retriever.retrieve(state["query"])}

    def grade(state: CragState) -> CragState:
        chunks = state["chunks"]
        top = max((c.score for c in chunks), default=0.0)
        if top >= s.grade_high:
            log.info("grade: relevant, top %.3f (no llm call)", top)
            return {"verdict": "relevant"}
        if top <= s.grade_low:
            log.info("grade: weak, top %.3f (no llm call)", top)
            return {"verdict": "weak"}
        prompt = GRADE_PROMPT.format(question=state["question"], context=format_context(chunks))
        grades = llm.generate(s.gemini_light_model, prompt, ChunkGrades)
        kept = [c for c in chunks if c.chunk.chunk_id in set(grades.relevant_chunk_ids)]
        log.info("grade: llm kept %d/%d chunks, top %.3f", len(kept), len(chunks), top)
        if kept:
            return {"verdict": "relevant", "chunks": kept}
        return {"verdict": "weak"}

    def after_grade(state: CragState) -> str:
        if state["verdict"] == "relevant":
            return "generate"
        return "rewrite" if state["retries"] < s.max_retries else "refuse"
    
    def after_verify(state: CragState) -> str:
        # the answer step can also find the evidence insufficient: try one better query first
        if state["final"].refused and state["retries"] < s.max_retries:
            log.info("verify: answer refused, rewriting the query")
            return "rewrite"
        return END
    
    def rewrite(state: CragState) -> CragState:
        prompt = REWRITE_PROMPT.format(question=state["question"])
        query = llm.generate(s.gemini_light_model, prompt, RewrittenQuery).query
        log.info("rewrite: %r -> %r", state["query"], query)
        return {"query": query, "retries": state["retries"] + 1}

    def generate(state: CragState) -> CragState:
        return {"draft": draft_answer(llm, s, state["question"], state["chunks"])}

    def verify_draft(state: CragState) -> CragState:
        final = verify(state["draft"], state["chunks"], s.quote_match_threshold, s.max_quote_words)
        if final.refused:
            path = "refused"
        elif state["retries"]:
            path = "corrected"
        else:
            path = "direct"
        return {"final": final, "path": path}

    def refuse(state: CragState) -> CragState:
        final = FinalAnswer(
            answer=REFUSAL,
            citations=[],
            refused=True,
            reason="retrieval stayed weak after rewriting the query",
        )
        return {"final": final, "path": "refused"}

    graph = StateGraph(CragState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify_draft)
    graph.add_node("refuse", refuse)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", after_grade, {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"}
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", after_verify, {"rewrite": "rewrite", END: END})
    graph.add_edge("refuse", END)
    graph.add_edge("refuse", END)
    return graph.compile()


def run_crag(app, question: str) -> tuple[FinalAnswer, str]:
    state = app.invoke({"question": question, "query": question, "retries": 0})
    return state["final"], state["path"]
