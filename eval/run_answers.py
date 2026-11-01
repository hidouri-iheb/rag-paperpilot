"""Answer eval: every question through 4 setups, every answer scored.

answered          share of answerable questions that got an answer
cited_gold        share of answerable questions with a verified citation on a gold page (+-1)
refusal_acc       share of unanswerable questions that were refused
citation_validity verified citations / all citations the model produced
faithfulness      share of answers the judge finds fully supported by their sources

All LLM calls are cached: if the quota or Google's servers fail, just run it again later.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

import yaml
from pydantic import BaseModel

from paperpilot.config import get_settings
from paperpilot.generation.generator import generate_answer
from paperpilot.generation.llm import LLM, LLMUnavailable
from paperpilot.graph.crag import build_crag, run_crag
from paperpilot.models import FinalAnswer
from paperpilot.retrieval.retriever import load_retriever

EVAL_DIR = Path(__file__).parent
CONFIGS = {
    "vector": {"use_bm25": False, "use_reranker": False, "crag": False},
    "hybrid": {"use_bm25": True, "use_reranker": False, "crag": False},
    "hybrid+rerank": {"use_bm25": True, "use_reranker": True, "crag": False},
    "hybrid+rerank+crag": {"use_bm25": True, "use_reranker": True, "crag": True},
}

JUDGE_PROMPT = """\
You are checking an answer written by a research assistant.

Answer:
{answer}

Sources the answer cites:
{sources}

Is every factual claim in the answer supported by these sources? Ignore style and wording.
Set faithful to true only if nothing in the answer goes beyond the sources, and list any
unsupported claims.
"""


class Judgement(BaseModel):
    faithful: bool
    unsupported_claims: list[str]


class CountingLLM:
    """Counts every generate() call, cached or not, so calls per question are comparable."""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.requests = 0

    def generate(self, model, prompt, schema):
        self.requests += 1
        return self.llm.generate(model, prompt, schema)


def cited_gold(final: FinalAnswer, gold: list[dict]) -> bool:
    return any(
        c.arxiv_id == g["arxiv_id"] and any(abs(c.page - p) <= 1 for p in g["pages"])
        for c in final.citations
        for g in gold
    )


def rate(part: float, whole: int) -> float | None:
    return part / whole if whole else None


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    judged = [r for r in ok if r["faithful"] is not None]
    produced = sum(r["valid_citations"] + r["dropped_citations"] for r in ok)
    return {
        "answered": rate(sum(not r["refused"] for r in answerable), len(answerable)),
        "cited_gold": rate(sum(r["cited_gold"] for r in answerable), len(answerable)),
        "refusal_acc": rate(sum(r["refused"] for r in unanswerable), len(unanswerable)),
        "citation_validity": rate(sum(r["valid_citations"] for r in ok), produced),
        "faithfulness": rate(sum(r["faithful"] for r in judged), len(judged)),
        "llm_calls_per_q": mean(r["llm_calls"] for r in ok) if ok else None,
        "paths": dict(Counter(r["path"] for r in ok)),
        "errors": len(rows) - len(ok),
    }


def fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:6.1%}"


def main() -> None:
    s = get_settings()
    questions = yaml.safe_load((EVAL_DIR / "questions.yaml").read_text("utf-8"))["questions"]
    llm = LLM.from_settings(s)
    counted = CountingLLM(llm)
    retriever = load_retriever(s)
    crag = build_crag(retriever, counted, s)

    def judge(final: FinalAnswer) -> Judgement:
        sources = "\n\n".join(
            f"[{c.title}, p.{c.page}]\n{retriever.chunks[c.chunk_id].text}" for c in final.citations
        )
        prompt = JUDGE_PROMPT.format(answer=final.answer, sources=sources)
        return llm.generate(s.gemini_light_model, prompt, Judgement)

    results: dict[str, list[dict]] = {}
    try:
        for name, cfg in CONFIGS.items():
            rows = []
            for q in questions:
                before = counted.requests
                try:
                    if cfg["crag"]:
                        final, path = run_crag(crag, q["question"])
                    else:
                        chunks = retriever.retrieve(
                            q["question"],
                            use_bm25=cfg["use_bm25"],
                            use_reranker=cfg["use_reranker"],
                        )
                        final = generate_answer(counted, s, q["question"], chunks)
                        path = "refused" if final.refused else "direct"
                    judgement = None if final.refused else judge(final)
                except LLMUnavailable as exc:
                    print(f"  {name} {q['id']}: {exc}")
                    rows.append({"id": q["id"], "error": str(exc)})
                    continue
                rows.append(
                    {
                        "id": q["id"],
                        "answerable": q["answerable"],
                        "path": path,
                        "refused": final.refused,
                        "cited_gold": cited_gold(final, q["gold"]),
                        "valid_citations": len(final.citations),
                        "dropped_citations": final.dropped_citations,
                        "faithful": None if judgement is None else judgement.faithful,
                        "unsupported_claims": []
                        if judgement is None
                        else judgement.unsupported_claims,
                        "llm_calls": counted.requests - before,
                        "answer": final.answer,
                        "citations": [c.model_dump() for c in final.citations],
                    }
                )
            results[name] = rows
            print(f"done: {name}")
    finally:
        retriever.close()
        llm.close()

    summaries = {name: summarize(rows) for name, rows in results.items()}
    print(
        "\n| config | answered | cited gold | refusal acc | cit. validity | faithful | calls/q |"
    )
    print("|---|---|---|---|---|---|---|")
    for name, m in summaries.items():
        calls = "n/a" if m["llm_calls_per_q"] is None else f"{m['llm_calls_per_q']:.2f}"
        print(
            f"| {name} | {fmt(m['answered'])} | {fmt(m['cited_gold'])} | {fmt(m['refusal_acc'])}"
            f" | {fmt(m['citation_validity'])} | {fmt(m['faithfulness'])} | {calls} |"
        )
    for name, m in summaries.items():
        print(f"{name}: paths {m['paths']}, errors {m['errors']}")

    crag_rows = results.get("hybrid+rerank+crag", [])
    misses = [
        r["id"]
        for r in crag_rows
        if "error" not in r and r["answerable"] and (r["refused"] or not r["cited_gold"])
    ]
    print("crag misses (for the error analysis):", misses or "none")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"answers_{stamp}.json"
    record = {
        "date": stamp,
        "main_model": s.gemini_main_model,
        "light_model": s.gemini_light_model,
        "grade_high": s.grade_high,
        "grade_low": s.grade_low,
        "summary": summaries,
        "rows": results,
    }
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
