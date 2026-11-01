"""Retrieval eval: Hit@5 and MRR for vector-only, hybrid, and hybrid + rerank."""

import json
from datetime import datetime
from pathlib import Path

import yaml

from paperpilot.config import get_settings
from paperpilot.models import RetrievedChunk
from paperpilot.retrieval.retriever import load_retriever

EVAL_DIR = Path(__file__).parent
CONFIGS = {
    "vector": {"use_bm25": False, "use_reranker": False},
    "hybrid": {"use_bm25": True, "use_reranker": False},
    "hybrid+rerank": {"use_bm25": True, "use_reranker": True},
}


def on_gold_page(result: RetrievedChunk, gold: list[dict]) -> bool:
    return any(
        result.chunk.arxiv_id == g["arxiv_id"]
        and any(abs(result.chunk.page - p) <= 1 for p in g["pages"])
        for g in gold
    )


def first_hit_rank(results: list[RetrievedChunk], gold: list[dict]) -> int | None:
    """1-based rank of the first chunk on a gold page (within one page), or None."""
    for rank, result in enumerate(results, start=1):
        if on_gold_page(result, gold):
            return rank
    return None


def main() -> None:
    s = get_settings()
    all_questions = yaml.safe_load((EVAL_DIR / "questions.yaml").read_text("utf-8"))["questions"]
    questions = [q for q in all_questions if q["answerable"]]

    retriever = load_retriever(s)
    summary: dict[str, dict] = {}
    ranks: dict[str, dict] = {}
    try:
        for name, flags in CONFIGS.items():
            ranks[name] = {
                q["id"]: first_hit_rank(retriever.retrieve(q["question"], **flags), q["gold"])
                for q in questions
            }
            found = [r for r in ranks[name].values() if r is not None]
            summary[name] = {
                "hit@5": len(found) / len(questions),
                "mrr": sum(1 / r for r in found) / len(questions),
            }
            print(f"done: {name}")
    finally:
        retriever.close()

    print(f"\n{'config':<15}{'Hit@5':>8}{'MRR':>8}")
    for name, m in summary.items():
        print(f"{name:<15}{m['hit@5']:>8.1%}{m['mrr']:>8.3f}")
    missed = [qid for qid, r in ranks["hybrid+rerank"].items() if r is None]
    print("missed by hybrid+rerank:", missed or "none")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    record = {
        "date": stamp,
        "n_questions": len(questions),
        "embedding_model": s.embedding_model,
        "reranker_model": s.reranker_model,
        "settings": {
            k: getattr(s, k)
            for k in ("vector_top_k", "bm25_top_k", "rrf_k", "rerank_candidates", "final_top_k")
        },
        "summary": summary,
        "ranks": ranks,
    }
    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"retrieval_{stamp}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
