"""Top reranker score per eval question, answerable vs not. Used to pick the CRAG thresholds."""

from pathlib import Path

import yaml

from paperpilot.config import get_settings
from paperpilot.retrieval.retriever import load_retriever

EVAL_DIR = Path(__file__).parent


def main() -> None:
    s = get_settings()
    questions = yaml.safe_load((EVAL_DIR / "questions.yaml").read_text("utf-8"))["questions"]
    retriever = load_retriever(s)
    rows = []
    try:
        for q in questions:
            results = retriever.retrieve(q["question"], use_bm25=True, use_reranker=True)
            rows.append((q["answerable"], results[0].score, q["id"], q["question"]))
    finally:
        retriever.close()

    for answerable in (True, False):
        print(f"\n=== {'answerable' if answerable else 'NOT answerable'} ===")
        group = sorted((r for r in rows if r[0] == answerable), key=lambda r: r[1])
        for _, score, qid, text in group:
            print(f"{score:6.3f}  {qid}  {text}")


if __name__ == "__main__":
    main()
