"""Print the CRAG misses from the latest answers_*.json, for the error analysis."""

import json
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).parent


def main() -> None:
    latest = sorted((EVAL_DIR / "results").glob("answers_*.json"))[-1]
    record = json.loads(latest.read_text("utf-8"))
    questions = yaml.safe_load((EVAL_DIR / "questions.yaml").read_text("utf-8"))["questions"]
    by_id = {q["id"]: q for q in questions}

    for row in record["rows"]["hybrid+rerank+crag"]:
        if "error" in row or not row["answerable"]:
            continue
        if not row["refused"] and row["cited_gold"]:
            continue
        q = by_id[row["id"]]
        gold = ", ".join(f"{g['arxiv_id']} p.{g['pages']}" for g in q["gold"])
        cited = ", ".join(f"{c['arxiv_id']} p.{c['page']}" for c in row["citations"]) or "none"
        print(f"\n{row['id']}  {q['question']}")
        print(f"  gold:   {gold}")
        print(f"  cited:  {cited}   (path: {row['path']})")
        print(f"  answer: {(row['answer'] or '')[:300]}")


if __name__ == "__main__":
    main()
