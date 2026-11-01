"""Turn eval/questions_draft.yaml into eval/questions.yaml with gold pages filled in.

Gold pages = pages of the question's paper whose chunks contain one of the hints.
Review the output against the PDFs: a page that mentions a phrase isn't always an answer.
"""

from collections import defaultdict
from pathlib import Path

import yaml

from paperpilot.config import get_settings
from paperpilot.models import Chunk

EVAL_DIR = Path(__file__).parent


def main() -> None:
    draft = yaml.safe_load((EVAL_DIR / "questions_draft.yaml").read_text("utf-8"))["questions"]
    lines = (get_settings().processed_dir / "chunks.jsonl").read_text("utf-8").splitlines()
    chunks = [Chunk.model_validate_json(line) for line in lines]

    by_paper: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_paper[c.arxiv_id].append(c)

    questions = []
    for q in draft:
        entry = {"id": q["id"], "question": q["question"], "category": q["category"]}
        if q.get("answerable", True) is False:
            entry.update(answerable=False, gold=[])
        else:
            hints = [h.lower() for h in q["hints"]]
            pages = sorted(
                {c.page for c in by_paper[q["paper"]] if any(h in c.text.lower() for h in hints)}
            )
            entry.update(answerable=True, gold=[{"arxiv_id": q["paper"], "pages": pages}])
            flag = "  <-- NO MATCH, find the page by hand" if not pages else ""
            print(f"{q['id']}  {q['paper']}  pages {pages}{flag}")
        questions.append(entry)

    out = EVAL_DIR / "questions.yaml"
    out.write_text(
        yaml.safe_dump({"questions": questions}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"\nwrote {len(questions)} questions -> {out}")


if __name__ == "__main__":
    main()
