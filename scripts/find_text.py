"""Find chunks containing a phrase: uv run python scripts/find_text.py "phrase" [arxiv_id]"""

import sys

from paperpilot.config import get_settings
from paperpilot.models import Chunk


def main() -> None:
    phrase = sys.argv[1].lower()
    paper = sys.argv[2] if len(sys.argv) > 2 else None
    lines = (get_settings().processed_dir / "chunks.jsonl").read_text("utf-8").splitlines()
    for line in lines:
        chunk = Chunk.model_validate_json(line)
        if paper and chunk.arxiv_id != paper:
            continue
        text = chunk.text.lower()
        if phrase in text:
            i = text.index(phrase)
            print(chunk.chunk_id, "|", chunk.section)
            print("   ...", chunk.text[max(0, i - 100) : i + 150].replace("\n", " "), "...")


if __name__ == "__main__":
    main()
