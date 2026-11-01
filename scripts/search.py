"""Compare vector-only with hybrid + rerank: uv run python scripts/search.py "question" """

import sys

from paperpilot.config import get_settings
from paperpilot.retrieval.retriever import load_retriever

MODES = [
    ("vector only", {"use_bm25": False, "use_reranker": False}),
    ("hybrid + rerank", {"use_bm25": True, "use_reranker": True}),
]


def main() -> None:
    query = " ".join(sys.argv[1:]) or "What dataset was CLIP trained on?"
    retriever = load_retriever(get_settings())
    try:
        for label, flags in MODES:
            print(f"\n=== {label} ===")
            for r in retriever.retrieve(query, **flags):
                print(f"{r.score:7.3f}  {r.chunk.header}")
                print("         " + r.chunk.text[:160].replace("\n", " "))
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
