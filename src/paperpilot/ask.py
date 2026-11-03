"""Ask PaperPilot from the terminal: uv run python -m paperpilot.ask "your question" """

import logging
import sys

from paperpilot.config import get_settings
from paperpilot.generation.generator import generate_answer
from paperpilot.generation.llm import LLM, LLMUnavailable
from paperpilot.graph.crag import build_crag, run_crag
from paperpilot.retrieval.retriever import load_retriever

BADGES = {
    "direct": "✅ answered directly",
    "corrected": "🔄 answered after a corrected search",
    "general": "◇ general knowledge, not from the library",
    "refused": "⚠️ not enough evidence",
}


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit('usage: uv run python -m paperpilot.ask "your question"')
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("paperpilot.graph").setLevel(logging.INFO)  # show the crag decisions

    s = get_settings()
    llm = LLM.from_settings(s)
    retriever = load_retriever(s)
    try:
        if s.use_crag:
            final, path = run_crag(build_crag(retriever, llm, s), question)
        else:
            final = generate_answer(llm, s, question, retriever.retrieve(question))
            path = "refused" if final.refused else "direct"
    except LLMUnavailable as exc:
        raise SystemExit(f"\n⚠️ {exc}") from None
    finally:
        retriever.close()
        llm.close()

    print(f"\n{BADGES[path]}\n\n{final.answer}")
    if path == "general":
        print(
            "\n(nothing in the library covers this; this answer has no sources and is unverified)"
        )
    for i, c in enumerate(final.citations, start=1):
        print(f'\n[{i}] {c.title} ({c.year}), p.{c.page}\n    "{c.quote}"\n    {c.url}')
    print(f"\n(gemini calls: {llm.api_calls}, dropped citations: {final.dropped_citations})")


if __name__ == "__main__":
    main()
