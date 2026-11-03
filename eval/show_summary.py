"""Print the summary table of the latest answers_*.json without re-running anything."""

import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent


def fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:6.1%}"


def main() -> None:
    latest = sorted((EVAL_DIR / "results").glob("answers_*.json"))[-1]
    summary = json.loads(latest.read_text("utf-8"))["summary"]
    print(f"file: {latest.name}\n")
    print("| config | answered | cited gold | refusal acc | cit. validity | faithful | calls/q |")
    print("|---|---|---|---|---|---|---|")
    for name, m in summary.items():
        calls = "n/a" if m["llm_calls_per_q"] is None else f"{m['llm_calls_per_q']:.2f}"
        print(
            f"| {name} | {fmt(m['answered'])} | {fmt(m['cited_gold'])} | {fmt(m['refusal_acc'])}"
            f" | {fmt(m['citation_validity'])} | {fmt(m['faithfulness'])} | {calls} |"
        )
    for name, m in summary.items():
        print(f"{name}: paths {m['paths']}, errors {m['errors']}")


if __name__ == "__main__":
    main()
