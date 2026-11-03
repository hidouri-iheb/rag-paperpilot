"""Build the whole library in one go: download -> parse -> chunk -> index.

uv run python -m paperpilot.ingest.build_library
"""

from paperpilot.ingest import build_index, chunk, download, parse


def main() -> None:
    for step in (download, parse, chunk, build_index):
        print(f"\n=== {step.__name__.rsplit('.', 1)[-1]} ===")
        step.main()


if __name__ == "__main__":
    main()
