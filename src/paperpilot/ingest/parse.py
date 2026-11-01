"""Turn every PDF into clean markdown pages with 1-indexed page numbers."""

import logging
import re
from collections import Counter

import pymupdf4llm

from paperpilot.config import get_settings
from paperpilot.models import Page

log = logging.getLogger(__name__)

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
HEADING = re.compile(r"^#{1,6}\s+.+$")
REFERENCES = re.compile(r"^(\d+\.?\s*)?(references|bibliography)$", re.IGNORECASE)


def plain(line: str) -> str:
    """'### 3.1 Encoder' -> '3.1 Encoder'"""
    return re.sub(r"[#*_`]", "", line).strip()


def clean_text(text: str) -> str:
    for lig, rep in LIGATURES.items():
        text = text.replace(lig, rep)
    text = re.sub(r"(\w)-\s*\n\s*([a-z])", r"\1\2", text)  # retriev-\nal -> retrieval
    text = text.replace("**", "")
    text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"\1", text)  # _N_ -> N
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_boilerplate(texts: list[str]) -> set[str]:
    """Short lines repeated at the top/bottom of at least half the pages."""
    if len(texts) < 4:
        return set()
    counts: Counter[str] = Counter()
    for text in texts:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        counts.update({ln for ln in lines[:2] + lines[-2:] if len(ln) < 100})
    return {ln for ln, n in counts.items() if n >= len(texts) / 2}


def process_paper(arxiv_id: str, raw_pages: list[tuple[int, str]]) -> list[Page]:
    texts = [(number, clean_text(text)) for number, text in raw_pages]
    boilerplate = find_boilerplate([t for _, t in texts])

    pages: list[Page] = []
    section: str | None = None
    in_refs = False
    for number, text in texts:
        start_section = section
        kept = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped in boilerplate or stripped.isdigit():
                continue
            label = plain(stripped)
            if REFERENCES.match(label):
                in_refs = True
                continue
            if HEADING.match(stripped):
                in_refs = False  # any new heading ends the reference list
                section = label
            if not in_refs:
                kept.append(line)
        body = "\n".join(kept).strip()
        if body:
            pages.append(Page(arxiv_id=arxiv_id, page=number, text=body, section=start_section))
    return pages


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    out = settings.processed_dir / "pages.jsonl"
    total = failed = 0
    with out.open("w", encoding="utf-8") as f:
        for pdf in sorted(settings.pdf_dir.glob("*.pdf")):
            try:
                chunks = pymupdf4llm.to_markdown(str(pdf), page_chunks=True)
            except Exception as exc:  # one bad pdf shouldn't stop the run
                failed += 1
                log.error("failed %s: %s", pdf.stem, exc)
                continue
            raw = [(c["metadata"]["page_number"], c["text"]) for c in chunks]
            pages = process_paper(pdf.stem, raw)
            for page in pages:
                f.write(page.model_dump_json() + "\n")
            total += len(pages)
            log.info("%s: kept %d/%d pages", pdf.stem, len(pages), len(raw))
    log.info("done: %d pages, %d failed pdfs -> %s", total, failed, out)


if __name__ == "__main__":
    main()
