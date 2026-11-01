"""Split pages into overlapping chunks that never cross a page boundary."""

import logging
import re
from collections import Counter
from collections.abc import Callable
from statistics import mean

from transformers import AutoTokenizer

from paperpilot.config import get_settings
from paperpilot.ingest.parse import HEADING, plain
from paperpilot.models import Chunk, Page, Paper

log = logging.getLogger(__name__)

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
MIN_CHARS = 50  # drops leftovers like a lone "Figure 3"


def split_long(text: str, limit: int) -> list[str]:
    """Break an oversized paragraph by sentences, then by words if needed."""
    pieces: list[str] = []
    for sentence in SENTENCE_END.split(text):
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    return pieces


def split_to_fit(body: str, fits: Callable[[str], bool]) -> list[str]:
    """Halve body at a sentence or word boundary until every part fits."""
    if fits(body) or len(body) < 2 * MIN_CHARS:
        return [body]
    mid = len(body) // 2
    cut = body.rfind(". ", 0, mid) + 1 or body.rfind(" ", 0, mid)
    if cut <= 0:
        cut = mid
    return split_to_fit(body[:cut].strip(), fits) + split_to_fit(body[cut:].strip(), fits)


def overlap_tail(text: str, size: int) -> str:
    """Last ~size chars of text, starting at a word boundary."""
    if len(text) <= size:
        return text
    tail = text[-size:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def make_chunks(
    page: Page,
    paper: Paper,
    size: int,
    overlap: int,
    max_size: int,
    count_tokens: Callable[[str], int] | None = None,
    max_tokens: int = 512,
) -> list[Chunk]:
    if size + overlap + 1 > max_size:
        raise ValueError("chunk_size + chunk_overlap must stay under chunk_max_size")

    def header_for(sec: str | None) -> str:
        where = f" › {sec}" if sec else ""
        return f"{paper.title} ({paper.year}){where} › p.{page.page}"

    def fits(text: str, sec: str | None) -> bool:
        if count_tokens is None:
            return True
        return count_tokens(header_for(sec) + "\n" + text) <= max_tokens

    # paragraphs (split further if too long), each tagged with its section
    section = page.section
    units: list[tuple[str, str | None]] = []
    for para in page.text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if HEADING.match(para.splitlines()[0]):
            section = plain(para.splitlines()[0])
        pieces = [para] if len(para) <= size else split_long(para, size)
        units.extend((piece, section) for piece in pieces)

    # pack units up to `size` characters
    bodies: list[tuple[str, str | None]] = []
    body, body_section = "", None
    for piece, sec in units:
        if body and len(body) + 2 + len(piece) > size:
            bodies.append((body, body_section))
            body = ""
        if not body:
            body_section = sec
        body = f"{body}\n\n{piece}" if body else piece
    if body:
        bodies.append((body, body_section))

    # dense text (math, tables) can still be too many tokens: split until it fits
    fitted: list[tuple[str, str | None]] = []
    for body, sec in bodies:
        for part in split_to_fit(body, lambda t, sec=sec: fits(t, sec)):
            fitted.append((part, sec))
    bodies = [(b, s) for b, s in fitted if len(b.strip()) >= MIN_CHARS]

    chunks = []
    for idx, (body, sec) in enumerate(bodies):
        text = body if idx == 0 else f"{overlap_tail(bodies[idx - 1][0], overlap)} {body}"
        if not fits(text, sec):
            text = body  # overlap would push it past the token limit
        chunks.append(
            Chunk(
                chunk_id=f"{page.arxiv_id}:p{page.page}:c{idx}",
                arxiv_id=page.arxiv_id,
                title=paper.title,
                year=paper.year,
                section=sec,
                page=page.page,
                text=text,
                header=header_for(sec),
            )
        )
    return chunks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = get_settings()
    tokenizer = AutoTokenizer.from_pretrained(s.embedding_model)

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, verbose=False)["input_ids"])

    papers_file = s.processed_dir / "papers.jsonl"
    pages_file = s.processed_dir / "pages.jsonl"
    papers = {
        p.arxiv_id: p
        for p in map(Paper.model_validate_json, papers_file.read_text("utf-8").splitlines())
    }
    pages = [Page.model_validate_json(line) for line in pages_file.read_text("utf-8").splitlines()]

    chunks: list[Chunk] = []
    for page in pages:
        chunks += make_chunks(
            page,
            papers[page.arxiv_id],
            s.chunk_size,
            s.chunk_overlap,
            s.chunk_max_size,
            count_tokens=count_tokens,
            max_tokens=s.embedding_max_tokens,
        )

    out = s.processed_dir / "chunks.jsonl"
    out.write_text("".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8")

    lengths = [len(c.text) for c in chunks]
    tokens = [count_tokens(c.header + "\n" + c.text) for c in chunks]
    per_paper = Counter(c.arxiv_id for c in chunks)
    log.info("chunks: %d", len(chunks))
    log.info("length: avg %d, min %d, max %d chars", mean(lengths), min(lengths), max(lengths))
    over = sum(t > s.embedding_max_tokens for t in tokens)
    log.info("tokens: max %d, over limit %d", max(tokens), over)
    log.info(
        "per paper: min %d, max %d (%s)",
        min(per_paper.values()),
        max(per_paper.values()),
        per_paper.most_common(1)[0][0],
    )
    log.info("unique ids: %s", len({c.chunk_id for c in chunks}) == len(chunks))


if __name__ == "__main__":
    main()
