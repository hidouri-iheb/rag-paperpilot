"""Check every citation against its source chunk; drop the ones that don't match."""

import logging
import re
import unicodedata

from rapidfuzz import fuzz

from paperpilot.models import Answer, FinalAnswer, RetrievedChunk, VerifiedCitation

log = logging.getLogger(__name__)

REFUSAL = "Not enough evidence in the library to answer this question."
MIN_QUOTE_WORDS = 3  # "the model" would match almost anything

_PUNCT = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}
)


def normalize(text: str) -> str:
    """For matching only: same quotes, dashes, case and spacing on both sides."""
    text = unicodedata.normalize("NFKC", text).translate(_PUNCT).lower()
    text = re.sub(r"[*_#`]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def verify(
    answer: Answer, chunks: list[RetrievedChunk], threshold: float, max_words: int
) -> FinalAnswer:
    by_id = {r.chunk.chunk_id: r.chunk for r in chunks}
    kept: list[VerifiedCitation] = []
    seen: set[tuple[str, str]] = set()
    dropped = 0

    for citation in answer.citations:
        chunk = by_id.get(citation.chunk_id)
        quote = normalize(citation.quote).strip("\"'. ")
        n_words = len(quote.split())

        if chunk is None:
            problem = "chunk was not in the context"
        elif not MIN_QUOTE_WORDS <= n_words <= max_words:
            problem = f"quote has {n_words} words"
        else:
            score = fuzz.partial_ratio(quote, normalize(chunk.text))
            problem = None if score >= threshold else f"match {score:.0f} < {threshold:.0f}"

        if problem:
            dropped += 1
            log.info("dropped citation %s: %s", citation.chunk_id, problem)
            continue
        if (chunk.chunk_id, quote) in seen:
            continue
        seen.add((chunk.chunk_id, quote))
        kept.append(
            VerifiedCitation(
                chunk_id=chunk.chunk_id,
                arxiv_id=chunk.arxiv_id,
                title=chunk.title,
                year=chunk.year,
                page=chunk.page,
                quote=citation.quote.strip(),
            )
        )

    if not answer.sufficient_evidence:
        return FinalAnswer(
            answer=REFUSAL,
            citations=[],
            refused=True,
            reason="model found the evidence insufficient",
            dropped_citations=dropped,
        )
    if not kept:
        return FinalAnswer(
            answer=REFUSAL,
            citations=[],
            refused=True,
            reason="no citation passed verification",
            dropped_citations=dropped,
        )
    return FinalAnswer(
        answer=answer.answer, citations=kept, refused=False, dropped_citations=dropped
    )
