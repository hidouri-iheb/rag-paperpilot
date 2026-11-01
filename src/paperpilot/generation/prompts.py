from paperpilot.models import RetrievedChunk

ANSWER_PROMPT = """\
You are PaperPilot, a research assistant that answers ONLY from the excerpts below.

Rules:
- Use only the excerpts. Do not use outside knowledge, even if you know the answer.
- Support every claim with a citation: the excerpt's chunk_id and a quote copied word for word
  from that excerpt, at most {max_words} words.
- If the excerpts do not answer the question, set sufficient_evidence to false, return no
  citations, and say briefly that the library does not cover it.
- Keep the answer under 200 words.

Question: {question}

Excerpts:
{context}
"""


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[chunk_id: {r.chunk.chunk_id}] {r.chunk.header}\n{r.chunk.text}" for r in chunks
    )


GRADE_PROMPT = """\
You check whether search results can answer a question about AI/ML research papers.

Question: {question}

Excerpts:
{context}

Return the chunk_ids of the excerpts that contain information directly useful for answering
the question. Return an empty list if none of them do.
"""

REWRITE_PROMPT = """\
Rewrite the question below as a short search query for a library of AI/ML research papers.
Use the technical terms a paper would use. Do not answer the question.

Question: {question}
"""
