from paperpilot.config import Settings
from paperpilot.generation.llm import LLM
from paperpilot.generation.prompts import ANSWER_PROMPT, format_context
from paperpilot.generation.verify import verify
from paperpilot.models import Answer, FinalAnswer, RetrievedChunk


def draft_answer(llm: LLM, s: Settings, question: str, chunks: list[RetrievedChunk]) -> Answer:
    prompt = ANSWER_PROMPT.format(
        max_words=s.max_quote_words, question=question, context=format_context(chunks)
    )
    return llm.generate(s.gemini_main_model, prompt, Answer)


def generate_answer(
    llm: LLM, s: Settings, question: str, chunks: list[RetrievedChunk]
) -> FinalAnswer:
    raw = draft_answer(llm, s, question, chunks)
    return verify(raw, chunks, s.quote_match_threshold, s.max_quote_words)
