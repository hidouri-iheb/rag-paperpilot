from paperpilot.config import Settings
from paperpilot.graph.crag import (
    ChunkGrades,
    GeneralAnswer,
    RewrittenQuery,
    build_crag,
    run_crag,
)
from paperpilot.models import Answer, Chunk, Citation, RetrievedChunk

TEXT = "The encoder is composed of a stack of N = 6 identical layers."
S = Settings(
    _env_file=None,
    grade_high=0.9,
    grade_low=0.1,
    gemini_main_model="main",
    gemini_light_model="light",
)


def hit(score: float) -> list[RetrievedChunk]:
    chunk = Chunk(
        chunk_id="a:p3:c0",
        arxiv_id="a",
        title="T",
        year=2017,
        section=None,
        page=3,
        text=TEXT,
        header="T (2017) › p.3",
    )
    return [RetrievedChunk(chunk=chunk, score=score)]


class FakeRetriever:
    def __init__(self, scores: dict[str, float]):
        self.scores = scores  # query -> top score, anything else scores 0.01

    def retrieve(self, query):
        return hit(self.scores.get(query, 0.01))


class FakeLLM:
    def __init__(self, grade_ids=(), insufficient_first=False, general_in_scope=False):
        self.calls: list[str] = []
        self.grade_ids = list(grade_ids)
        self.insufficient_first = insufficient_first
        self.general_in_scope = general_in_scope
        self.answers = 0

    def generate(self, model, prompt, schema):
        self.calls.append(schema.__name__)
        if schema is ChunkGrades:
            return ChunkGrades(relevant_chunk_ids=self.grade_ids)
        if schema is RewrittenQuery:
            return RewrittenQuery(query="better query")
        if schema is GeneralAnswer:
            text = "A general answer." if self.general_in_scope else ""
            return GeneralAnswer(in_scope=self.general_in_scope, answer=text)
        self.answers += 1
        sufficient = not (self.insufficient_first and self.answers == 1)
        return Answer(
            answer="Six layers.",
            citations=[Citation(chunk_id="a:p3:c0", quote="stack of N = 6 identical layers")],
            sufficient_evidence=sufficient,
        )


def ask(retriever, llm, allow_general=None):
    return run_crag(build_crag(retriever, llm, S), "q", allow_general)


def test_direct_path_uses_one_llm_call():
    llm = FakeLLM()
    final, path = ask(FakeRetriever({"q": 0.95}), llm)
    assert path == "direct" and not final.refused
    assert llm.calls == ["Answer"]


def test_corrected_path_after_rewrite():
    llm = FakeLLM()
    final, path = ask(FakeRetriever({"q": 0.05, "better query": 0.95}), llm)
    assert path == "corrected" and not final.refused
    assert llm.calls == ["RewrittenQuery", "Answer"]


def test_refuses_after_exactly_one_retry():
    llm = FakeLLM()
    final, path = ask(FakeRetriever({}), llm)
    assert path == "refused" and final.refused
    assert llm.calls.count("RewrittenQuery") == 1


def test_middle_zone_asks_the_grader():
    llm = FakeLLM(grade_ids=["a:p3:c0"])
    final, path = ask(FakeRetriever({"q": 0.5}), llm)
    assert path == "direct"
    assert llm.calls == ["ChunkGrades", "Answer"]


def test_refused_answer_triggers_one_rewrite():
    llm = FakeLLM(insufficient_first=True)
    final, path = ask(FakeRetriever({"q": 0.95, "better query": 0.95}), llm)
    assert path == "corrected" and not final.refused
    assert llm.calls == ["Answer", "RewrittenQuery", "Answer"]


def test_general_answer_when_library_has_nothing():
    llm = FakeLLM(general_in_scope=True)
    final, path = ask(FakeRetriever({}), llm)
    assert path == "general" and final.source == "general"
    assert final.refused  # the library still had no evidence
    assert final.citations == []  # general answers never carry citations
    assert final.answer == "A general answer."


def test_off_topic_question_is_still_refused():
    llm = FakeLLM(general_in_scope=False)
    final, path = ask(FakeRetriever({}), llm)
    assert path == "refused" and final.source == "library"


def test_strict_mode_never_calls_the_general_model():
    llm = FakeLLM(general_in_scope=True)
    final, path = ask(FakeRetriever({}), llm, allow_general=False)
    assert path == "refused"
    assert "GeneralAnswer" not in llm.calls
