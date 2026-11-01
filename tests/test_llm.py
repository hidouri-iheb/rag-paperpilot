from types import SimpleNamespace

from pydantic import BaseModel

from paperpilot.generation.llm import LLM


class Echo(BaseModel):
    value: str


class FakeModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        return SimpleNamespace(text='{"value": "ok"}')


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


def test_identical_call_hits_cache(tmp_path):
    client = FakeClient()
    llm = LLM(client, tmp_path, temperature=0.0)
    first = llm.generate("m", "hello", Echo)
    second = llm.generate("m", "hello", Echo)
    llm.close()
    assert first == second == Echo(value="ok")
    assert client.models.calls == 1


def test_different_prompt_is_a_new_call(tmp_path):
    client = FakeClient()
    llm = LLM(client, tmp_path, temperature=0.0)
    llm.generate("m", "hello", Echo)
    llm.generate("m", "something else", Echo)
    llm.close()
    assert client.models.calls == 2
