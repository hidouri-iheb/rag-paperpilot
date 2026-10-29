import pytest
from pydantic import ValidationError

from paperpilot.config import Settings


def make(**overrides) -> Settings:
    # _env_file=None: tests never read your real .env, so they behave the same in CI.
    return Settings(_env_file=None, **overrides)


def test_defaults_load_without_env_file():
    s = make()
    assert s.final_top_k == 5
    assert s.pdf_dir == s.data_dir / "pdfs"


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        make(chunk_overlap=1000)


def test_grade_thresholds_must_be_ordered():
    with pytest.raises(ValidationError):
        make(grade_low=0.8, grade_high=0.5)


def test_api_key_is_hidden_when_printed():
    s = make(gemini_api_key="secret-123")
    assert "secret-123" not in repr(s)