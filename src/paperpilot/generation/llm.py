"""Gemini client: structured output, disk cache, retries on rate limits."""

import hashlib
import json
import logging
from pathlib import Path
from typing import TypeVar

import diskcache
from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from paperpilot.config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """Gemini can't answer right now; callers show a friendly message instead of crashing."""


class QuotaExhausted(LLMUnavailable):
    """The free quota is used up for now."""


class ModelBusy(LLMUnavailable):
    """Google's servers are overloaded (503); usually fine again a minute later."""


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, errors.APIError) and exc.code in (429, 500, 503)


class LLM:
    def __init__(self, client, cache_dir: Path, temperature: float):
        self.client = client
        self.cache = diskcache.Cache(str(cache_dir))
        self.temperature = temperature
        self.api_calls = 0  # successful uncached calls, the ones that use quota

    @classmethod
    def from_settings(cls, s: Settings) -> "LLM":
        if s.gemini_api_key is None:
            raise RuntimeError("GEMINI_API_KEY is missing in .env")
        client = genai.Client(api_key=s.gemini_api_key.get_secret_value())
        return cls(client, s.llm_cache_dir, s.llm_temperature)

    def cache_key(self, model: str, prompt: str, schema: type[BaseModel]) -> str:
        payload = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "schema": schema.model_json_schema(),
                "temperature": self.temperature,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(self, model: str, prompt: str, schema: type[T]) -> T:
        key = self.cache_key(model, prompt, schema)
        cached = self.cache.get(key)
        if cached is not None:
            return schema.model_validate_json(cached)
        try:
            result = self._call(model, prompt, schema)
        except errors.APIError as exc:
            if exc.code == 429:
                raise QuotaExhausted("Gemini free quota exhausted, try again later") from exc
            if exc.code in (500, 503):
                raise ModelBusy(f"{model} is overloaded right now, try again in a minute") from exc
            raise
        self.cache.set(key, result.model_dump_json())
        self.api_calls += 1
        log.info("gemini call #%d (%s)", self.api_calls, model)
        return result

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _call(self, model: str, prompt: str, schema: type[T]) -> T:
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return schema.model_validate_json(response.text)

    def close(self) -> None:
        self.cache.close()
