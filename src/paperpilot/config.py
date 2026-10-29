"""Central configuration: every tunable of PaperPilot lives here.

Priority (highest first): environment variables, then the .env file, then the defaults below.
"""

from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# This file is src/paperpilot/config.py, so parents[2] is the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths (override DATA_DIR in Docker) ---
    data_dir: Path = PROJECT_ROOT / "data"

    # --- Phase 1: download ---
    download_delay_seconds: float = 3.0  # polite pause between arXiv requests

    # --- Phase 3: chunking (characters) ---
    chunk_size: int = 1000  # target size
    chunk_max_size: int = 1200  # hard limit
    chunk_overlap: int = 150

    # --- Phase 4: models and indexes ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    qdrant_collection: str = "paperpilot_chunks"

    # --- Phase 5: retrieval (flags are used by the ablation) ---
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    final_top_k: int = 5
    use_bm25: bool = True
    use_reranker: bool = True

    # --- Phase 7: generation (model names are set in .env, never hardcoded) ---
    gemini_api_key: SecretStr | None = None
    gemini_main_model: str = ""
    gemini_light_model: str = ""
    llm_temperature: float = 0.0
    max_quote_words: int = 30
    quote_match_threshold: float = 90.0  # rapidfuzz score, 0-100

    # --- Phase 8: Corrective RAG ---
    use_crag: bool = True
    grade_high: float = 0.7  # placeholder: tuned on the eval set in Phase 8
    grade_low: float = 0.2  # placeholder: tuned on the eval set in Phase 8
    max_retries: int = 1

    # --- Derived paths (computed from data_dir, so one setting moves them all) ---
    @property
    def papers_yaml(self) -> Path:
        return self.data_dir / "papers.yaml"

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def qdrant_path(self) -> Path:
        return self.index_dir / "qdrant"

    @property
    def bm25_path(self) -> Path:
        return self.index_dir / "bm25.pkl"

    @property
    def llm_cache_dir(self) -> Path:
        return PROJECT_ROOT / ".cache" / "llm"

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if not 0 <= self.chunk_overlap < self.chunk_size <= self.chunk_max_size:
            raise ValueError("Need 0 <= chunk_overlap < chunk_size <= chunk_max_size")
        if not 0.0 <= self.grade_low < self.grade_high <= 1.0:
            raise ValueError("Need 0 <= grade_low < grade_high <= 1")
        if self.final_top_k > min(self.vector_top_k, self.bm25_top_k):
            raise ValueError("final_top_k cannot exceed the retrieval top-k values")
        return self


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse them everywhere."""
    return Settings()