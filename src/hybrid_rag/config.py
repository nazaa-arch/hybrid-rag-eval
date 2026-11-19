"""Configuration management for Hybrid RAG Eval via RAG_* environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def _load_dotenv(env_path: Path | None = None) -> None:
    """Lightweight .env loader without requiring python-dotenv."""
    path = env_path or Path(".env")
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    """Application settings loaded from RAG_* environment variables."""

    # LLM / Synthesis provider
    llm_provider: Literal["extractive-grounded", "ollama", "gemini", "openai"] = Field(
        default="extractive-grounded"
    )
    llm_model: str = Field(default="extractive-grounded-v1")
    llm_timeout_seconds: float = Field(default=30.0, gt=0)

    # Pluggable embedding backend
    embedding_provider: Literal["local-svd", "ollama", "gemini", "openai"] = Field(
        default="local-svd"
    )
    embedding_model: str = Field(default="tfidf-svd-128")
    embedding_dim: int = Field(default=128, ge=8, le=1024)

    # External provider endpoints & keys
    ollama_base_url: str = Field(default="http://localhost:11434")
    gemini_api_key: str = Field(default="")
    gemini_base_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta")
    openai_api_key: str = Field(default="")
    openai_base_url: str = Field(default="https://api.openai.com/v1")

    # Chunking parameters
    chunk_size: int = Field(default=680, ge=120, le=4000)
    chunk_overlap: int = Field(default=96, ge=0, le=1000)

    # Retrieval & Reranking parameters
    rrf_k: int = Field(default=60, ge=1, le=500)
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_pool_size: int = Field(default=25, ge=5, le=200)
    bm25_k1: float = Field(default=1.5, ge=0.1, le=5.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)

    # Grounding & Abstention controls
    min_confidence_threshold: float = Field(default=0.32, ge=0.0, le=1.0)
    min_term_coverage: float = Field(default=0.22, ge=0.0, le=1.0)
    min_dense_similarity: float = Field(default=0.16, ge=0.0, le=1.0)
    max_answer_sentences: int = Field(default=4, ge=1, le=10)

    # Persistence & Corpus paths
    index_dir: Path = Field(default=Path("data/index"))
    corpus_dir: Path = Field(default=Path("data/corpus"))
    eval_report_path: Path = Field(default=Path("eval/latest_report.json"))
    auto_load_corpus: bool = Field(default=True)

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> Settings:
        """Instantiate settings from environment variables (with optional .env file)."""
        _load_dotenv(env_path)
        return cls(
            llm_provider=_env_str("RAG_LLM_PROVIDER", "extractive-grounded"),  # type: ignore[arg-type]
            llm_model=_env_str("RAG_LLM_MODEL", "extractive-grounded-v1"),
            llm_timeout_seconds=_env_float("RAG_LLM_TIMEOUT_SECONDS", 30.0),
            embedding_provider=_env_str("RAG_EMBEDDING_PROVIDER", "local-svd"),  # type: ignore[arg-type]
            embedding_model=_env_str("RAG_EMBEDDING_MODEL", "tfidf-svd-128"),
            embedding_dim=_env_int("RAG_EMBEDDING_DIM", 128),
            ollama_base_url=_env_str("RAG_OLLAMA_BASE_URL", "http://localhost:11434"),
            gemini_api_key=_env_str("RAG_GEMINI_API_KEY", ""),
            gemini_base_url=_env_str(
                "RAG_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
            ),
            openai_api_key=_env_str("RAG_OPENAI_API_KEY", ""),
            openai_base_url=_env_str("RAG_OPENAI_BASE_URL", "https://api.openai.com/v1"),
            chunk_size=_env_int("RAG_CHUNK_SIZE", 680),
            chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 96),
            rrf_k=_env_int("RAG_RRF_K", 60),
            top_k=_env_int("RAG_TOP_K", 5),
            candidate_pool_size=_env_int("RAG_CANDIDATE_POOL_SIZE", 25),
            bm25_k1=_env_float("RAG_BM25_K1", 1.5),
            bm25_b=_env_float("RAG_BM25_B", 0.75),
            min_confidence_threshold=_env_float("RAG_MIN_CONFIDENCE_THRESHOLD", 0.32),
            min_term_coverage=_env_float("RAG_MIN_TERM_COVERAGE", 0.22),
            min_dense_similarity=_env_float("RAG_MIN_DENSE_SIMILARITY", 0.16),
            max_answer_sentences=_env_int("RAG_MAX_ANSWER_SENTENCES", 4),
            index_dir=Path(_env_str("RAG_INDEX_DIR", "data/index")),
            corpus_dir=Path(_env_str("RAG_CORPUS_DIR", "data/corpus")),
            eval_report_path=Path(_env_str("RAG_EVAL_REPORT_PATH", "eval/latest_report.json")),
            auto_load_corpus=_env_bool("RAG_AUTO_LOAD_CORPUS", True),
            host=_env_str("RAG_HOST", "0.0.0.0"),
            port=_env_int("RAG_PORT", 8000),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached process-wide settings."""
    return Settings.from_env()
