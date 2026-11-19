"""Shared pytest fixtures for Hybrid RAG Eval tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from hybrid_rag.api import create_app
from hybrid_rag.config import Settings
from hybrid_rag.engine import HybridRAGEngine

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "corpus"


@pytest.fixture()
def isolated_settings(tmp_path: Path) -> Settings:
    """Create isolated settings pointing to a temporary SQLite/NumPy index dir."""
    return Settings(
        index_dir=tmp_path / "index",
        corpus_dir=CORPUS_DIR,
        eval_report_path=REPO_ROOT / "eval" / "latest_report.json",
        llm_provider="extractive-grounded",
        embedding_provider="local-svd",
        auto_load_corpus=True,
    )


@pytest.fixture()
def populated_engine(isolated_settings: Settings) -> HybridRAGEngine:
    """Return a HybridRAGEngine preloaded with the 6-document bilingual corpus."""
    engine = HybridRAGEngine(settings=isolated_settings)
    engine.load_corpus_directory(CORPUS_DIR)
    return engine


@pytest.fixture()
def api_client(isolated_settings: Settings) -> Iterator[TestClient]:
    """Return a FastAPI TestClient running with isolated temporary persistence."""
    app = create_app(settings=isolated_settings)
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def sample_pdf_bytes() -> bytes:
    """Generate a valid multi-page PDF document in memory using PyMuPDF."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "Payment Gateway Resilience Policy\n\n"
        "All payment settlement webhooks must verify HMAC-SHA256 signatures.\n"
        "Failed webhook deliveries are retried with exponential backoff up to 5 attempts.",
        fontsize=11,
    )
    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "Disaster Recovery Targets\n\n"
        "Recovery Time Objective (RTO) for payment clearing is 15 minutes.\n"
        "Recovery Point Objective (RPO) is zero data loss via synchronous replication.",
        fontsize=11,
    )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes
