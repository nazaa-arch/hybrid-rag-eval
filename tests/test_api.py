"""Integration tests for FastAPI REST endpoints, file ingestion, SSE streaming, and UI."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from hybrid_rag.api import create_app
from hybrid_rag.config import Settings
from hybrid_rag.providers import BaseAnswerProvider
from hybrid_rag.schemas import RetrievalHit


def test_ui_and_health_endpoints(api_client: TestClient) -> None:
    ui_resp = api_client.get("/")
    assert ui_resp.status_code == 200
    assert "Hybrid RAG Eval Workbench" in ui_resp.text

    health_resp = api_client.get("/health")
    assert health_resp.status_code == 200
    data = health_resp.json()
    assert data["status"] == "ok"
    assert data["documents_indexed"] == 6
    assert data["chunks_indexed"] >= 30


def test_list_documents_and_json_ingest_delete(api_client: TestClient) -> None:
    docs_resp = api_client.get("/v1/documents")
    assert docs_resp.status_code == 200
    assert len(docs_resp.json()) == 6

    ingest_resp = api_client.post(
        "/v1/ingest",
        json={
            "doc_id": "custom_sla",
            "title": "Custom Internal SLA",
            "content": "# Custom SLA\n\n## Uptime\nThe internal ledger API guarantees 99.99% uptime and p99 latency under 85 ms.",
            "source_type": "markdown",
        },
    )
    assert ingest_resp.status_code == 200
    body = ingest_resp.json()
    assert body["document"]["doc_id"] == "custom_sla"
    assert body["total_documents"] == 7

    # Query newly ingested document immediately
    q_resp = api_client.post(
        "/v1/query",
        json={"query": "What uptime and p99 latency does the internal ledger API guarantee?", "mode": "hybrid"},
    )
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    assert q_data["abstained"] is False
    assert "99.99%" in q_data["answer"]
    assert any(c["doc_id"] == "custom_sla" for c in q_data["citations"])

    # Delete custom document
    del_resp = api_client.delete("/v1/documents/custom_sla")
    assert del_resp.status_code == 200
    assert del_resp.json()["total_documents"] == 6


def test_multipart_markdown_and_pdf_upload(
    api_client: TestClient,
    sample_pdf_bytes: bytes,
) -> None:
    md_resp = api_client.post(
        "/v1/ingest",
        files={"file": ("oncall_notes.md", b"# Oncall Notes\n\n## Escalation\nEscalate P1 billing alerts to #payments-oncall.", "text/markdown")},
    )
    assert md_resp.status_code == 200
    assert md_resp.json()["document"]["doc_id"] == "oncall_notes"

    pdf_resp = api_client.post(
        "/v1/ingest",
        files={"file": ("payment_resilience.pdf", sample_pdf_bytes, "application/pdf")},
        data={"doc_id": "payment_resilience_pdf"},
    )
    assert pdf_resp.status_code == 200
    pdf_data = pdf_resp.json()
    assert pdf_data["document"]["source_type"] == "pdf"
    assert pdf_data["chunks_indexed"] >= 2

    q_resp = api_client.post(
        "/v1/query",
        json={"query": "What is the RTO for payment clearing and webhook signature algorithm?", "mode": "hybrid"},
    )
    assert q_resp.status_code == 200
    assert q_resp.json()["abstained"] is False
    assert any(c["doc_id"] == "payment_resilience_pdf" for c in q_resp.json()["citations"])


def test_query_modes_and_sse_streaming(api_client: TestClient) -> None:
    for mode in ("bm25", "dense", "hybrid"):
        resp = api_client.post(
            "/v1/query",
            json={
                "query": "What PSI thresholds define amber and critical data drift?",
                "mode": mode,
                "top_k": 4,
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["mode"] == mode
        assert payload["abstained"] is False
        assert len(payload["hits"]) == 4
        assert payload["faithfulness_score"] >= 0.90

    # SSE streaming endpoint
    with api_client.stream(
        "POST",
        "/v1/query/stream",
        json={
            "query": "Apa jadwal rotasi rahasia untuk API key dan kunci KMS?",
            "mode": "hybrid",
            "top_k": 3,
        },
    ) as stream_resp:
        assert stream_resp.status_code == 200
        assert "text/event-stream" in stream_resp.headers["content-type"]
        raw = "".join(stream_resp.iter_text())

    events: list[tuple[str, dict[str, object]]] = []
    for block in raw.strip().split("\n\n"):
        ev_name = ""
        ev_data = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                ev_name = line[6:].strip()
            elif line.startswith("data:"):
                ev_data += line[5:].strip()
        if ev_name and ev_data:
            events.append((ev_name, json.loads(ev_data)))

    event_names = [e[0] for e in events]
    assert event_names[0] == "metadata"
    assert "token" in event_names
    assert event_names[-1] == "done"

    done_payload = events[-1][1]
    assert done_payload["abstained"] is False
    assert float(done_payload["faithfulness_score"]) >= 0.90  # type: ignore[arg-type]


def test_latest_eval_endpoint(api_client: TestClient) -> None:
    resp = api_client.get("/v1/eval/latest")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["modes"].keys()) == {"bm25", "dense", "hybrid"}
    assert data["total_queries"] >= 25


def test_security_upload_validation_and_sanitization(
    api_client: TestClient,
    tmp_path: Path,
) -> None:
    # 1. Unsupported extension -> 400
    bad_ext = api_client.post(
        "/v1/ingest",
        files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
    )
    assert bad_ext.status_code == 400

    # 2. Corrupted PDF -> 400 (never 500)
    bad_pdf = api_client.post(
        "/v1/ingest",
        files={"file": ("broken.pdf", b"not-a-real-pdf-header", "application/pdf")},
    )
    assert bad_pdf.status_code == 400

    # 3. Path traversal filename is stripped to basename and safe doc_id
    traversal = api_client.post(
        "/v1/ingest",
        files={"file": ("../../etc/safe_notes.md", b"# Safe\nPolicy content.", "text/markdown")},
    )
    assert traversal.status_code == 200
    doc = traversal.json()["document"]
    assert doc["doc_id"] == "safe_notes"
    assert doc["metadata"]["filename"] == "safe_notes.md"

    # 4. Unsafe doc_id (XSS / traversal / citation break) rejected with 422 / 400
    for bad_id in ("../etc/passwd", "x');alert(1);//", "bad#id", "has spaces"):
        resp = api_client.post(
            "/v1/ingest",
            json={
                "doc_id": bad_id,
                "title": "Test",
                "content": "Valid body content.",
                "source_type": "markdown",
            },
        )
        assert resp.status_code == 422

    # 5. Whitespace-only content rejected with 400
    ws_resp = api_client.post(
        "/v1/ingest",
        json={
            "doc_id": "blank_doc",
            "title": "Blank",
            "content": "   \n\t  ",
            "source_type": "text",
        },
    )
    assert ws_resp.status_code == 400

    # 6. Malformed JSON rejected with 400
    malformed = api_client.post(
        "/v1/ingest",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert malformed.status_code == 400
    assert malformed.json()["detail"] == "Invalid JSON payload."

    # 7. Invalid doc_id on DELETE rejected with 400
    bad_del = api_client.delete("/v1/documents/bad;id")
    assert bad_del.status_code == 400

    # 8. Oversized file rejected with 413
    small_limit_settings = Settings(
        index_dir=tmp_path / "small_idx",
        auto_load_corpus=False,
        max_upload_bytes=1024,
    )
    with TestClient(create_app(settings=small_limit_settings)) as limited_client:
        oversized = limited_client.post(
            "/v1/ingest",
            files={"file": ("big.txt", b"a" * 2048, "text/plain")},
        )
        assert oversized.status_code == 413


def test_ui_html_hardened_against_inline_xss(api_client: TestClient) -> None:
    html = api_client.get("/").text
    assert "onclick=" not in html
    assert "&#39;" in html
    assert "data-delete-doc" in html


class _FailingProvider(BaseAnswerProvider):
    name = "failing-test-provider"

    async def generate(
        self,
        *,
        query: str,
        hits: Sequence[RetrievalHit],
        language: Literal["id", "en"],
    ) -> str:
        raise RuntimeError("secret upstream error")

    async def stream(
        self,
        *,
        query: str,
        hits: Sequence[RetrievalHit],
        language: Literal["id", "en"],
    ):
        raise RuntimeError("secret stream failure")
        yield ""  # pragma: no cover


def test_provider_failure_sanitized_in_rest_and_sse(
    isolated_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(settings=isolated_settings)
    with TestClient(app) as client:
        monkeypatch.setattr(app.state.engine, "provider", _FailingProvider())

        resp = client.post(
            "/v1/query",
            json={"query": "What PSI thresholds define amber and critical data drift?", "mode": "hybrid"},
        )
        assert resp.status_code == 502
        assert "secret upstream error" not in resp.text

        with client.stream(
            "POST",
            "/v1/query/stream",
            json={"query": "What PSI thresholds define amber and critical data drift?", "mode": "hybrid"},
        ) as stream_resp:
            assert stream_resp.status_code == 200
            raw = "".join(stream_resp.iter_text())
        assert "event: error" in raw
        assert "secret stream failure" not in raw
