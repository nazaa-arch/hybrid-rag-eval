"""Unit tests for BM25, DenseVectorRetriever, RRF, CrossSignalReranker, and HybridRetriever."""

from __future__ import annotations

import httpx
import numpy as np
import pytest

from hybrid_rag.engine import HybridRAGEngine
from hybrid_rag.providers import GeminiProvider
from hybrid_rag.retrieval import (
    BM25Retriever,
    DenseVectorRetriever,
    EmbeddingBackend,
    GeminiEmbeddingBackend,
    HybridRetriever,
    LocalSVDEmbeddingBackend,
    ReciprocalRankFusion,
)
from hybrid_rag.schemas import Chunk, RetrievalHit
from hybrid_rag.text_utils import detect_language, tokenize


def test_bilingual_tokenization_and_language_detection() -> None:
    id_tokens = tokenize("Berapa batas waktu notifikasi kebocoran data yg wajib menurut UU PDP?")
    assert "notifikasi" in id_tokens
    assert "kebocoran" in id_tokens
    assert "yg" not in id_tokens  # normalized to stopword 'yang' and stripped
    assert detect_language("Berapa denda maksimal pelanggaran UU PDP?") == "id"
    assert detect_language("What are the PSI thresholds for critical data drift?") == "en"


def test_bm25_exact_lexical_match(populated_engine: HybridRAGEngine) -> None:
    hits = populated_engine.retriever.retrieve(
        "SOP-AI-GOV-04 Tier 1 High Risk",
        mode="bm25",
        top_k=3,
    )
    assert len(hits) == 3
    assert hits[0].chunk.doc_id == "ai_governance_sop"
    assert hits[0].bm25_rank == 1
    assert hits[0].bm25_score > 0.0


def test_dense_vector_semantic_and_morphological_match(populated_engine: HybridRAGEngine) -> None:
    hits = populated_engine.retriever.retrieve(
        "perlindungan data pribadi notifikasi kebocoran 72 jam",
        mode="dense",
        top_k=3,
    )
    assert len(hits) == 3
    assert hits[0].chunk.doc_id == "uu_pdp_ringkasan"
    assert hits[0].dense_rank == 1
    assert 0.0 < hits[0].dense_score <= 1.0


def test_rrf_fuses_ranks_accurately() -> None:
    rrf = ReciprocalRankFusion(k=60)
    bm25_list = [(0, 10.0, ["psi"]), (1, 8.0, ["drift"]), (2, 5.0, ["sla"])]
    dense_list = [(1, 0.88), (2, 0.72), (3, 0.65)]
    fused = rrf.fuse(bm25_list, dense_list)

    # Doc 1 is rank 2 in BM25 and rank 1 in Dense -> highest combined RRF
    assert fused[0]["idx"] == 1
    assert fused[0]["bm25_rank"] == 2
    assert fused[0]["dense_rank"] == 1
    expected_rrf = (1.0 / 62.0) + (1.0 / 61.0)
    assert abs(float(fused[0]["rrf_score"]) - expected_rrf) < 1e-6


def test_hybrid_mode_returns_full_explainability_signals(
    populated_engine: HybridRAGEngine,
) -> None:
    hits = populated_engine.retriever.retrieve(
        "What PSI thresholds define amber and critical data drift?",
        mode="hybrid",
        top_k=5,
    )
    assert len(hits) == 5
    top = hits[0]
    assert top.chunk.doc_id == "mlops_incident_runbook"
    assert top.rrf_score > 0.0
    assert 0.0 < top.rerank_score <= 1.0
    assert top.signals.term_coverage > 0.4
    assert top.signals.dense_sim > 0.2
    # Results must be sorted descending by rerank_score
    scores = [h.rerank_score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_tiny_corpus_edge_case_does_not_crash() -> None:
    chunks = [
        Chunk(
            chunk_id="c000",
            doc_id="tiny",
            title="Tiny Doc",
            section="Main",
            text="KMS encrypts storage volumes using AES-256-GCM.",
        )
    ]
    retriever = HybridRetriever()
    retriever.index(chunks)
    hits = retriever.retrieve("AES-256-GCM encryption", mode="hybrid", top_k=3)
    assert len(hits) == 1
    assert hits[0].chunk.doc_id == "tiny"

    # Also test empty index safely
    empty_bm25 = BM25Retriever()
    empty_bm25.fit([])
    assert empty_bm25.search("anything") == []
    empty_dense = DenseVectorRetriever()
    empty_dense.fit([])
    assert empty_dense.search("anything") == []

    # Punctuation-only corpus must not crash TfidfVectorizer with empty vocabulary
    svd = LocalSVDEmbeddingBackend()
    vecs = svd.fit_transform(["!!!", "???"])
    assert vecs.shape[0] == 2
    q_vec = svd.embed_query("hello world")
    assert q_vec.shape[0] == vecs.shape[1]


def test_hybrid_retriever_embeds_query_only_once() -> None:
    class _CountingBackend(EmbeddingBackend):
        name = "counting"

        def __init__(self) -> None:
            self.embed_calls = 0

        def fit_transform(self, texts: list[str]) -> np.ndarray:
            return np.ones((len(texts), 4), dtype=np.float32) * 0.5

        def embed_query(self, query: str) -> np.ndarray:
            self.embed_calls += 1
            return np.ones((4,), dtype=np.float32) * 0.5

    backend = _CountingBackend()
    retriever = HybridRetriever(embedding_backend=backend)
    retriever.index(
        [
            Chunk(chunk_id="c000", doc_id="d1", title="T", section="S", text="hello alpha"),
            Chunk(chunk_id="c001", doc_id="d2", title="T", section="S", text="hello beta"),
        ]
    )
    retriever.retrieve("hello alpha", mode="hybrid", top_k=2)
    assert backend.embed_calls == 1


@pytest.mark.asyncio
async def test_gemini_auth_uses_header_not_url_query_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requests: list[tuple[str, dict[str, str]]] = []
    secret_key = "test-gemini-secret-key"

    def _mock_post(self: httpx.Client, url: str, **kwargs: object) -> httpx.Response:
        headers = {k.lower(): str(v) for k, v in (kwargs.get("headers") or {}).items()}  # type: ignore[union-attr]
        captured_requests.append((str(url), headers))
        return httpx.Response(
            200,
            json={"embedding": {"values": [0.6, 0.8]}},
            request=httpx.Request("POST", url),
        )

    async def _mock_async_post(
        self: httpx.AsyncClient, url: str, **kwargs: object
    ) -> httpx.Response:
        headers = {k.lower(): str(v) for k, v in (kwargs.get("headers") or {}).items()}  # type: ignore[union-attr]
        captured_requests.append((str(url), headers))
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "Grounded answer [d1#c000]."}]}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", _mock_post)
    monkeypatch.setattr(httpx.AsyncClient, "post", _mock_async_post)

    emb = GeminiEmbeddingBackend(
        api_key=secret_key,
        base_url="https://generativelanguage.googleapis.com/v1beta",
        model="text-embedding-004",
    )
    vec = emb.fit_transform(["hello"])
    assert vec.shape == (1, 2)

    provider = GeminiProvider(
        api_key=secret_key,
        base_url="https://generativelanguage.googleapis.com/v1beta",
        model="gemini-2.0-flash",
    )
    hit = RetrievalHit(
        chunk=Chunk(chunk_id="c000", doc_id="d1", title="T", section="S", text="hello"),
        rank=1,
        score=0.9,
    )
    ans = await provider.generate(query="hello", hits=[hit], language="en")
    assert "Grounded answer" in ans

    assert len(captured_requests) == 2
    for url, headers in captured_requests:
        assert secret_key not in url
        assert "key=" not in url
        assert headers.get("x-goog-api-key") == secret_key
