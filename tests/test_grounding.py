"""Unit tests for strict citation grounding, faithfulness verification, and abstention."""

from __future__ import annotations

import pytest

from hybrid_rag.engine import HybridRAGEngine
from hybrid_rag.schemas import QueryRequest


@pytest.mark.asyncio
async def test_grounded_answer_contains_valid_inline_citations(
    populated_engine: HybridRAGEngine,
) -> None:
    resp = await populated_engine.query(
        QueryRequest(
            query="Berapa batas waktu notifikasi kebocoran data pribadi menurut UU PDP?",
            mode="hybrid",
            top_k=5,
        )
    )
    assert resp.abstained is False
    assert resp.language == "id"
    assert resp.faithfulness_score >= 0.90
    assert resp.supported_claims == resp.total_claims
    assert resp.total_claims >= 1
    assert len(resp.citations) >= 1
    assert any(c.doc_id == "uu_pdp_ringkasan" for c in resp.citations)
    assert "3x24" in resp.answer or "72 jam" in resp.answer
    for cite in resp.citations:
        assert f"[{cite.citation_key}]" in resp.answer


@pytest.mark.asyncio
async def test_automatic_abstention_on_out_of_domain_queries(
    populated_engine: HybridRAGEngine,
) -> None:
    resp_id = await populated_engine.query(
        QueryRequest(
            query="Bagaimana resep tradisional membuat rendang daging sapi khas Padang?",
            mode="hybrid",
            top_k=5,
        )
    )
    assert resp_id.abstained is True
    assert resp_id.language == "id"
    assert resp_id.citations == []
    assert "tidak dapat menjawab" in resp_id.answer.lower()
    assert resp_id.abstention_reason is not None

    resp_en = await populated_engine.query(
        QueryRequest(
            query="Who won the FIFA World Cup 2022 final in Qatar?",
            mode="hybrid",
            top_k=5,
        )
    )
    assert resp_en.abstained is True
    assert resp_en.language == "en"
    assert resp_en.citations == []
    assert "abstained" in resp_en.answer.lower()


def test_faithfulness_verifier_penalizes_hallucinated_and_uncited_claims(
    populated_engine: HybridRAGEngine,
) -> None:
    hits = populated_engine.retriever.retrieve(
        "What PSI thresholds define critical data drift?",
        mode="hybrid",
        top_k=3,
    )
    valid_key = hits[0].chunk.citation_key

    # 1. Completely uncited answer -> 0.0 faithfulness
    uncited = populated_engine.grounding.verify_answer(
        "Critical data drift occurs when PSI exceeds 0.25 and pages the on-call engineer.",
        hits,
    )
    assert uncited.faithfulness_score == 0.0
    assert uncited.supported_claims == 0

    # 2. Fabricated claim with a real citation tag -> low faithfulness
    hallucinated = populated_engine.grounding.verify_answer(
        f"Bananas grow on tropical volcanic islands with purple moonlight [{valid_key}].",
        hits,
    )
    assert hallucinated.faithfulness_score < 0.35
    assert hallucinated.supported_claims == 0
