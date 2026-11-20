"""Pydantic v2 domain models and API schemas for Hybrid RAG Eval."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

RetrievalMode = Literal["hybrid", "bm25", "dense"]

SAFE_DOC_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")


def validate_doc_id(value: str) -> str:
    """Validate doc_id against traversal, XSS, and citation-tag breaking characters."""
    cleaned = value.strip()
    if not SAFE_DOC_ID_RE.match(cleaned) or ".." in cleaned:
        raise ValueError(
            "Invalid doc_id: must start with an alphanumeric character, contain only "
            "letters, digits, '_', '-', or '.', be 1-128 chars, and not contain '..'."
        )
    return cleaned


class Document(BaseModel):
    """Represents an ingested source document."""

    doc_id: str = Field(..., min_length=1, max_length=128)
    title: str = Field(..., min_length=1, max_length=300)
    source_type: Literal["markdown", "text", "pdf"] = "markdown"
    content_hash: str = ""
    char_count: int = 0
    chunk_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Chunk(BaseModel):
    """Structure-aware text chunk with positional & hierarchical metadata."""

    chunk_id: str
    doc_id: str
    title: str
    section: str = "General"
    page: int | None = None
    chunk_index: int = 0
    char_start: int = 0
    char_end: int = 0
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def citation_key(self) -> str:
        """Canonical inline citation key: doc_id#chunk_id."""
        return f"{self.doc_id}#{self.chunk_id}"


class RetrievalSignals(BaseModel):
    """Decomposed cross-signal reranking features for full explainability."""

    rrf_norm: float = 0.0
    term_coverage: float = 0.0
    phrase_boost: float = 0.0
    title_section_match: float = 0.0
    proximity: float = 0.0
    dense_sim: float = 0.0


class RetrievalHit(BaseModel):
    """Single retrieved chunk with multi-stage ranking breakdown."""

    chunk: Chunk
    rank: int
    score: float
    bm25_rank: int | None = None
    bm25_score: float = 0.0
    dense_rank: int | None = None
    dense_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    matched_terms: list[str] = Field(default_factory=list)
    signals: RetrievalSignals = Field(default_factory=RetrievalSignals)


class Citation(BaseModel):
    """Verified inline citation linking generated claims back to a source chunk."""

    citation_key: str
    doc_id: str
    chunk_id: str
    title: str
    section: str
    page: int | None = None
    snippet: str
    support_score: float = Field(ge=0.0, le=1.0)


class IngestTextRequest(BaseModel):
    """JSON payload for ingesting raw text or markdown into the index."""

    doc_id: str = Field(..., min_length=1, max_length=128)
    title: str = Field(..., min_length=1, max_length=300)
    content: str = Field(..., min_length=1, max_length=2_000_000)
    source_type: Literal["markdown", "text"] = "markdown"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("doc_id")
    @classmethod
    def _check_doc_id(cls, value: str) -> str:
        return validate_doc_id(value)


class IngestResponse(BaseModel):
    """Response returned after indexing a document."""

    document: Document
    chunks_indexed: int
    total_documents: int
    total_chunks: int


class QueryRequest(BaseModel):
    """RAG query request supporting selectable retrieval modes."""

    query: str = Field(..., min_length=2, max_length=2000)
    mode: RetrievalMode = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class LatencyBreakdown(BaseModel):
    """Per-stage latency measurements in milliseconds."""

    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    verification_ms: float = 0.0
    total_ms: float = 0.0


class QueryResponse(BaseModel):
    """Full grounded RAG answer with citations, faithfulness, and explainability."""

    query: str
    mode: RetrievalMode
    language: Literal["id", "en"]
    answer: str
    abstained: bool = False
    abstention_reason: str | None = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    supported_claims: int = 0
    total_claims: int = 0
    provider: str = "extractive-grounded"
    citations: list[Citation] = Field(default_factory=list)
    hits: list[RetrievalHit] = Field(default_factory=list)
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)


class HealthResponse(BaseModel):
    """Service health and index readiness snapshot."""

    status: Literal["ok"] = "ok"
    version: str
    documents_indexed: int
    chunks_indexed: int
    llm_provider: str
    embedding_provider: str
    index_dir: str


class ModeEvalMetrics(BaseModel):
    """Aggregated evaluation metrics for a single retrieval mode."""

    mode: RetrievalMode
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    citation_faithfulness: float
    keyword_answer_recall: float
    abstention_accuracy: float
    abstention_precision: float
    abstention_recall: float
    latency_p50_ms: float
    latency_p95_ms: float


class EvalSummary(BaseModel):
    """Complete benchmark report comparing BM25, Dense, and Hybrid modes."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dataset_name: str = "hybrid-rag-eval-bilingual-v1"
    total_queries: int
    answerable_queries: int
    unanswerable_queries: int
    corpus_documents: int
    corpus_chunks: int
    modes: dict[str, ModeEvalMetrics]
    per_query_results: list[dict[str, Any]] = Field(default_factory=list)
