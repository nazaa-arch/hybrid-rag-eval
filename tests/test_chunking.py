"""Unit tests for StructureAwareChunker across Markdown, Plain Text, and PDF inputs."""

from __future__ import annotations

import pytest

from hybrid_rag.chunking import StructureAwareChunker


def test_markdown_heading_breadcrumbs_and_metadata() -> None:
    md = """# Security Handbook

## Access Control

### Just-in-Time Elevation
Engineers request temporary production access for at most 4 hours with peer review.

## Encryption Standards
All production storage volumes must use AES-256-GCM backed by KMS.
"""
    chunker = StructureAwareChunker(chunk_size=300, chunk_overlap=40)
    doc, chunks = chunker.chunk_text(
        doc_id="sec_handbook",
        title="Security Handbook",
        content=md,
        source_type="markdown",
    )

    assert doc.doc_id == "sec_handbook"
    assert doc.chunk_count == len(chunks)
    assert len(chunks) == 2
    assert chunks[0].section == "Access Control > Just-in-Time Elevation"
    assert chunks[1].section == "Encryption Standards"
    assert chunks[0].citation_key == "sec_handbook#c000"
    assert chunks[0].char_end > chunks[0].char_start


def test_recursive_splitting_with_overlap() -> None:
    paragraphs = [
        f"Paragraph {i}: Production RAG systems require careful monitoring of latency, recall, and faithfulness metrics."
        for i in range(1, 10)
    ]
    text = "\n\n".join(paragraphs)
    chunker = StructureAwareChunker(chunk_size=260, chunk_overlap=60)
    doc, chunks = chunker.chunk_text(
        doc_id="long_doc",
        title="Long Doc",
        content=text,
        source_type="text",
    )

    assert doc.source_type == "text"
    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk.text) <= 320
        assert chunk.text.strip() != ""
    # Adjacent chunks should share overlapping trailing words
    tail_words = set(chunks[0].text.split()[-6:])
    head_words = set(chunks[1].text.split()[:16])
    assert len(tail_words & head_words) >= 2


def test_pdf_chunking_preserves_pages(sample_pdf_bytes: bytes) -> None:
    chunker = StructureAwareChunker(chunk_size=500, chunk_overlap=60)
    doc, chunks = chunker.chunk_pdf_bytes(
        doc_id="payment_policy",
        title="Payment Gateway Resilience Policy",
        pdf_bytes=sample_pdf_bytes,
    )

    assert doc.source_type == "pdf"
    assert doc.metadata["page_count"] == 2
    assert len(chunks) >= 2
    pages = {c.page for c in chunks}
    assert pages == {1, 2}
    combined = " ".join(c.text for c in chunks)
    assert "HMAC-SHA256" in combined
    assert "15 minutes" in combined


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        StructureAwareChunker(chunk_size=200, chunk_overlap=200)
