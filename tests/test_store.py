"""Unit tests for SQLite + NumPy persistent IndexStore."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hybrid_rag.chunking import StructureAwareChunker
from hybrid_rag.store import IndexStore


def test_store_upsert_reload_and_delete(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "idx")
    chunker = StructureAwareChunker(chunk_size=400, chunk_overlap=50)

    doc1, chunks1 = chunker.chunk_text(
        doc_id="doc_alpha",
        title="Alpha Guide",
        content="## Intro\nAlpha uses AES-256 encryption and 90 day rotation.",
    )
    store.upsert_document(doc1, chunks1)
    assert store.counts() == (1, len(chunks1))

    # Upsert same doc_id with updated content replaces chunks idempotently
    doc1_v2, chunks1_v2 = chunker.chunk_text(
        doc_id="doc_alpha",
        title="Alpha Guide v2",
        content="## Intro\nUpdated policy.\n\n## Details\nSecond section content here.",
    )
    store.upsert_document(doc1_v2, chunks1_v2)
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0].title == "Alpha Guide v2"
    assert len(store.load_all_chunks()) == len(chunks1_v2)

    # Vector artifact persistence
    mat = np.eye(len(chunks1_v2), dtype=np.float32)
    keys = [c.citation_key for c in chunks1_v2]
    store.save_vector_artifacts(keys, mat, {"backend": "test"})
    loaded = store.load_vector_artifacts()
    assert loaded is not None
    loaded_keys, loaded_mat, loaded_meta = loaded
    assert loaded_keys == keys
    assert np.allclose(loaded_mat, mat)
    assert loaded_meta["backend"] == "test"

    # Cascade delete
    assert store.delete_document("doc_alpha") is True
    assert store.delete_document("nonexistent") is False
    assert store.counts() == (0, 0)
