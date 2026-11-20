"""SQLite + JSON/NumPy persistent index store for documents, chunks, and dense vectors."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_rag.schemas import Chunk, Document


class IndexStore:
    """Persists documents and chunks in SQLite alongside NumPy/JSON vector artifacts."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.index_dir / "index.sqlite"
        self.vectors_path = self.index_dir / "embeddings.npz"
        self.vector_meta_path = self.index_dir / "vector_meta.json"
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    citation_key TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    section TEXT NOT NULL,
                    page INTEGER,
                    chunk_index INTEGER NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
                """
            )

    def upsert_document(self, document: Document, chunks: list[Chunk]) -> None:
        """Atomically replace a document and all of its chunks."""
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (document.doc_id,))
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (
                    doc_id, title, source_type, content_hash,
                    char_count, chunk_count, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.doc_id,
                    document.title,
                    document.source_type,
                    document.content_hash,
                    document.char_count,
                    len(chunks),
                    json.dumps(document.metadata, ensure_ascii=False),
                    document.created_at.isoformat(),
                ),
            )
            conn.executemany(
                """
                INSERT INTO chunks (
                    citation_key, chunk_id, doc_id, title, section, page,
                    chunk_index, char_start, char_end, text, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.citation_key,
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.title,
                        chunk.section,
                        chunk.page,
                        chunk.chunk_index,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.text,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                    )
                    for chunk in chunks
                ],
            )

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks; return True if a document was deleted."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return cursor.rowcount > 0

    def list_documents(self) -> list[Document]:
        """Return all indexed documents ordered by doc_id."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY doc_id ASC").fetchall()
        return [
            Document(
                doc_id=row["doc_id"],
                title=row["title"],
                source_type=row["source_type"],
                content_hash=row["content_hash"],
                char_count=row["char_count"],
                chunk_count=row["chunk_count"],
                metadata=json.loads(row["metadata_json"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def load_all_chunks(self) -> list[Chunk]:
        """Load every chunk ordered deterministically by doc_id and chunk_index."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks ORDER BY doc_id ASC, chunk_index ASC"
            ).fetchall()
        return [
            Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                title=row["title"],
                section=row["section"],
                page=row["page"],
                chunk_index=row["chunk_index"],
                char_start=row["char_start"],
                char_end=row["char_end"],
                text=row["text"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def counts(self) -> tuple[int, int]:
        """Return (document_count, chunk_count)."""
        with self._connect() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return int(doc_count), int(chunk_count)

    def save_vector_artifacts(
        self,
        citation_keys: list[str],
        embeddings: np.ndarray,
        meta: dict[str, Any],
    ) -> None:
        """Persist L2-normalized chunk embeddings as compressed NumPy + JSON metadata."""
        np.savez_compressed(self.vectors_path, embeddings=embeddings.astype(np.float32))
        payload = {"citation_keys": citation_keys, "meta": meta}
        self.vector_meta_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear_vector_artifacts(self) -> None:
        """Remove persisted vector artifacts when the index becomes empty."""
        for path in (self.vectors_path, self.vector_meta_path):
            if path.is_file():
                path.unlink()

    def load_vector_artifacts(self) -> tuple[list[str], np.ndarray, dict[str, Any]] | None:
        """Load persisted dense embeddings if present."""
        if not self.vectors_path.is_file() or not self.vector_meta_path.is_file():
            return None
        meta_payload = json.loads(self.vector_meta_path.read_text(encoding="utf-8"))
        with np.load(self.vectors_path) as data:
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        return meta_payload["citation_keys"], embeddings, meta_payload.get("meta", {})
