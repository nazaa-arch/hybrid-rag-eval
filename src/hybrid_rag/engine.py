"""High-level HybridRAGEngine orchestrating ingestion, retrieval, grounding, and streaming."""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from hybrid_rag.chunking import StructureAwareChunker
from hybrid_rag.config import Settings
from hybrid_rag.grounding import GroundingEngine
from hybrid_rag.providers import BaseAnswerProvider, create_answer_provider
from hybrid_rag.retrieval import HybridRetriever, create_embedding_backend
from hybrid_rag.schemas import (
    Document,
    IngestResponse,
    LatencyBreakdown,
    QueryRequest,
    QueryResponse,
    validate_doc_id,
)
from hybrid_rag.store import IndexStore
from hybrid_rag.text_utils import detect_language


class HybridRAGEngine:
    """Stateful orchestrator for document indexing, hybrid retrieval, and grounded QA."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        provider: BaseAnswerProvider | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.chunker = StructureAwareChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.store = IndexStore(self.settings.index_dir)
        self.embedding_backend = create_embedding_backend(self.settings)
        self.retriever = HybridRetriever(
            bm25_k1=self.settings.bm25_k1,
            bm25_b=self.settings.bm25_b,
            rrf_k=self.settings.rrf_k,
            candidate_pool_size=self.settings.candidate_pool_size,
            embedding_backend=self.embedding_backend,
        )
        self.grounding = GroundingEngine(
            min_confidence_threshold=self.settings.min_confidence_threshold,
            min_term_coverage=self.settings.min_term_coverage,
            min_dense_similarity=self.settings.min_dense_similarity,
            max_answer_sentences=self.settings.max_answer_sentences,
        )
        self.provider = provider or create_answer_provider(self.settings, self.grounding)
        self._lock = threading.RLock()
        self.rebuild_index_from_store()

    def rebuild_index_from_store(self) -> tuple[int, int]:
        """Load persisted chunks from SQLite, refit indices, and persist vector artifacts."""
        with self._lock:
            chunks = self.store.load_all_chunks()
            embeddings = self.retriever.index(chunks)
            if chunks:
                self.store.save_vector_artifacts(
                    citation_keys=[c.citation_key for c in chunks],
                    embeddings=embeddings,
                    meta={
                        "embedding_backend": self.embedding_backend.name,
                        "chunk_count": len(chunks),
                        "vector_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
                    },
                )
            else:
                self.store.clear_vector_artifacts()
            return self.store.counts()

    def load_corpus_directory(self, corpus_dir: Path | None = None) -> int:
        """Ingest all .md, .txt, and .pdf files in a directory and rebuild index once."""
        target = Path(corpus_dir or self.settings.corpus_dir)
        if not target.is_dir():
            return 0
        supported = {".md", ".markdown", ".txt", ".pdf"}
        files = sorted(p for p in target.iterdir() if p.is_file() and p.suffix.lower() in supported)
        with self._lock:
            for file_path in files:
                doc, chunks = self.chunker.chunk_file(file_path)
                self.store.upsert_document(doc, chunks)
            self.rebuild_index_from_store()
        return len(files)

    def ingest_text(
        self,
        *,
        doc_id: str,
        title: str,
        content: str,
        source_type: str = "markdown",
        metadata: dict[str, Any] | None = None,
    ) -> IngestResponse:
        """Chunk, store, and index raw markdown or text."""
        safe_doc_id = validate_doc_id(doc_id)
        doc, chunks = self.chunker.chunk_text(
            doc_id=safe_doc_id,
            title=title,
            content=content,
            source_type=source_type,
            metadata=metadata,
        )
        with self._lock:
            self.store.upsert_document(doc, chunks)
            total_docs, total_chunks = self.rebuild_index_from_store()
        return IngestResponse(
            document=doc,
            chunks_indexed=len(chunks),
            total_documents=total_docs,
            total_chunks=total_chunks,
        )

    def ingest_file_bytes(
        self,
        *,
        filename: str,
        raw_bytes: bytes,
        doc_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResponse:
        """Ingest uploaded .md, .txt, or .pdf bytes with strict filename & doc_id sanitization."""
        safe_filename = Path(filename.replace("\\", "/")).name or "upload.txt"
        path = Path(safe_filename)
        suffix = path.suffix.lower()
        if suffix not in {".md", ".markdown", ".txt", ".pdf"}:
            raise ValueError(f"Unsupported file extension '{suffix}'. Use .md, .txt, or .pdf.")

        resolved_doc_id = (
            validate_doc_id(doc_id)
            if doc_id
            else validate_doc_id(self.chunker._slugify(path.stem))
        )
        meta = {"filename": safe_filename, **(metadata or {})}

        if suffix == ".pdf":
            resolved_title = title or path.stem.replace("_", " ").replace("-", " ").title()
            doc, chunks = self.chunker.chunk_pdf_bytes(
                doc_id=resolved_doc_id,
                title=resolved_title,
                pdf_bytes=raw_bytes,
                metadata=meta,
            )
        else:
            text = raw_bytes.decode("utf-8", errors="replace")
            source_type = "markdown" if suffix in {".md", ".markdown"} else "text"
            resolved_title = title or self.chunker._infer_title(text, fallback=path.stem)
            doc, chunks = self.chunker.chunk_text(
                doc_id=resolved_doc_id,
                title=resolved_title,
                content=text,
                source_type=source_type,
                metadata=meta,
            )

        with self._lock:
            self.store.upsert_document(doc, chunks)
            total_docs, total_chunks = self.rebuild_index_from_store()
        return IngestResponse(
            document=doc,
            chunks_indexed=len(chunks),
            total_documents=total_docs,
            total_chunks=total_chunks,
        )

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by doc_id and refresh the in-memory & vector indices."""
        with self._lock:
            deleted = self.store.delete_document(doc_id)
            if deleted:
                self.rebuild_index_from_store()
            return deleted

    def list_documents(self) -> list[Document]:
        """List all indexed documents."""
        return self.store.list_documents()

    def counts(self) -> tuple[int, int]:
        """Return (document_count, chunk_count)."""
        return self.store.counts()

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Execute full retrieval -> abstention gate -> grounded synthesis -> verification."""
        t0 = time.perf_counter()
        language = detect_language(request.query)

        t_ret0 = time.perf_counter()
        hits = self.retriever.retrieve(
            request.query,
            mode=request.mode,
            top_k=request.top_k,
        )
        retrieval_ms = (time.perf_counter() - t_ret0) * 1000.0

        threshold = (
            request.min_confidence
            if request.min_confidence is not None
            else self.settings.min_confidence_threshold
        )
        decision = self.grounding.evaluate_confidence(
            request.query,
            hits,
            threshold_override=threshold,
        )

        if decision.should_abstain:
            answer = self.grounding.build_abstention_message(
                language=language,
                confidence=decision.confidence_score,
                threshold=threshold,
            )
            total_ms = (time.perf_counter() - t0) * 1000.0
            return QueryResponse(
                query=request.query,
                mode=request.mode,
                language=language,
                answer=answer,
                abstained=True,
                abstention_reason=decision.reason,
                confidence_score=decision.confidence_score,
                faithfulness_score=1.0,
                supported_claims=0,
                total_claims=0,
                provider=self.provider.name,
                citations=[],
                hits=hits,
                latency=LatencyBreakdown(
                    retrieval_ms=round(retrieval_ms, 2),
                    generation_ms=0.0,
                    verification_ms=0.0,
                    total_ms=round(total_ms, 2),
                ),
            )

        t_gen0 = time.perf_counter()
        answer = await self.provider.generate(
            query=request.query,
            hits=hits,
            language=language,
        )
        generation_ms = (time.perf_counter() - t_gen0) * 1000.0

        t_ver0 = time.perf_counter()
        verification = self.grounding.verify_answer(answer, hits)
        verification_ms = (time.perf_counter() - t_ver0) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        return QueryResponse(
            query=request.query,
            mode=request.mode,
            language=language,
            answer=answer,
            abstained=False,
            abstention_reason=None,
            confidence_score=decision.confidence_score,
            faithfulness_score=verification.faithfulness_score,
            supported_claims=verification.supported_claims,
            total_claims=verification.total_claims,
            provider=self.provider.name,
            citations=verification.citations,
            hits=hits,
            latency=LatencyBreakdown(
                retrieval_ms=round(retrieval_ms, 2),
                generation_ms=round(generation_ms, 2),
                verification_ms=round(verification_ms, 2),
                total_ms=round(total_ms, 2),
            ),
        )

    async def stream_query(
        self, request: QueryRequest
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield structured (event_name, payload) tuples for Server-Sent Events streaming."""
        t0 = time.perf_counter()
        language = detect_language(request.query)

        t_ret0 = time.perf_counter()
        hits = self.retriever.retrieve(
            request.query,
            mode=request.mode,
            top_k=request.top_k,
        )
        retrieval_ms = (time.perf_counter() - t_ret0) * 1000.0

        threshold = (
            request.min_confidence
            if request.min_confidence is not None
            else self.settings.min_confidence_threshold
        )
        decision = self.grounding.evaluate_confidence(
            request.query,
            hits,
            threshold_override=threshold,
        )

        yield (
            "metadata",
            {
                "query": request.query,
                "mode": request.mode,
                "language": language,
                "abstained": decision.should_abstain,
                "confidence_score": decision.confidence_score,
                "retrieval_ms": round(retrieval_ms, 2),
                "hits": [h.model_dump(mode="json") for h in hits],
            },
        )

        if decision.should_abstain:
            answer = self.grounding.build_abstention_message(
                language=language,
                confidence=decision.confidence_score,
                threshold=threshold,
            )
            yield ("token", {"delta": answer})
            total_ms = (time.perf_counter() - t0) * 1000.0
            final_resp = QueryResponse(
                query=request.query,
                mode=request.mode,
                language=language,
                answer=answer,
                abstained=True,
                abstention_reason=decision.reason,
                confidence_score=decision.confidence_score,
                faithfulness_score=1.0,
                supported_claims=0,
                total_claims=0,
                provider=self.provider.name,
                citations=[],
                hits=hits,
                latency=LatencyBreakdown(
                    retrieval_ms=round(retrieval_ms, 2),
                    generation_ms=0.0,
                    verification_ms=0.0,
                    total_ms=round(total_ms, 2),
                ),
            )
            yield ("done", final_resp.model_dump(mode="json"))
            return

        t_gen0 = time.perf_counter()
        chunks_buf: list[str] = []
        async for delta in self.provider.stream(
            query=request.query,
            hits=hits,
            language=language,
        ):
            chunks_buf.append(delta)
            yield ("token", {"delta": delta})
        generation_ms = (time.perf_counter() - t_gen0) * 1000.0
        answer = "".join(chunks_buf).strip()

        t_ver0 = time.perf_counter()
        verification = self.grounding.verify_answer(answer, hits)
        verification_ms = (time.perf_counter() - t_ver0) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        final_resp = QueryResponse(
            query=request.query,
            mode=request.mode,
            language=language,
            answer=answer,
            abstained=False,
            abstention_reason=None,
            confidence_score=decision.confidence_score,
            faithfulness_score=verification.faithfulness_score,
            supported_claims=verification.supported_claims,
            total_claims=verification.total_claims,
            provider=self.provider.name,
            citations=verification.citations,
            hits=hits,
            latency=LatencyBreakdown(
                retrieval_ms=round(retrieval_ms, 2),
                generation_ms=round(generation_ms, 2),
                verification_ms=round(verification_ms, 2),
                total_ms=round(total_ms, 2),
            ),
        )
        yield ("done", final_resp.model_dump(mode="json"))
