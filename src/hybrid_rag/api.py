"""FastAPI application for Hybrid RAG Eval with REST endpoints, SSE streaming, and Web UI."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import ValidationError

from hybrid_rag import __version__
from hybrid_rag.config import Settings, get_settings
from hybrid_rag.engine import HybridRAGEngine
from hybrid_rag.schemas import (
    Document,
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        engine = HybridRAGEngine(settings=cfg)
        doc_count, _ = engine.counts()
        if doc_count == 0 and cfg.auto_load_corpus and cfg.corpus_dir.is_dir():
            engine.load_corpus_directory(cfg.corpus_dir)
        application.state.engine = engine
        application.state.settings = cfg
        yield

    application = FastAPI(
        title="Hybrid RAG Eval",
        version=__version__,
        description=(
            "Production-grade Hybrid RAG Engine with Okapi BM25 + Dense Vector retrieval, "
            "Reciprocal Rank Fusion (RRF), Cross-Signal Reranking, strict inline citation "
            "grounding with automatic abstention, SSE streaming, and automated evaluation."
        ),
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _get_engine(request: Request) -> HybridRAGEngine:
        return request.app.state.engine  # type: ignore[no-any-return]

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_ui() -> HTMLResponse:
        index_html = STATIC_DIR / "index.html"
        if not index_html.is_file():
            return HTMLResponse("<h1>Hybrid RAG Eval UI not found</h1>", status_code=404)
        return HTMLResponse(index_html.read_text(encoding="utf-8"))

    @application.get("/health", response_model=HealthResponse, tags=["System"])
    async def health(request: Request) -> HealthResponse:
        engine = _get_engine(request)
        doc_count, chunk_count = engine.counts()
        return HealthResponse(
            status="ok",
            version=__version__,
            documents_indexed=doc_count,
            chunks_indexed=chunk_count,
            llm_provider=engine.provider.name,
            embedding_provider=engine.embedding_backend.name,
            index_dir=str(engine.settings.index_dir),
        )

    @application.get("/v1/documents", response_model=list[Document], tags=["Corpus"])
    async def list_documents(request: Request) -> list[Document]:
        return _get_engine(request).list_documents()

    @application.delete("/v1/documents/{doc_id}", tags=["Corpus"])
    async def delete_document(doc_id: str, request: Request) -> dict[str, object]:
        engine = _get_engine(request)
        deleted = engine.delete_document(doc_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{doc_id}' not found.",
            )
        docs, chunks = engine.counts()
        return {
            "deleted": True,
            "doc_id": doc_id,
            "total_documents": docs,
            "total_chunks": chunks,
        }

    @application.post(
        "/v1/ingest",
        response_model=IngestResponse,
        tags=["Corpus"],
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": IngestTextRequest.model_json_schema(),
                    },
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "file": {"type": "string", "format": "binary"},
                                "doc_id": {"type": "string"},
                                "title": {"type": "string"},
                            },
                            "required": ["file"],
                        }
                    },
                },
            }
        },
    )
    async def ingest_document(request: Request) -> IngestResponse:
        """Ingest raw JSON text/markdown or upload a .md / .txt / .pdf file."""
        engine = _get_engine(request)
        content_type = request.headers.get("content-type", "").lower()

        if "multipart/form-data" in content_type:
            form = await request.form()
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Multipart request must include a 'file' upload field.",
                )
            raw_bytes = await upload.read()
            if not raw_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file is empty.",
                )
            filename = getattr(upload, "filename", None) or "upload.txt"
            doc_id_raw = form.get("doc_id")
            title_raw = form.get("title")
            doc_id = str(doc_id_raw).strip() if doc_id_raw else None
            title = str(title_raw).strip() if title_raw else None
            try:
                return engine.ingest_file_bytes(
                    filename=filename,
                    raw_bytes=raw_bytes,
                    doc_id=doc_id or None,
                    title=title or None,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                ) from exc

        try:
            payload_raw = await request.json()
            payload = IngestTextRequest.model_validate(payload_raw)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=exc.errors(),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON payload: {exc}",
            ) from exc

        return engine.ingest_text(
            doc_id=payload.doc_id,
            title=payload.title,
            content=payload.content,
            source_type=payload.source_type,
            metadata=payload.metadata,
        )

    @application.post("/v1/query", response_model=QueryResponse, tags=["RAG"])
    async def query_rag(body: QueryRequest, request: Request) -> QueryResponse:
        """Run Hybrid, BM25, or Dense RAG query with inline citations and faithfulness score."""
        return await _get_engine(request).query(body)

    @application.post("/v1/query/stream", tags=["RAG"])
    async def query_rag_stream(body: QueryRequest, request: Request) -> StreamingResponse:
        """Stream retrieval hits, token deltas, and final grounded verification over SSE."""
        engine = _get_engine(request)

        async def event_generator() -> AsyncIterator[str]:
            async for event_name, payload in engine.stream_query(body):
                data_json = json.dumps(payload, ensure_ascii=False)
                yield f"event: {event_name}\ndata: {data_json}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @application.get("/v1/eval/latest", tags=["Evaluation"])
    async def get_latest_eval(request: Request) -> JSONResponse:
        """Return the latest benchmark comparison report generated by eval/run_eval.py."""
        report_path = Path(request.app.state.settings.eval_report_path)
        if not report_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Evaluation report not found at '{report_path}'. "
                    "Run `python eval/run_eval.py` to generate it."
                ),
            )
        data = json.loads(report_path.read_text(encoding="utf-8"))
        return JSONResponse(content=data)

    return application


app = create_app()
