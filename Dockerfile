FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    RAG_HOST=0.0.0.0 \
    RAG_PORT=8000 \
    RAG_INDEX_DIR=/app/data/index \
    RAG_CORPUS_DIR=/app/data/corpus

WORKDIR /app

# Create non-root user for container security
RUN useradd --create-home --shell /bin/bash appuser

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/corpus/ ./data/corpus/
COPY eval/ ./eval/

RUN mkdir -p /app/data/index && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "hybrid_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
