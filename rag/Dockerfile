# RAG ingestion CLI image (Manual QA remediation Q.1, §A). A one-shot
# utility image, never a long-running service -- see the `rag-ingest`
# entry in the root docker-compose.yml (profiles: ["ingest"], so it never
# starts as part of a bare `docker compose up` and the documented
# "exactly five long-running services" topology is unchanged).
#
# Self-contained: only rag/ itself is copied in. This CLI never imports
# from services/, providers/, or orchestration/ -- it is the frozen
# Phase 3 ingestion pipeline (rag/ingest.py, rag/qdrant_store.py,
# rag/embeddings.py) plus this remediation's own rag/ingest_cli.py, and
# nothing else.
FROM python:3.12-slim

WORKDIR /app

# Same reasoning as services/istanbul-expert-b/Dockerfile: torch installed
# first, in its own layer, from the CPU-only wheel index -- the default
# PyPI index pulls in several gigabytes of unused CUDA packages.
RUN pip install --no-cache-dir --default-timeout=300 --retries=5 \
    torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --default-timeout=300 --retries=5 -r requirements.txt

COPY . ./rag
# rag/*.py uses absolute imports like `from rag import ingest` -- so /app
# (the parent of the rag/ package directory) must be on PYTHONPATH, not
# /app/rag itself.
ENV PYTHONPATH=/app
ENV EMBEDDING_CACHE_DIR=/data/model_cache
ENV HF_HOME=/data/model_cache

RUN mkdir -p /data/model_cache

ENTRYPOINT ["python", "-m", "rag.ingest_cli"]
CMD ["bootstrap"]
