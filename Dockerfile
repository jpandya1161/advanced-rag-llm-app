# Container image for the Advanced RAG + LLM application.
# Works on Render, Railway, Fly.io, or any host that runs a Dockerfile.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLM_PROVIDER=offline \
    RAG_DATA_DIR=/app/.rag_data

# Install production dependencies (plus pypdf so PDF uploads work in the demo).
COPY requirements.txt .
RUN pip install -r requirements.txt pypdf

# Copy only what the running service needs.
COPY backend ./backend
COPY frontend ./frontend
COPY sample_docs ./sample_docs
COPY evals ./evals

EXPOSE 8000

# Hosts inject the port through $PORT; fall back to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn backend.app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
