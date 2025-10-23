from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .evaluation import run_evaluation
from .rag import RAGService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class IngestPathRequest(BaseModel):
    path: str = "sample_docs"
    reset: bool = False


app = FastAPI(title="Advanced RAG and LLM API App", version="0.1.0")
service = RAGService()

cors_allow_origins = [
    origin.strip()
    for origin in service.settings.cors_allow_origins.split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend has not been built.")
    return FileResponse(index_path)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "documents": len(service.list_documents()),
        "chunks": service.store.count_chunks(),
        "provider": service.settings.llm_provider,
    }


@app.get("/api/documents")
def documents() -> dict:
    return {"documents": service.list_documents()}


@app.delete("/api/documents")
def clear_documents() -> dict:
    service.clear()
    return {"ok": True}


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    try:
        content = await file.read()
        if len(content) > service.settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {service.settings.max_upload_mb} MB limit.",
            )
        return service.ingest_bytes(file.filename or "uploaded.txt", content)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ingest-path")
def ingest_path(request: IngestPathRequest) -> dict:
    path = Path(request.path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")
    try:
        if request.reset:
            service.clear()
        if path.is_dir():
            ingested = service.ingest_directory(path)
        else:
            ingested = [service.ingest_path(path)]
        return {"documents": ingested, "count": len(ingested)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    try:
        return service.ask(request.question, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/evaluate")
def evaluate() -> dict:
    dataset = PROJECT_ROOT / "evals" / "sample_questions.jsonl"
    docs = PROJECT_ROOT / "sample_docs"
    return run_evaluation(dataset, docs, reset=False)
