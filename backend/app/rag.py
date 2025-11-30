from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .embeddings import make_embedder
from .llm import cited_numbers, make_llm, strip_invalid_citations
from .retrieval import HybridRetriever
from .settings import Settings
from .store import SQLiteRAGStore
from .text_processing import chunk_text, extract_text, iter_supported_files, stable_id


class RAGService:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings.from_env()
        self.embedder = make_embedder(self.settings)
        self.store = SQLiteRAGStore(self.settings.database_path)
        self.retriever = HybridRetriever(self.embedder)
        self.llm = make_llm(self.settings)

    def clear(self) -> None:
        self.store.clear()

    def ingest_bytes(self, filename: str, content: bytes, *, title: Optional[str] = None) -> Dict[str, Any]:
        text = extract_text(filename, content)
        if not text:
            raise ValueError("No text could be extracted from the document.")
        document_title = title or Path(filename).stem.replace("_", " ").replace("-", " ").strip() or filename
        document_id = stable_id(filename, text)
        chunks = chunk_text(
            text,
            document_id=document_id,
            title=document_title,
            source=filename,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("Document did not produce any chunks.")

        embeddings = self.embedder.embed_many(chunk.text for chunk in chunks)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        count = self.store.upsert_document(document_id, document_title, filename, chunks)
        return {
            "document_id": document_id,
            "title": document_title,
            "source": filename,
            "chunk_count": count,
        }

    def ingest_path(self, path: Path) -> Dict[str, Any]:
        content = path.read_bytes()
        return self.ingest_bytes(str(path), content, title=path.stem.replace("_", " "))

    def ingest_directory(self, directory: Path) -> List[Dict[str, Any]]:
        ingested = []
        for path in iter_supported_files([directory]):
            ingested.append(self.ingest_path(path))
        return ingested

    def list_documents(self) -> List[Dict[str, Any]]:
        return self.store.list_documents()

    def ask(self, question: str, *, top_k: Optional[int] = None) -> Dict[str, Any]:
        started = time.perf_counter()
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question is required.")

        chunks = self.store.all_chunks()
        results = self.retriever.search(normalized_question, chunks, top_k or self.settings.top_k)
        if self._should_refuse(results):
            return {
                "answer": "The answer is not available from the indexed documents.",
                "citations": [],
                "sources": [result.citation(index) for index, result in enumerate(results, start=1)],
                "diagnostics": self._diagnostics(started, len(chunks), len(results), refused=True),
            }

        raw_answer = self.llm.generate(normalized_question, results, self.settings)
        answer = strip_invalid_citations(raw_answer, len(results))
        cited = cited_numbers(answer, len(results))
        uncited = bool(results) and not cited

        citations = [results[number - 1].citation(number) for number in cited]
        return {
            "answer": answer,
            "citations": citations,
            "sources": [result.citation(index) for index, result in enumerate(results, start=1)],
            "diagnostics": self._diagnostics(
                started, len(chunks), len(results), refused=False, uncited=uncited
            ),
        }

    def _diagnostics(
        self,
        started: float,
        chunk_count: int,
        retrieved_count: int,
        *,
        refused: bool,
        uncited: bool = False,
    ) -> Dict[str, Any]:
        return {
            "provider": getattr(self.llm, "provider_name", self.settings.llm_provider),
            "chunk_count": chunk_count,
            "retrieved_count": retrieved_count,
            "top_k": self.settings.top_k,
            "refused": refused,
            "uncited": uncited,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def _should_refuse(self, results) -> bool:
        if not results:
            return True
        best = results[0]
        return (
            best.score < self.settings.min_hybrid_score
            and best.vector_score < 0.20
            and best.lexical_overlap <= 0.50
            and best.keyword_score <= self.settings.max_refusal_keyword_score
        )
