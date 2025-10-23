from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .models import DocumentChunk


class SQLiteRAGStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
                """
            )

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")

    def upsert_document(self, document_id: str, title: str, source: str, chunks: Iterable[DocumentChunk]) -> int:
        chunk_list = list(chunks)
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            connection.execute(
                """
                INSERT INTO documents(document_id, title, source, chunk_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(document_id)
                DO UPDATE SET title = excluded.title, source = excluded.source, chunk_count = excluded.chunk_count
                """,
                (document_id, title, source, len(chunk_list)),
            )
            connection.executemany(
                """
                INSERT INTO chunks(chunk_id, document_id, title, source, position, text, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.title,
                        chunk.source,
                        chunk.position,
                        chunk.text,
                        json.dumps(chunk.embedding),
                        json.dumps(chunk.metadata),
                    )
                    for chunk in chunk_list
                ],
            )
        return len(chunk_list)

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT document_id, title, source, chunk_count, created_at
                FROM documents
                ORDER BY created_at DESC, title ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def all_chunks(self) -> List[DocumentChunk]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, document_id, title, source, position, text, embedding, metadata
                FROM chunks
                ORDER BY document_id, position
                """
            ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def count_chunks(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return int(row["count"])

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            title=row["title"],
            source=row["source"],
            position=int(row["position"]),
            text=row["text"],
            embedding=[float(value) for value in json.loads(row["embedding"])],
            metadata=json.loads(row["metadata"]),
        )
