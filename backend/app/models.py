from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    title: str
    source: str
    position: int
    text: str
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    vector_score: float
    keyword_score: float
    lexical_overlap: float
    rank: int

    def citation(self, number: int) -> Dict[str, Any]:
        return {
            "number": number,
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "title": self.chunk.title,
            "source": self.chunk.source,
            "position": self.chunk.position,
            "score": round(self.score, 4),
            "excerpt": self.chunk.text[:420],
        }
