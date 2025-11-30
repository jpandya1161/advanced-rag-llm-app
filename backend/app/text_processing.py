from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Iterable, List

from .models import DocumentChunk


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".html", ".py"}


def stable_id(*parts: str, length: int = 16) -> str:
    digest = hashlib.sha256("::".join(parts).encode("utf-8", errors="ignore")).hexdigest()
    return digest[:length]


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        return _extract_pdf_text(content)
    if extension in SUPPORTED_TEXT_EXTENSIONS or not extension:
        return normalize_text(content.decode("utf-8", errors="replace"))
    raise ValueError(f"Unsupported file type '{extension}'. Use text, markdown, JSON, CSV, HTML, or PDF.")


def _extract_pdf_text(content: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise ValueError("PDF ingestion requires the optional dependency: pip install pypdf") from exc

    reader = PdfReader(io.BytesIO(content))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n\n[Page {index}]\n{text}")
    return normalize_text("\n".join(pages))


def chunk_text(
    text: str,
    *,
    document_id: str,
    title: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[DocumentChunk]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    paragraphs = _split_paragraphs(normalized)
    pieces: List[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                pieces.append(current.strip())
                current = ""
            pieces.extend(_split_long_text(paragraph, chunk_size, chunk_overlap))
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            pieces.append(current.strip())
            overlap = _tail_overlap(current, chunk_overlap)
            current = f"{overlap}\n\n{paragraph}".strip() if overlap else paragraph
    if current:
        pieces.append(current.strip())

    chunks: List[DocumentChunk] = []
    for position, piece in enumerate(pieces):
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document_id}#chunk-{position:04d}",
                document_id=document_id,
                title=title,
                source=source,
                position=position,
                text=piece,
                metadata={"chars": len(piece)},
            )
        )
    return chunks


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if paragraphs:
        return paragraphs
    return [text]


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        projected = current_len + len(word) + (1 if current else 0)
        if current and projected > chunk_size:
            piece = " ".join(current)
            chunks.append(piece)
            overlap_words = _tail_words(piece, chunk_overlap)
            current = overlap_words + [word]
            current_len = len(" ".join(current))
        else:
            current.append(word)
            current_len = projected
    if current:
        chunks.append(" ".join(current))
    return chunks


def _tail_overlap(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text if len(text) <= max_chars else ""
    tail = text[-max_chars:]
    boundary = tail.find(" ")
    return tail[boundary + 1 :].strip() if boundary >= 0 else tail.strip()


def _tail_words(text: str, max_chars: int) -> List[str]:
    if max_chars <= 0:
        return []
    words = text.split()
    selected: List[str] = []
    total = 0
    for word in reversed(words):
        projected = total + len(word) + (1 if selected else 0)
        if projected > max_chars:
            break
        selected.append(word)
        total = projected
    return list(reversed(selected))


def iter_supported_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from iter_supported_files(sorted(path.iterdir()))
        elif path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS or path.suffix.lower() == ".pdf":
            yield path
