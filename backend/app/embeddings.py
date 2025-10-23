from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable, List, Sequence


TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_+-]{1,}")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


def tokenize(text: str, *, keep_stopwords: bool = False) -> List[str]:
    tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
    if keep_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS]


class HashingEmbedder:
    """Deterministic local embedding baseline.

    This is not a replacement for a high-quality embedding model, but it makes the
    project runnable without network access or API keys. The provider interface can
    later be swapped for OpenAI, Cohere, Voyage, BGE, or another embedding service.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        counts = Counter(tokenize(text, keep_stopwords=False))
        vector = [0.0] * self.dim
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dim
            sign = -1.0 if (value >> 8) & 1 else 1.0
            vector[index] += sign * (1.0 + math.log(count))
        return normalize(vector)

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]


class SentenceTransformerEmbedder:
    """Optional real embedding adapter backed by sentence-transformers.

    Mirrors the HashingEmbedder interface (``embed`` / ``embed_many``) so it can be
    swapped in via the ``make_embedder`` factory. The model is loaded lazily on first
    use, and the heavy dependency is imported inside the method (guarded like pypdf
    elsewhere) so the default install stays lightweight and offline.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers embeddings require the optional dependency: "
                    "pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> List[float]:
        model = self._get_model()
        vector = model.encode(text)
        return normalize([float(value) for value in vector])

    def embed_many(self, texts: Iterable[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]


def make_embedder(settings):
    """Return an embedder based on ``settings.embedding_provider`` (duck-typed).

    Defaults to the deterministic, offline HashingEmbedder. ``Settings`` is not
    imported here to avoid any circular import; attributes are read off the passed
    object.
    """

    provider = getattr(settings, "embedding_provider", "hashing")
    if provider in ("sentence-transformers", "st", "sbert"):
        return SentenceTransformerEmbedder(settings.embedding_model)
    return HashingEmbedder(dim=settings.embedding_dim)


def normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [float(value / norm) for value in vector]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))
