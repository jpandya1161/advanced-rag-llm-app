from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(".rag_data")
    chunk_size: int = 900
    chunk_overlap: int = 160
    embedding_dim: int = 384
    top_k: int = 5
    min_hybrid_score: float = 0.10
    max_refusal_keyword_score: float = 6.0
    llm_provider: str = "offline"
    openai_api_key: str = ""
    # Models whose id starts with "gpt-5" additionally trigger the
    # openai_reasoning_effort field below.
    openai_model: str = "gpt-4o-mini"
    openai_reasoning_effort: str = "low"
    openai_max_output_tokens: int = 700
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    embedding_provider: str = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cors_allow_origins: str = "http://127.0.0.1:8000,http://localhost:8000"
    max_upload_mb: int = 10

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(override=False)
        return cls(
            data_dir=Path(os.getenv("RAG_DATA_DIR", ".rag_data")),
            chunk_size=_int_env("CHUNK_SIZE", 900),
            chunk_overlap=_int_env("CHUNK_OVERLAP", 160),
            embedding_dim=_int_env("EMBEDDING_DIM", 384),
            top_k=_int_env("TOP_K", 5),
            min_hybrid_score=_float_env("MIN_HYBRID_SCORE", 0.10),
            max_refusal_keyword_score=_float_env("MAX_REFUSAL_KEYWORD_SCORE", 6.0),
            llm_provider=os.getenv("LLM_PROVIDER", "offline").strip().lower() or "offline",
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "low"),
            openai_max_output_tokens=_int_env("OPENAI_MAX_OUTPUT_TOKENS", 700),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "hashing"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            cors_allow_origins=os.getenv(
                "CORS_ALLOW_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000"
            ),
            max_upload_mb=_int_env("MAX_UPLOAD_MB", 10),
        )

    @property
    def database_path(self) -> Path:
        return self.data_dir / "rag_index.sqlite3"
