from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

from backend.app.llm import OpenAIResponsesLLM
from backend.app.models import DocumentChunk, RetrievalResult
from backend.app.settings import Settings


def test_openai_provider_uses_responses_api() -> None:
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(output_text="Grounded answer [1]")

    class FakeOpenAI:
        def __init__(self, api_key: str):
            captured["api_key"] = api_key
            self.responses = FakeResponses()

    fake_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
    result = RetrievalResult(
        chunk=DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            title="Policy",
            source="policy.txt",
            position=0,
            text="Missing evidence requires a refusal.",
        ),
        score=1.0,
        vector_score=1.0,
        keyword_score=1.0,
        lexical_overlap=1.0,
        rank=1,
    )
    settings = Settings(
        data_dir=Path(".rag_data"),
        openai_api_key="test-key",
        openai_model="gpt-5.6-luna",
        openai_reasoning_effort="low",
        openai_max_output_tokens=500,
    )

    with patch.dict(sys.modules, {"openai": fake_module}):
        answer = OpenAIResponsesLLM().generate("What is the rule?", [result], settings)

    assert answer == "Grounded answer [1]"
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "gpt-5.6-luna"
    assert captured["reasoning"] == {"effort": "low"}
    assert captured["max_output_tokens"] == 500
    assert "Context:" in captured["input"]
    assert "grounded RAG assistant" in captured["instructions"]
