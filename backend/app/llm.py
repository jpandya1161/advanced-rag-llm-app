from __future__ import annotations

import json
import re
import urllib.request
from typing import List, Sequence

from .embeddings import tokenize
from .models import RetrievalResult
from .settings import Settings


SYSTEM_INSTRUCTIONS = """You are a grounded RAG assistant.
Answer only from the supplied context. Cite sources inline using [1], [2], etc.
If the context is insufficient, say that the answer is not available from the indexed documents.
Keep answers concise and factual."""


def build_prompt(question: str, results: Sequence[RetrievalResult]) -> str:
    return f"{SYSTEM_INSTRUCTIONS}\n\n{build_user_prompt(question, results)}"


def build_user_prompt(question: str, results: Sequence[RetrievalResult]) -> str:
    context_blocks = []
    for index, result in enumerate(results, start=1):
        context_blocks.append(
            f"[{index}] {result.chunk.title} ({result.chunk.source}, chunk {result.chunk.position})\n"
            f"{result.chunk.text}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer with citations:"
    )


class OfflineLLM:
    provider_name = "offline-extractive"

    def generate(self, question: str, results: Sequence[RetrievalResult], settings: Settings) -> str:
        del settings
        if not results:
            return "The answer is not available from the indexed documents."

        query_terms = set(tokenize(question))
        candidates = []
        for citation_number, result in enumerate(results, start=1):
            sentence, overlap = _best_sentence(result.chunk.text, query_terms)
            if sentence:
                candidates.append((overlap, -result.rank, citation_number, sentence))
        candidates.sort(reverse=True)

        answer_parts: List[str] = []
        for _, _, citation_number, sentence in candidates:
            if sentence and sentence not in answer_parts:
                answer_parts.append(f"{sentence} [{citation_number}]")
            if len(answer_parts) >= 3:
                break

        if not answer_parts:
            return "The answer is not available from the indexed documents."
        if len(answer_parts) == 1:
            return f"Based on the indexed documents, {answer_parts[0]}"
        return "Based on the indexed documents:\n\n" + "\n".join(f"- {part}" for part in answer_parts)


class OpenAIResponsesLLM:
    provider_name = "openai"

    def generate(self, question: str, results: Sequence[RetrievalResult], settings: Settings) -> str:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OpenAI provider requires: pip install openai") from exc
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

        client = OpenAI(api_key=settings.openai_api_key)
        request = {
            "model": settings.openai_model,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": build_user_prompt(question, results),
            "max_output_tokens": settings.openai_max_output_tokens,
        }
        if settings.openai_model.startswith("gpt-5") and settings.openai_reasoning_effort:
            request["reasoning"] = {"effort": settings.openai_reasoning_effort}
        response = client.responses.create(**request)
        answer = response.output_text or ""
        if not answer.strip():
            raise RuntimeError("OpenAI Responses API returned no text output")
        return answer


# Kept as an import-compatible alias for earlier project revisions.
OpenAIChatLLM = OpenAIResponsesLLM


class OllamaLLM:
    provider_name = "ollama"

    def generate(self, question: str, results: Sequence[RetrievalResult], settings: Settings) -> str:
        payload = {
            "model": settings.ollama_model,
            "prompt": build_prompt(question, results),
            "stream": False,
            "options": {"temperature": 0.1},
        }
        request = urllib.request.Request(
            f"{settings.ollama_base_url.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data.get("response", ""))


def make_llm(settings: Settings):
    if settings.llm_provider == "openai":
        return OpenAIResponsesLLM()
    if settings.llm_provider == "ollama":
        return OllamaLLM()
    return OfflineLLM()


def strip_invalid_citations(answer: str, result_count: int) -> str:
    def replace(match: re.Match[str]) -> str:
        value = int(match.group(1))
        return match.group(0) if 1 <= value <= result_count else ""

    return re.sub(r"\[(\d+)\]", replace, answer).strip()


def cited_numbers(answer: str, result_count: int) -> List[int]:
    numbers = []
    for match in re.finditer(r"\[(\d+)\]", answer):
        value = int(match.group(1))
        if 1 <= value <= result_count and value not in numbers:
            numbers.append(value)
    return numbers


def _best_sentence(text: str, query_terms: set[str]) -> tuple[str, int]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not sentences:
        return text[:260].strip(), 0
    scored = []
    for sentence in sentences:
        sentence_terms = set(tokenize(sentence))
        overlap = len(query_terms & sentence_terms)
        scored.append((overlap, len(sentence), sentence))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    best = scored[0][2]
    if len(best) > 320:
        return best[:317].rstrip() + "...", scored[0][0]
    return best, scored[0][0]
