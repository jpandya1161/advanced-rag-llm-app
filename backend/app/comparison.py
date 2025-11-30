from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .evaluation import load_jsonl
from .models import DocumentChunk, RetrievalResult
from .rag import RAGService
from .settings import Settings


RETRIEVAL_MODES = ("vector_only", "keyword_only", "hybrid")


def run_comparison(
    dataset: Path,
    docs_dir: Path,
    output_path: Path,
    *,
    top_k: int = 5,
    embedding_provider: str = "hashing",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """Run a deterministic source-level retrieval comparison and write its results."""
    cases = load_jsonl(dataset)
    _validate_cases(cases)

    with tempfile.TemporaryDirectory(prefix="rag-comparison-") as temp_dir:
        settings = Settings(
            data_dir=Path(temp_dir),
            chunk_size=900,
            chunk_overlap=160,
            embedding_dim=384,
            top_k=top_k,
            llm_provider="offline",
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
        service = RAGService(settings=settings)
        documents = service.ingest_directory(docs_dir)
        chunks = service.store.all_chunks()
        known_sources = {Path(item["source"]).name for item in documents}
        _validate_sources(cases, known_sources)

        configurations = [
            _evaluate_mode(mode, cases, chunks, service, top_k) for mode in RETRIEVAL_MODES
        ]

    ranked = sorted(
        configurations,
        key=lambda item: (
            item["summary"]["benchmark_score"],
            item["summary"]["ndcg_at_k"],
            item["summary"]["mrr"],
        ),
        reverse=True,
    )
    result = {
        "benchmark": {
            "dataset": str(dataset),
            "documents_directory": str(docs_dir),
            "cases": len(cases),
            "documents": len(documents),
            "chunks": len(chunks),
            "top_k": top_k,
            "chunk_size": 900,
            "chunk_overlap": 160,
            "embedding_dimension": 384,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model if embedding_provider != "hashing" else None,
            "ranking_unit": "unique source document",
        },
        "configurations": configurations,
        "winner": ranked[0]["name"] if ranked else None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _evaluate_mode(
    mode: str,
    cases: Sequence[Dict[str, Any]],
    chunks: Sequence[DocumentChunk],
    service: RAGService,
    top_k: int,
) -> Dict[str, Any]:
    case_results: List[Dict[str, Any]] = []
    for case in cases:
        ranked_results = _rank_results(mode, case["question"], chunks, service)
        source_ranking = _unique_source_ranking(ranked_results)
        relevant_sources = set(case["expected_sources"])
        top_sources = source_ranking[:top_k]

        if relevant_sources:
            recall = len(relevant_sources.intersection(top_sources)) / len(relevant_sources)
            reciprocal_rank = _reciprocal_rank(source_ranking, relevant_sources)
            ndcg = _ndcg_at_k(source_ranking, relevant_sources, top_k)
            abstained = False
            abstention_correct = None
        else:
            recall = None
            reciprocal_rank = None
            ndcg = None
            abstained = _should_abstain(
                mode,
                ranked_results,
                service.settings.min_hybrid_score,
                service.settings.max_refusal_keyword_score,
            )
            abstention_correct = abstained

        case_results.append(
            {
                "id": case["id"],
                "scenario": case["scenario"],
                "question": case["question"],
                "expected_sources": sorted(relevant_sources),
                "top_sources": top_sources,
                "recall_at_k": _round_optional(recall),
                "reciprocal_rank": _round_optional(reciprocal_rank),
                "ndcg_at_k": _round_optional(ndcg),
                "abstained": abstained,
                "abstention_correct": abstention_correct,
            }
        )

    answerable = [item for item in case_results if item["expected_sources"]]
    unanswerable = [item for item in case_results if not item["expected_sources"]]
    benchmark_values = [float(item["ndcg_at_k"]) for item in answerable]
    benchmark_values.extend(1.0 if item["abstention_correct"] else 0.0 for item in unanswerable)
    summary = {
        "cases": len(case_results),
        "retrieval_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "recall_at_k": _average(float(item["recall_at_k"]) for item in answerable),
        "mrr": _average(float(item["reciprocal_rank"]) for item in answerable),
        "ndcg_at_k": _average(float(item["ndcg_at_k"]) for item in answerable),
        "unanswerable_abstention_accuracy": _average(
            1.0 if item["abstention_correct"] else 0.0 for item in unanswerable
        ),
        "benchmark_score": _average(benchmark_values),
    }
    return {
        "name": mode,
        "summary": summary,
        "by_scenario": _summarize_scenarios(case_results),
        "cases": case_results,
    }


def _rank_results(
    mode: str,
    question: str,
    chunks: Sequence[DocumentChunk],
    service: RAGService,
) -> List[RetrievalResult]:
    results = service.retriever.search(question, chunks, max(len(chunks), 1))
    if mode == "vector_only":
        results.sort(
            key=lambda item: (item.vector_score, item.lexical_overlap, item.keyword_score),
            reverse=True,
        )
    elif mode == "keyword_only":
        results.sort(
            key=lambda item: (item.keyword_score, item.lexical_overlap, item.vector_score),
            reverse=True,
        )
    elif mode == "hybrid":
        results.sort(key=lambda item: item.score, reverse=True)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    return results


def _unique_source_ranking(results: Sequence[RetrievalResult]) -> List[str]:
    ranking: List[str] = []
    seen = set()
    for result in results:
        source = Path(result.chunk.source).name
        if source not in seen:
            seen.add(source)
            ranking.append(source)
    return ranking


def _should_abstain(
    mode: str,
    results: Sequence[RetrievalResult],
    min_hybrid_score: float,
    max_refusal_keyword_score: float,
) -> bool:
    if not results:
        return True
    best = results[0]
    if mode == "vector_only":
        return best.vector_score < 0.18
    if mode == "keyword_only":
        return best.keyword_score <= 0.0
    return (
        best.score < min_hybrid_score
        and best.vector_score < 0.20
        and best.lexical_overlap <= 0.50
        and best.keyword_score <= max_refusal_keyword_score
    )


def _reciprocal_rank(ranking: Sequence[str], relevant_sources: Iterable[str]) -> float:
    relevant = set(relevant_sources)
    for rank, source in enumerate(ranking, start=1):
        if source in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(ranking: Sequence[str], relevant_sources: Iterable[str], top_k: int) -> float:
    relevant = set(relevant_sources)
    gains = [1.0 if source in relevant else 0.0 for source in ranking[:top_k]]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal_hits = min(len(relevant), top_k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _summarize_scenarios(case_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in case_results:
        grouped[item["scenario"]].append(item)

    summary: Dict[str, Any] = {}
    for scenario, items in sorted(grouped.items()):
        retrieval_items = [item for item in items if item["expected_sources"]]
        abstention_items = [item for item in items if not item["expected_sources"]]
        summary[scenario] = {
            "cases": len(items),
            "recall_at_k": _average(
                float(item["recall_at_k"]) for item in retrieval_items
            )
            if retrieval_items
            else None,
            "mrr": _average(float(item["reciprocal_rank"]) for item in retrieval_items)
            if retrieval_items
            else None,
            "ndcg_at_k": _average(float(item["ndcg_at_k"]) for item in retrieval_items)
            if retrieval_items
            else None,
            "abstention_accuracy": _average(
                1.0 if item["abstention_correct"] else 0.0 for item in abstention_items
            )
            if abstention_items
            else None,
        }
    return summary


def _validate_cases(cases: Sequence[Dict[str, Any]]) -> None:
    if len(cases) < 30:
        raise ValueError("Comparison dataset must contain at least 30 cases.")
    required_scenarios = {"answerable", "ambiguous", "conflicting", "unanswerable"}
    scenarios = {case.get("scenario") for case in cases}
    missing = required_scenarios - scenarios
    if missing:
        raise ValueError(f"Dataset is missing scenarios: {', '.join(sorted(missing))}")
    required_fields = {"id", "question", "expected_sources", "scenario"}
    for case in cases:
        absent = required_fields - case.keys()
        if absent:
            raise ValueError(f"Case {case.get('id', '<unknown>')} is missing: {sorted(absent)}")


def _validate_sources(cases: Sequence[Dict[str, Any]], known_sources: Iterable[str]) -> None:
    known = set(known_sources)
    expected = {source for case in cases for source in case["expected_sources"]}
    missing = expected - known
    if missing:
        raise ValueError(f"Dataset references missing source documents: {sorted(missing)}")


def _average(values: Iterable[float]) -> float:
    values_list = list(values)
    return round(sum(values_list) / len(values_list), 4) if values_list else 0.0


def _round_optional(value: Any) -> Any:
    return round(float(value), 4) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare deterministic RAG retrieval modes.")
    parser.add_argument("--dataset", default="evals/sample_questions.jsonl")
    parser.add_argument("--docs", default="sample_docs")
    parser.add_argument("--output", default="evals/comparison_results.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-provider", default="hashing", help="hashing (default) or sentence-transformers")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    result = run_comparison(
        Path(args.dataset),
        Path(args.docs),
        Path(args.output),
        top_k=args.top_k,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
    )
    print(
        json.dumps(
            {
                "benchmark": result["benchmark"],
                "winner": result["winner"],
                "summaries": {
                    item["name"]: item["summary"] for item in result["configurations"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
