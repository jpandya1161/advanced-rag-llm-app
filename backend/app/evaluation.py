from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .rag import RAGService


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return rows


def run_evaluation(dataset: Path, docs_dir: Path, reset: bool = False) -> Dict[str, Any]:
    service = RAGService()
    if reset:
        service.clear()
    if service.store.count_chunks() == 0 and docs_dir.exists():
        service.ingest_directory(docs_dir)

    cases = load_jsonl(dataset)
    results = []
    for case in cases:
        response = service.ask(case["question"])
        answer = response["answer"].lower()
        required_facts = [fact.lower() for fact in case.get("required_facts", [])]
        matched_facts = [fact for fact in required_facts if _contains_fact(answer, fact)]
        refused = response["diagnostics"].get("refused", False)
        expected_refusal = case.get("scenario") == "unanswerable"
        citations_ok = bool(response["citations"]) or refused
        results.append(
            {
                "id": case.get("id"),
                "scenario": case.get("scenario", "answerable"),
                "question": case["question"],
                "fact_recall": (
                    len(matched_facts) / len(required_facts) if required_facts else None
                ),
                "matched_facts": matched_facts,
                "citation_present_or_refused": citations_ok,
                "expected_refusal": expected_refusal,
                "refusal_correct": refused if expected_refusal else not refused,
                "refused": refused,
                "latency_ms": response["diagnostics"].get("latency_ms"),
            }
        )

    fact_cases = [item for item in results if item["fact_recall"] is not None]
    unanswerable_cases = [item for item in results if item["expected_refusal"]]
    retrieval_cases = [item for item in results if not item["expected_refusal"]]
    summary = {
        "cases": len(results),
        "fact_scored_cases": len(fact_cases),
        "average_fact_recall": _average(float(item["fact_recall"]) for item in fact_cases),
        "citation_or_refusal_rate": _average(
            1.0 if item["citation_present_or_refused"] else 0.0 for item in results
        ),
        "unanswerable_refusal_accuracy": _average(
            1.0 if item["refusal_correct"] else 0.0 for item in unanswerable_cases
        ),
        "answerable_non_refusal_rate": _average(
            1.0 if item["refusal_correct"] else 0.0 for item in retrieval_cases
        ),
        "average_latency_ms": _average(item["latency_ms"] for item in results),
        "results": results,
    }
    return summary


def _contains_fact(answer: str, fact: str) -> bool:
    terms = [term for term in fact.split() if len(term) > 3]
    if not terms:
        return fact in answer
    return sum(1 for term in terms if term in answer) / len(terms) >= 0.5


def _average(values: Iterable[float]) -> float:
    value_list = [float(value) for value in values]
    if not value_list:
        return 0.0
    return round(sum(value_list) / len(value_list), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG sample evaluation set.")
    parser.add_argument("dataset", nargs="?", default="evals/sample_questions.jsonl")
    parser.add_argument("--docs", default="sample_docs", help="Directory to ingest when the index is empty.")
    parser.add_argument("--reset", action="store_true", help="Clear the local index before evaluation.")
    parser.add_argument("--output", default="evals/results.json", help="Path to write JSON results.")
    args = parser.parse_args()

    summary = run_evaluation(Path(args.dataset), Path(args.docs), reset=args.reset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    print(f"Wrote detailed results to {output_path}")


if __name__ == "__main__":
    main()
