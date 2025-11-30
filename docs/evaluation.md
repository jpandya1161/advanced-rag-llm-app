# Evaluation Guide

The evaluation layer should measure whether the RAG system retrieves the right evidence and whether the LLM produces a correct, grounded answer.

## Dataset Format

Evaluation questions are stored as JSON Lines in `evals/sample_questions.jsonl`. Each line is one test case.

Recommended fields:

- `id`: Stable test identifier.
- `question`: User-facing question.
- `expected_answer`: Short reference answer or required facts.
- `expected_sources`: Source titles or document identifiers expected in retrieval.
- `required_facts`: Atomic facts that must appear in or be implied by the answer.
- `category`: The capability under test, such as factual lookup, comparison, refusal, or synthesis.
- `difficulty`: `easy`, `medium`, or `hard`.

## Metrics

| Metric | Type | Description |
| --- | --- | --- |
| retrieval_precision_at_k | Automatic/manual | Share of top-k retrieved chunks that are relevant. |
| context_recall | Manual or LLM-judged | Whether retrieved context contains all facts needed to answer. |
| answer_correctness | Manual or LLM-judged | Whether the answer matches the expected facts. |
| groundedness | Manual or LLM-judged | Whether every substantive claim is supported by retrieved context. |
| citation_accuracy | Manual | Whether citations point to documents that support the answer. |
| refusal_quality | Manual | Whether the system refuses gracefully when context is insufficient. |
| latency_ms | Automatic | End-to-end time for retrieval and generation. |
| estimated_cost_usd | Automatic | Approximate model and embedding cost for each query. |

## Suggested Scoring Rubric

Use a 0-2 scale for manual review:

| Score | Meaning |
| --- | --- |
| 0 | Incorrect, unsupported, or missing. |
| 1 | Partially correct but incomplete, vague, or weakly supported. |
| 2 | Correct, complete, and supported by retrieved context. |

Track separate scores for retrieval, answer correctness, groundedness, and citation quality. A strong answer should not receive a high groundedness score if the correct evidence was not retrieved.

## Baseline Evaluation Procedure

1. Run all questions with fixed retrieval and generation settings.
2. Save raw retrieved chunks, model answers, citations, latency, and cost estimates.
3. Compare retrieved sources with `expected_sources`.
4. Check whether `required_facts` are present in the answer.
5. Manually review groundedness and citation quality for a small dataset.
6. Record common failure patterns in an error analysis note.

The implemented baseline can be run with:

```bash
python3 -m backend.app.evaluation evals/sample_questions.jsonl --reset --docs sample_docs --output evals/results.json
```

The current sample run writes `evals/results.json` and reports fact recall,
citation-or-refusal coverage, and average latency. These are lightweight development
metrics; manual review is still required for a final internship report.

## Common Failure Categories

- Missing evidence because chunks are too large, too small, or poorly overlapped.
- Retrieval returns semantically similar but factually wrong passages.
- The LLM answers from prior knowledge rather than retrieved context.
- Citations are present but point to irrelevant chunks.
- Multi-hop questions retrieve only one of several required facts.
- The system does not refuse when the corpus lacks enough information.
