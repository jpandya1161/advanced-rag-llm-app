from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.comparison import RETRIEVAL_MODES, _ndcg_at_k, _reciprocal_rank, run_comparison
from backend.app.evaluation import load_jsonl, run_evaluation


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "sample_questions.jsonl"
DOCS = ROOT / "sample_docs"


class EvaluationDatasetTests(unittest.TestCase):
    def test_corpus_and_dataset_cover_required_scenarios(self) -> None:
        cases = load_jsonl(DATASET)
        documents = sorted(DOCS.glob("*.txt"))

        self.assertGreaterEqual(len(documents), 8)
        self.assertGreaterEqual(len(cases), 30)
        self.assertEqual(
            {case["scenario"] for case in cases},
            {"answerable", "ambiguous", "conflicting", "unanswerable"},
        )
        self.assertGreaterEqual(
            sum(case["scenario"] == "unanswerable" for case in cases),
            5,
        )

        source_names = {path.name for path in documents}
        referenced_sources = {
            source for case in cases for source in case.get("expected_sources", [])
        }
        self.assertTrue(referenced_sources.issubset(source_names))

    def test_source_ranking_metrics(self) -> None:
        ranking = ["unrelated.txt", "primary.txt", "secondary.txt"]
        relevant = {"primary.txt", "secondary.txt"}

        self.assertEqual(_reciprocal_rank(ranking, relevant), 0.5)
        self.assertAlmostEqual(_ndcg_at_k(ranking, relevant, 3), 0.6934, places=4)

    def test_repeatable_comparison_writes_all_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "comparison.json"
            first = run_comparison(DATASET, DOCS, output, top_k=5)
            written = json.loads(output.read_text(encoding="utf-8"))
            second = run_comparison(DATASET, DOCS, output, top_k=5)

        self.assertEqual(first, written)
        self.assertEqual(first, second)
        self.assertEqual(first["benchmark"]["cases"], 41)
        self.assertGreaterEqual(first["benchmark"]["documents"], 8)
        self.assertEqual(
            {item["name"] for item in first["configurations"]},
            set(RETRIEVAL_MODES),
        )
        self.assertIn(first["winner"], RETRIEVAL_MODES)
        for configuration in first["configurations"]:
            summary = configuration["summary"]
            self.assertEqual(summary["cases"], 41)
            for metric in (
                "recall_at_k",
                "mrr",
                "ndcg_at_k",
                "unanswerable_abstention_accuracy",
                "benchmark_score",
            ):
                self.assertGreaterEqual(summary[metric], 0.0)
                self.assertLessEqual(summary[metric], 1.0)

    def test_unanswerable_cases_are_not_scored_as_missing_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs = root / "docs"
            docs.mkdir()
            docs.joinpath("policy.txt").write_text(
                "The project charter is due on 5 June.", encoding="utf-8"
            )
            dataset = root / "cases.jsonl"
            dataset.write_text(
                '{"id":"known","question":"When is the project charter due?",'
                '"required_facts":["5 June"],"scenario":"answerable"}\n'
                '{"id":"unknown","question":"What colour is the parking permit?",'
                '"required_facts":[],"scenario":"unanswerable"}\n',
                encoding="utf-8",
            )
            settings_dir = root / "data"
            with patch.dict("os.environ", {"RAG_DATA_DIR": str(settings_dir)}):
                summary = run_evaluation(dataset, docs, reset=True)

        self.assertEqual(summary["fact_scored_cases"], 1)
        self.assertIsNone(summary["results"][1]["fact_recall"])
        self.assertEqual(summary["average_fact_recall"], 1.0)

    def test_hybrid_refusal_rejects_low_evidence_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs = root / "docs"
            docs.mkdir()
            docs.joinpath("policy.txt").write_text(
                "The project charter is due on 5 June.", encoding="utf-8"
            )
            dataset = root / "cases.jsonl"
            dataset.write_text(
                '{"id":"unknown","question":"How many qubits are available on the quantum computer?",'
                '"required_facts":[],"scenario":"unanswerable"}\n',
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"RAG_DATA_DIR": str(root / "data")}):
                summary = run_evaluation(dataset, docs, reset=True)

        self.assertEqual(summary["unanswerable_refusal_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
