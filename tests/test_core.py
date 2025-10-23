from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from backend.app.evaluation import run_evaluation
from backend.app.rag import RAGService
from backend.app.settings import Settings
from backend.app.text_processing import chunk_text


class RAGCoreTests(unittest.TestCase):
    def make_service(self) -> RAGService:
        temp_dir = Path(tempfile.mkdtemp())
        settings = Settings(data_dir=temp_dir, chunk_size=240, chunk_overlap=40, top_k=3)
        return RAGService(settings=settings)

    def test_chunk_text_preserves_metadata(self) -> None:
        chunks = chunk_text(
            "Alpha beta gamma.\n\nRetrieval augmented generation uses citations.",
            document_id="doc-1",
            title="Test Doc",
            source="test.txt",
            chunk_size=80,
            chunk_overlap=10,
        )
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0].document_id, "doc-1")
        self.assertEqual(chunks[0].title, "Test Doc")
        self.assertIn("Alpha", chunks[0].text)

    def test_ingest_and_answer_with_citation(self) -> None:
        service = self.make_service()
        service.ingest_bytes(
            "policy.txt",
            (
                "The assistant should fail closed when evidence is missing. "
                "It must say the answer is not available from indexed documents."
            ).encode("utf-8"),
        )
        response = service.ask("What should the assistant do when evidence is missing?")
        self.assertIn("indexed documents", response["answer"])
        self.assertGreaterEqual(len(response["citations"]), 1)
        self.assertFalse(response["diagnostics"]["refused"])

    def test_evaluation_runs_against_sample_docs(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            docs = root_path / "docs"
            docs.mkdir()
            dataset = root_path / "dataset.jsonl"
            docs.joinpath("rag.txt").write_text(
                "RAG retrieves external context before generation. "
                "This improves groundedness and supports source-aware answers.",
                encoding="utf-8",
            )
            dataset.write_text(
                '{"id":"case-1","question":"Why is RAG useful?",'
                '"required_facts":["retrieves external context","improves groundedness"]}\n',
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {
                    "RAG_DATA_DIR": str(root_path / "data"),
                    "CHUNK_SIZE": "240",
                    "CHUNK_OVERLAP": "20",
                    "LLM_PROVIDER": "offline",
                },
            ):
                summary = run_evaluation(dataset, docs, reset=True)

        self.assertEqual(summary["cases"], 1)
        self.assertIn("average_fact_recall", summary)


if __name__ == "__main__":
    unittest.main()
