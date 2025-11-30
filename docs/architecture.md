# Architecture

This document describes the target architecture for the advanced RAG and LLM API application. It is intentionally implementation-neutral so the backend, frontend, and evaluation layers can evolve independently.

## High-Level Flow

1. Source documents are added to a controlled corpus.
2. The ingestion pipeline extracts text, cleans it, chunks it, and records metadata.
3. Each chunk is embedded through the configured embedding provider (`EMBEDDING_PROVIDER`, default `hashing`).
4. Embeddings and metadata are stored in a linear-scan vector store (brute-force cosine over a SQLite-backed chunk store); there is no ANN index, so each query scores all chunks in memory.
5. A user submits a question through the API or UI.
6. The retriever finds the most relevant chunks for the question.
7. The prompt builder formats instructions, retrieved context, and the question.
8. The LLM generates an answer with source references.
9. The application returns the answer, citations, and optional diagnostics.
10. The evaluation harness measures retrieval and answer quality against fixed examples.

## Components

| Component | Responsibility | Key Design Questions |
| --- | --- | --- |
| Document loader | Read documents from local files or configured sources | Which file types are supported? How are failed parses reported? |
| Text chunker | Split documents into retrieval-sized passages | What chunk size and overlap preserve enough context? |
| Metadata normalizer | Attach source, title, page, section, and timestamps | Which metadata is required for citations and debugging? |
| Embedding client | Convert text into dense vectors | Which provider/model balances quality, cost, and latency? |

The embedding client is selected by `EMBEDDING_PROVIDER`. The default `hashing` provider is offline, deterministic, and zero-dependency, but strongly lexical rather than semantic. Setting `EMBEDDING_PROVIDER=sentence-transformers` (with `pip install sentence-transformers` and an optional `EMBEDDING_MODEL`, default `sentence-transformers/all-MiniLM-L6-v2`) swaps in learned embeddings behind the same duck-typed interface without changing the retrieval or storage layers.
| Vector store | Persist vectors and metadata for similarity search | Should the project use local storage or a managed vector database? |
| Retriever | Return top-k context chunks for a query | Is reranking needed? How should filters work? |
| Prompt builder | Compose grounded LLM prompts | How much context fits within the target model window? |
| LLM client | Call the generation model | How are retries, timeouts, and rate limits handled? |
| Answer formatter | Return response text, citations, and diagnostics | What source format is useful to the user? |
| Evaluation runner | Execute test questions and collect metrics | Which checks are automatic and which require human review? |

## Suggested Data Contracts

### Document Chunk

```json
{
  "chunk_id": "doc-001#chunk-004",
  "source_id": "doc-001",
  "source_path": "data/policies/refund_policy.md",
  "title": "Refund Policy",
  "section": "Eligibility",
  "text": "Customers may request a refund within 30 days...",
  "metadata": {
    "page": 2,
    "created_at": "2026-07-01"
  }
}
```

### Retrieval Result

```json
{
  "chunk_id": "doc-001#chunk-004",
  "score": 0.82,
  "text": "Customers may request a refund within 30 days...",
  "source": {
    "title": "Refund Policy",
    "source_path": "data/policies/refund_policy.md",
    "section": "Eligibility"
  }
}
```

### Answer Response

```json
{
  "answer": "Customers may request a refund within 30 days if they meet the eligibility conditions.",
  "citations": [
    {
      "chunk_id": "doc-001#chunk-004",
      "title": "Refund Policy",
      "section": "Eligibility"
    }
  ],
  "diagnostics": {
    "retrieval_k": 5,
    "model": "configured-generation-model",
    "latency_ms": 1200
  }
}
```

## Design Principles

- Keep retrieval and generation separable so each can be evaluated independently.
- Store enough metadata to debug poor answers and verify citations.
- Prefer deterministic evaluation settings: low temperature, fixed top-k, and versioned datasets.
- Treat prompts as code: version them, test them, and document intended behavior.
- Fail closed when context is missing. The assistant should say it does not know instead of inventing unsupported details.

