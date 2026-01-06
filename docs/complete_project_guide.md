# Complete Technical Guide: Development of Advanced AI Applications using RAG and LLM APIs

## 1. Purpose of this document

This handbook explains the complete summer-internship project, **Development of Advanced AI Applications using RAG and LLM APIs**, from introductory concepts to the implemented software design. It is written for a master's student, supervisor, examiner, or developer who needs to understand what the application does, why each component exists, how the components interact, how results were measured, and what work remains before a production deployment.

The project is a local knowledge assistant. A user loads a controlled document collection, asks a question, receives an answer grounded in retrieved passages, and can inspect the sources and citations behind that answer. The system supports a deterministic offline mode for reproducibility and optional OpenAI and Ollama providers for API-backed generation. It also includes a repeatable evaluation corpus, retrieval comparisons, unit tests, and browser-level testing.

This is a working educational prototype. It is deliberately transparent and modular rather than a claim that the system is ready for sensitive, high-scale, or regulated use.

## 2. Executive summary

Large language models can write fluent answers, but they do not automatically know the contents of an organisation's latest documents. They can also produce plausible statements that are unsupported by the available evidence. Retrieval-Augmented Generation (RAG) addresses this problem by retrieving relevant passages from a document collection at question time and providing them as context to a generation model.

This project implements a full RAG pipeline using Python, FastAPI, SQLite, a deterministic hashing-based embedding baseline, vector similarity, BM25-style keyword ranking, Reciprocal Rank Fusion (RRF), an evidence gate, citation validation, and a browser interface. The implementation was evaluated on 41 labeled questions over 13 synthetic university AI-research internship documents. The questions include ordinary lookups, ambiguous requests, conflicting policy documents, and genuinely unanswerable requests.

The recorded offline baseline achieved an average heuristic required-fact recall of **0.8241**, a citation-or-refusal coverage rate of **1.0000**, an answerable non-refusal rate of **1.0000**, and unanswerable-refusal accuracy of **0.8000**. In the retrieval comparison, hybrid retrieval produced the highest combined benchmark score, **0.9126**. These are development measurements on the included synthetic corpus. They are useful for engineering iteration, but they are not a general claim of model quality or real-world reliability.

## 3. Project goals and non-goals

### 3.1 Goals

- Build an end-to-end RAG application that can be run locally without an API key.
- Ingest text-based documents and supported PDFs, normalize the content, split it into traceable chunks, and store it persistently.
- Retrieve evidence using both semantic-like vector similarity and lexical keyword ranking.
- Produce answers with source citations and expose retrieved passages to the user.
- Refuse clearly when the retrieved evidence is too weak rather than inventing an answer.
- Allow the generation provider to be switched between offline, OpenAI Responses API, and Ollama modes through configuration.
- Evaluate retrieval and answer behavior with a versioned JSON Lines dataset.
- Demonstrate software-engineering practice through tests, diagnostics, configuration, documentation, and browser verification.

### 3.2 Non-goals

- This is not a production identity, access-control, or multi-tenant platform.
- The deterministic hashing embedder is not intended to compete with a learned embedding model.
- The current system does not prove that every citation entails every answer claim.
- The project does not train or fine-tune an LLM.
- The included corpus is synthetic and not a substitute for a held-out, domain-approved evaluation dataset.

## 4. Foundational concepts

### 4.1 Artificial intelligence, machine learning, and NLP

**Artificial intelligence (AI)** is the broad field of building systems that perform tasks associated with reasoning, perception, language, planning, or decision support. **Machine learning (ML)** is a branch of AI in which systems learn patterns from data rather than being programmed with a separate explicit rule for every situation. **Natural language processing (NLP)** focuses on enabling software to analyze, understand, search, or generate human language.

This project uses NLP techniques for tokenization, text normalization, chunking, keyword search, embedding generation, sentence selection, and text generation. The offline baseline is mostly deterministic NLP and information-retrieval logic; the optional providers introduce an external generative LLM.

### 4.2 Large language models

A **large language model (LLM)** is a neural network trained on extensive text collections to predict and generate sequences of tokens. A token is a unit of text processed by a model; depending on the model, it may be a word, subword, character sequence, or punctuation fragment. LLMs can summarize, answer questions, extract information, write code, and follow instructions, but their internal knowledge can be outdated, incomplete, or unsuitable for a private corpus.

An LLM is not inherently a database or a search engine. Its training data is not a reliable source of current organisational facts, and a model can generate a convincing answer even when its context contains no supporting evidence. That risk motivates RAG.

### 4.3 Retrieval-Augmented Generation

**Retrieval-Augmented Generation (RAG)** is an architecture that augments a generative model with retrieved external information. At query time, the system searches a knowledge base, selects the most relevant passages, and gives those passages to the generator as context. A well-designed RAG system should make it possible to inspect where an answer came from and should avoid answering beyond the available evidence.

The high-level RAG loop is:

- Ingest source documents and divide them into chunks.
- Convert each chunk into a searchable representation and store metadata.
- Convert the user question into a compatible query representation.
- Rank chunks by relevance and select the top results.
- Check whether the retrieved evidence is strong enough to support an answer.
- Ask a generation provider to answer only from the retrieved context.
- Validate citations, return sources, and display diagnostics.

### 4.4 Why RAG instead of model training?

Fine-tuning changes model parameters using training examples. RAG keeps the model unchanged and retrieves current external content at inference time. For a document-assistant use case, RAG is usually more practical because documents can be updated without retraining, sources can be displayed, access can later be controlled at retrieval time, and retrieval quality can be measured separately from generation quality. Fine-tuning and RAG can be combined in more advanced systems, but they solve different problems.

## 5. Problem statement and user workflow

The target use case is a grounded assistant over a controlled knowledge base. In the demonstration domain, the corpus contains documents about a university AI research internship, such as eligibility, stipend, working schedule, ethics, computing access, publication policy, and archived policy. A user can ask questions such as "What evidence is required for travel reimbursement?" or "What should the assistant do when evidence is missing?"

The desired behavior is not merely to return fluent language. The system should identify relevant evidence, explain the answer using that evidence, show citations, and say that an answer is unavailable when the indexed documents do not support one. This makes the system useful for document exploration while exposing the limits of its knowledge.

The browser workflow is:

- Start the local server and open the Knowledge Assistant interface.
- Upload a document or load the included sample corpus.
- Ask a grounded question.
- Read the answer and click or inspect citation chips and source excerpts.
- Review latency, provider, document count, chunk count, and refusal state.
- Run the sample evaluation to obtain repeatable development metrics.

## 6. System architecture

### 6.1 Architectural view

The application uses a modular pipeline. Each module has a limited responsibility so that it can be tested, replaced, or improved without rewriting unrelated parts of the system.

| Layer | Main responsibility | Implemented module |
| --- | --- | --- |
| Configuration | Load environment variables and defaults | `backend/app/settings.py` |
| Ingestion | Extract, normalize, identify, and chunk documents | `backend/app/text_processing.py` and `backend/app/rag.py` |
| Embeddings | Transform text into deterministic numeric vectors | `backend/app/embeddings.py` |
| Storage | Persist documents, chunks, embeddings, and metadata | `backend/app/store.py` |
| Retrieval | Rank chunks using vector, keyword, and fusion signals | `backend/app/retrieval.py` |
| Generation | Produce grounded answers using offline, OpenAI, or Ollama mode | `backend/app/llm.py` |
| Orchestration | Connect ingestion, retrieval, refusal, citations, and diagnostics | `backend/app/rag.py` |
| API | Expose HTTP endpoints and static web content | `backend/app/api.py` |
| Frontend | Provide document controls, chat, source display, and evaluation UI | `frontend/` |
| Evaluation | Run labeled examples and retrieval comparisons | `backend/app/evaluation.py` and `backend/app/comparison.py` |

### 6.2 Data flow for ingestion

When a document is uploaded or ingested from a permitted local path, the RAG service first extracts text. Plain-text formats such as `.txt`, `.md`, `.csv`, `.json`, `.html`, and `.py` are decoded as UTF-8 with replacement for invalid characters. PDF extraction is supported when the optional `pypdf` dependency is installed. The extracted content is normalized by removing null characters, normalizing line endings, compressing repeated spaces, and limiting excessive blank lines.

The system constructs a stable document identifier from the filename and content, creates a readable title, divides the text into chunks, embeds each chunk, and writes the document, chunks, metadata, and vectors to SQLite. A chunk records its document ID, source filename, title, zero-based position, text, metadata, and embedding. That provenance is later used to create citations.

### 6.3 Data flow for a question

When a user asks a question, the service validates that the question is non-empty, loads stored chunks, ranks them with the hybrid retriever, applies the evidence gate, and either refuses or calls the configured generation provider. Citation numbers in the returned answer are checked against the number of retrieval results. Valid citation numbers are expanded into source objects containing title, source path, position, relevance scores, and an excerpt. The API returns both the answer and diagnostics such as latency, provider, number of indexed chunks, number of retrieved chunks, top-k setting, and refusal state.

## 7. Document ingestion and chunking

### 7.1 Why documents are chunked

An entire long document is usually too large and too unfocused to send to a retrieval or generation model for every question. **Chunking** divides a document into smaller passages that preserve enough local meaning to answer questions. Instead of searching a 50-page handbook as one item, the retriever can rank a paragraph-sized or section-sized chunk about the relevant policy.

Chunk size is a trade-off. Chunks that are too small may omit context needed to interpret a fact. Chunks that are too large may mix unrelated topics, reduce ranking precision, and consume more generation context. This implementation uses a default chunk size of 900 characters and a default overlap of 160 characters.

### 7.2 Chunking approach used in this project

The chunker first splits normalized text into paragraphs. It accumulates adjacent paragraphs until adding another paragraph would exceed the configured chunk size. When it begins a new chunk, it carries a tail overlap from the previous chunk so that context at boundaries is less likely to be lost. Paragraphs longer than the configured size are split on word boundaries with overlap.

The procedure is intentionally understandable and deterministic. It does not yet recognize document headings, tables, page layout, semantic sections, or scanned-PDF OCR. Those are important future improvements for complex source material.

### 7.3 Stable IDs and provenance

A stable identifier is produced by hashing the filename and normalized text with SHA-256 and retaining a short prefix. Each chunk uses an ID in the form `document-id#chunk-0000`. Provenance means that every searchable item can be traced back to its original document and position. Provenance is essential for citations, debugging, evaluation labels, data deletion, and future access control.

## 8. Embeddings and vector similarity

### 8.1 What is an embedding?

An **embedding** is a numeric vector intended to represent aspects of text meaning or content. If two pieces of text are semantically related, a useful embedding model places their vectors near one another in a high-dimensional vector space. Embeddings allow retrieval to match a query such as "payment for interns" to a passage using the word "stipend," even if the exact words differ. The strength of this behavior depends heavily on the embedding model.

### 8.2 Deterministic hashing embedding baseline

To keep the project runnable offline and reproducible, the application uses `HashingEmbedder`, a 384-dimensional hashing baseline. Tokens are counted, mapped deterministically into vector positions using a BLAKE2b hash, signed to reduce bias, weighted by `1 + log(count)`, and L2-normalized. The result is stable across runs without model downloads or API calls.

This is not a learned semantic embedding model. Hash collisions can occur, synonyms may not align, and the representation remains strongly lexical. It exists so the whole pipeline can be demonstrated and tested without external services. A practical next step is to make the embedding interface configurable and compare a learned embedding model against this baseline on a held-out dataset.

### 8.3 Cosine similarity

The implementation compares normalized vectors with **cosine similarity**. For vectors `a` and `b`, cosine similarity is:

```text
cosine(a, b) = sum(a_i * b_i) / (||a|| * ||b||)
```

Because vectors are normalized in this project, the calculation reduces to their dot product. A higher score indicates that vectors point in a more similar direction. The code clamps negative vector scores to zero for the retrieval signal. Cosine similarity is useful, but it is only one relevance signal and should not be confused with human judgement of answer correctness.

## 9. Keyword retrieval, BM25, and hybrid ranking

### 9.1 Tokenization and lexical matching

**Tokenization** splits text into units used by later processing. The project lowercases alphanumeric tokens and removes a compact list of common stopwords by default. **Lexical retrieval** searches for overlapping words rather than inferred meaning. It is especially helpful for policy names, acronyms, dates, IDs, proper nouns, and exact technical terms.

The implementation computes lexical overlap as the fraction of distinct query terms that also occur in a chunk. This score contributes to hybrid ranking and to the evidence gate.

### 9.2 BM25-style ranking

**BM25** is a classic information-retrieval ranking function. It rewards documents containing query terms, gives rarer terms more importance through inverse document frequency (IDF), and normalizes for document length. In simplified form, each matched term contributes an IDF-weighted term-frequency score. The implementation uses `k1 = 1.5` and `b = 0.75`, common BM25-style parameters.

Keyword ranking is often powerful when the user uses the same vocabulary as the source document. It may perform poorly when the question uses synonyms or a paraphrase. That is why it is combined with vector-based ranking.

### 9.3 Reciprocal Rank Fusion

**Reciprocal Rank Fusion (RRF)** combines rankings from different retrieval methods. Rather than comparing raw scores that may have different scales, it gives a contribution based on rank:

```text
RRF score = 1 / (k + vector_rank) + 1 / (k + keyword_rank)
```

In this project, `k = 60`. A passage ranked highly by both vector and keyword retrieval receives a stronger fused score. The final hybrid score adds `0.15 * lexical_overlap` to the RRF score. The approach is simple, explainable, and suitable for a baseline, though fusion weights should be tuned on validation data rather than chosen only by intuition.

### 9.4 Retrieval result diagnostics

Each `RetrievalResult` retains the fused score, vector score, keyword score, lexical overlap, rank, and source chunk. Keeping these signals is useful because retrieval errors can otherwise be opaque. A developer can examine whether a failure arose from the embedding baseline, lexical mismatch, rank fusion, chunking, or a weak evidence threshold.

## 10. Evidence gating, refusal, and citations

### 10.1 Why an evidence gate is needed

Retrieval systems often return the top results even when every result is weakly related to the question. Without a gate, the generator may treat the best available but irrelevant chunk as evidence and produce an unsupported answer. An **evidence gate** is a policy that checks retrieval quality before generation and returns a refusal when the evidence is insufficient.

The project refuses when there are no retrieval results or when the best result simultaneously has low hybrid score, low vector similarity, limited lexical overlap, and weak keyword evidence. The current defaults are `MIN_HYBRID_SCORE=0.10`, vector score below `0.20`, lexical overlap at most `0.50`, and `MAX_REFUSAL_KEYWORD_SCORE=6.0`. Combining signals reduces the chance that one unusual score causes an answer to be rejected.

These values were calibrated against the included development corpus. They are not universal thresholds and must be retuned on a held-out dataset. An evidence gate is a useful safeguard, but it is not a mathematical proof that an accepted answer is correct.

### 10.2 Refusal behavior

When the gate rejects retrieval, the system returns: "The answer is not available from the indexed documents." It still includes the retrieved source diagnostics when available, so a user or developer can inspect what was considered. Refusal is important because a high-quality assistant needs to distinguish "I have evidence" from "I do not have evidence." The evaluation therefore measures both correct refusal on unanswerable cases and non-refusal on answerable cases.

### 10.3 Grounded prompt construction

For accepted retrieval, the system constructs a context block for each result. Each block includes a numerical citation label, title, source path, chunk position, and chunk text. The generation instructions say to answer only from supplied context, cite sources as `[1]`, `[2]`, and so on, refuse when context is insufficient, and remain concise and factual.

The prompt pattern improves traceability because the model receives a finite set of evidence blocks. It does not make a model incapable of hallucinating. A future system should add claim-level entailment checking, structured answer schemas, and citation-to-claim validation.

### 10.4 Citation validation

The project removes citation numbers outside the valid retrieved-result range. It then maps valid citation markers to detailed citation records. If an answer is accepted but lacks a citation marker, the orchestration layer appends `[1]` when results exist. This guarantees structural citation presence, not semantic support. Therefore, the metric named citation-or-refusal coverage should not be interpreted as citation accuracy.

## 11. Generation providers and LLM APIs

### 11.1 Offline extractive provider

The default provider is `offline-extractive`. It does not call an external LLM. For each retrieved chunk, it chooses the sentence with the greatest overlap with query terms, selects up to three distinct candidate sentences, and adds inline citations. This baseline is deterministic, private to the local machine, inexpensive, and convenient for automated tests. Its limitation is that it does not perform rich synthesis, follow-up reasoning, or semantic paraphrase at the level of a modern LLM.

### 11.2 OpenAI Responses API provider

When `LLM_PROVIDER=openai`, the application uses the OpenAI Python SDK and the Responses API. It passes the system instructions separately from the user context, sends a configurable model name, sets a maximum output-token budget, and optionally uses GPT-5 reasoning effort. The included defaults are `OPENAI_MODEL=gpt-5.6-luna`, `OPENAI_REASONING_EFFORT=low`, and `OPENAI_MAX_OUTPUT_TOKENS=700`.

An API key must be supplied through `OPENAI_API_KEY`; it is not stored in frontend code or source control. A live OpenAI request was not run in this project environment because no credential was provided. The adapter has unit coverage using a mocked client, but real-provider behavior, latency, cost, rate limits, and failure handling remain separate work.

### 11.3 Ollama provider

Ollama is a local model-serving runtime. With `LLM_PROVIDER=ollama`, the application sends a non-streaming request to `OLLAMA_BASE_URL`, defaulting to `http://localhost:11434`, and uses `OLLAMA_MODEL`, defaulting to `llama3.1:8b`. This option can keep model execution local while still enabling generative answers. Local execution does not remove the need for host security, model governance, resource planning, or evaluation.

### 11.4 Provider abstraction

All providers expose the same generation responsibility: accept a question, retrieved results, and settings; return answer text. This abstraction allows the RAG orchestration code to remain stable while a provider is replaced. Provider abstraction is a major engineering design decision because it separates retrieval reliability from vendor-specific API details.

## 12. Persistence, API, and user interface

### 12.1 SQLite storage

**SQLite** is an embedded relational database stored in a local file. It requires no separate database server and is appropriate for a single-user demonstration. The system stores indexed state in `.rag_data/rag_index.sqlite3` by default. Documents and chunks can be listed, replaced, or cleared through the service.

The present implementation stores vectors as JSON and scans all chunks at query time. That design is transparent and adequate for a 29-chunk demonstration corpus, but it will not scale efficiently to very large collections or high concurrency. A production implementation would likely use a vector database or approximate-nearest-neighbour index, asynchronous ingestion, and stronger operational controls.

### 12.2 FastAPI backend

**FastAPI** is a Python web framework for building HTTP APIs with type-aware request and response models. The API exposes health status, document listing, document clearing, upload, local path ingestion, question answering, and evaluation. Errors such as empty questions or unsupported file types are converted into HTTP 400 responses.

The API response for a question includes the answer, cited sources, retrieved sources, and diagnostics. This contract is useful because a frontend can show user-friendly answers while a developer can still see the data required for debugging and evaluation.

### 12.3 Frontend behavior

The frontend uses browser-native HTML, CSS, and JavaScript. It provides a compact operational interface rather than a marketing landing page. Users can upload files, load the sample corpus, clear the index, ask questions, inspect source items, view latency and health status, and run the sample evaluation. The application serves frontend assets from the same FastAPI process, which reduces setup complexity for an internship prototype.

## 13. Configuration, installation, and operation

### 13.1 Prerequisites

- Python 3.9 or later.
- A virtual environment.
- Project dependencies from `requirements-dev.txt`.
- Chromium for the Playwright browser smoke test.
- Optional: `pypdf` for PDF extraction.
- Optional: an OpenAI API key or a running Ollama service for remote/local generative providers.

### 13.2 Installation commands

```bash
cd /Users/architjain/projects/02-advanced-rag-llm-app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
make browser-install
cp .env.example .env
```

Start the development server with:

```bash
make dev
```

Then open `http://127.0.0.1:8000` in a browser.

### 13.3 Important environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `RAG_DATA_DIR` | `.rag_data` | Directory that holds the SQLite index |
| `CHUNK_SIZE` | `900` | Maximum target chunk size in characters |
| `CHUNK_OVERLAP` | `160` | Context retained between adjacent chunks |
| `EMBEDDING_DIM` | `384` | Dimension of the hashing vector baseline |
| `TOP_K` | `5` | Number of highest-ranked chunks used for an answer |
| `MIN_HYBRID_SCORE` | `0.10` | One threshold used by the evidence gate |
| `MAX_REFUSAL_KEYWORD_SCORE` | `6.0` | Keyword condition used by the evidence gate |
| `LLM_PROVIDER` | `offline` | `offline`, `openai`, or `ollama` |
| `OPENAI_API_KEY` | empty | Secret used only for OpenAI mode |
| `OPENAI_MODEL` | `gpt-5.6-luna` | OpenAI model identifier |
| `OPENAI_REASONING_EFFORT` | `low` | Reasoning setting for supported GPT-5 models |
| `OPENAI_MAX_OUTPUT_TOKENS` | `700` | Maximum OpenAI generation output |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama service URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Model selected through Ollama |

### 13.4 Reproducible commands

```bash
make test
make eval
make compare
make test-ui
make report
```

`make test` runs unit tests. `make eval` runs the answer-oriented JSON Lines evaluation. `make compare` produces vector-only, keyword-only, and hybrid retrieval comparisons. `make test-ui` runs the Playwright browser workflow against a running local server. `make report` generates the original final report document.

## 14. Evaluation methodology

### 14.1 Why evaluation must be split

RAG quality is not one number. A system can retrieve correct evidence but generate a poor answer, or generate fluent text even when retrieval missed the required fact. Evaluation should therefore distinguish retrieval quality, generation quality, citation support, refusal quality, latency, and cost.

The project uses a JSON Lines evaluation file. Each line is an independent JSON object containing an identifier, question, expected answer, expected source documents, required fact phrases where applicable, category, and difficulty. JSON Lines is useful because cases can be versioned, reviewed, extended, and processed one at a time.

### 14.2 Dataset composition

The included dataset has 41 cases over 13 synthetic internship documents:

- 30 answerable cases.
- 3 ambiguous cases.
- 3 conflicting-policy cases.
- 5 unanswerable cases.

The corpus includes an archived 2025 FAQ and more current policy documents so that the project can exercise source conflict and recency issues. It is still a development corpus. A valid final research claim needs separate validation and test splits that are not reused for tuning chunking, fusion, or refusal thresholds.

### 14.3 Answer-oriented metrics

**Required-fact recall** measures how many expected fact phrases appear in an answer. The current implementation uses a lightweight lexical proxy: it lowercases text and marks a fact as matched when at least half of its terms longer than three characters appear in the answer. This is easy to run but does not understand semantic equivalence, negation, contradiction, or claim boundaries.

**Citation-or-refusal coverage** is true when a response has at least one returned citation or the service refuses. It checks structural behavior, not whether a cited source supports a particular statement.

**Unanswerable-refusal accuracy** is the proportion of truly unanswerable cases correctly refused. **Answerable non-refusal rate** is the proportion of answerable cases that were not incorrectly refused. Together, these metrics help reveal the trade-off between unsafe answers and overly conservative refusals.

### 14.4 Retrieval-oriented metrics

**Recall@5** asks whether an expected source is present among the top five retrieved source documents. **Mean Reciprocal Rank (MRR)** rewards the system when the first relevant result appears early in the ranking. **Normalized Discounted Cumulative Gain at 5 (nDCG@5)** rewards relevant documents near the top while discounting lower positions. The comparison uses source-level rather than chunk-level relevance because the dataset labels expected source documents.

The benchmark score is a project-defined combined measure of retrieval and abstention behavior. It is useful for selecting a baseline on the included corpus, but it should not be reported as a universal quality standard because its weighting choices are local to this experiment.

### 14.5 Recorded results

| Metric | Recorded offline result | Meaning |
| --- | ---: | --- |
| Evaluation cases | 41 | Full included development set |
| Fact-scored cases | 36 | Unanswerable cases excluded from fact recall |
| Average heuristic required-fact recall | 0.8241 | Lexical fact-coverage proxy |
| Citation-or-refusal coverage | 1.0000 | Every response had a citation or refusal |
| Answerable non-refusal rate | 1.0000 | No answerable case rejected in the recorded run |
| Unanswerable-refusal accuracy | 0.8000 | Four of five unsupported requests refused |
| Average local pipeline latency | about 4.6 ms | Small offline corpus only |

| Retrieval configuration | Recall@5 | MRR | nDCG@5 | Unanswerable abstention | Combined score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector-only | 0.9583 | 0.9028 | 0.9033 | 0.6000 | 0.8663 |
| Keyword-only | 1.0000 | 0.9306 | 0.9441 | 0.0000 | 0.8290 |
| Hybrid | 1.0000 | 0.9097 | 0.9283 | 0.8000 | **0.9126** |

The keyword-only configuration ranked expected sources well on this vocabulary-aligned corpus, but hybrid retrieval won the combined score because the evidence gate refused four unsupported questions. This is an engineering observation, not proof that hybrid retrieval is semantically superior in all domains.

## 15. Testing and quality assurance

### 15.1 Unit tests

The Pytest suite covers core text processing, ingestion-to-answer behavior, evaluation logic, comparison behavior, low-evidence refusal, and the mocked OpenAI provider. The final recorded suite contains nine passing tests. Unit tests are valuable because they protect known contracts, such as correct chunking, deterministic source handling, and API-provider request shape.

### 15.2 Browser smoke test

The Playwright smoke test opens the local application in Chromium and verifies health status, clearing the index, file upload, sample-corpus ingestion, question answering, visible citations, visible sources, latency display, evaluation output, JavaScript console errors, page errors, and mobile horizontal overflow. It captures screenshots in `artifacts/browser/` for visual inspection.

This test verifies that major components work together. It does not establish model quality, protect against all security issues, or replace usability research with real users.

### 15.3 Reproducibility practices

The project supports reproducibility by using deterministic local embeddings and offline extraction, storing evaluation data as versioned JSON Lines, preserving configuration in `.env.example`, recording benchmark JSON outputs, and keeping provider code behind explicit configuration. A future experimental report should additionally record a corpus version, Git revision, Python dependency lockfile, provider model version, prompt version, latency distribution, random seed where relevant, and cost.

## 16. Security, privacy, and ethics

### 16.1 Current security boundaries

The project is designed for trusted local use. It does not include authentication, authorization, document-level access control, tenant isolation, quotas, audit logging, strict CORS, rate limiting, malware scanning, or secure production deployment. The path-ingestion endpoint is especially unsuitable for an exposed service unless it is restricted to an allow-listed storage area.

### 16.2 Sensitive data and external providers

Before sending documents or questions to an external API, an operator must determine whether that information is permitted to leave the environment. Consider data classification, retention terms, regional processing requirements, contractual controls, user consent, and logs. API keys belong in environment variables or a secret manager, never in the repository, browser code, prompts, screenshots, or user documents.

### 16.3 Prompt injection

**Prompt injection** occurs when untrusted content attempts to manipulate model instructions, for example by placing text in an indexed document that says to ignore system rules or reveal private data. RAG can bring such text directly into generation context. The current system separates instructions from context but does not detect malicious content or enforce policy-aware retrieval. Future controls include content filtering, provenance labels, role-aware retrieval, prompt-injection test cases, output validation, and least-privilege access.

### 16.4 Ethical use

Sources and refusals can improve transparency, but they do not eliminate bias, outdated content, incorrect extraction, or over-trust. Users should treat the application as decision support, not as an autonomous authority in consequential settings. A responsible system should permit corrections, deletion requests, human review, and clear disclosure of uncertainty.

## 17. Limitations and future work

### 17.1 Current limitations

- The corpus is small, synthetic, and used during development; it is not a held-out real-world benchmark.
- The hashing embedder is deterministic but weak for semantic matching.
- All chunks are scanned in memory per query, which does not scale well.
- Citation markers are range-validated but not checked for claim-level entailment.
- The fact-recall metric is lexical and can misjudge paraphrases, negations, or partial statements.
- Remote provider runs, real API costs, rate-limit behavior, and model-specific failure cases are not included in the recorded results.
- The system has no conversation memory, reranker, query rewriting, metadata access filters, OCR, structured table extraction, or conflict-resolution policy.
- Production security and observability are intentionally absent.

### 17.2 Recommended next steps

**Independent data split:** Create validation and test datasets with human-reviewed source relevance and answer-quality labels.

**Retrieval experiments:** Compare learned embeddings, hybrid fusion weights, chunk size, overlap, and a cross-encoder reranker using only validation data for tuning.

**Human evaluation:** Add claim-level groundedness and citation-correctness evaluation with at least two human reviewers and reported agreement.

**Provider measurement:** Run OpenAI and Ollama experiments with recorded model identity, prompt version, token usage, cost, latency percentiles, and errors.

**Document intelligence:** Add OCR, structured document parsing, metadata filters, incremental ingestion, and approximate-nearest-neighbour indexing.

**Deployment controls:** Introduce authentication, role-based access control, secure file handling, allowed origins, audit logs, deletion workflows, rate limiting, and monitoring before any sensitive deployment.

**Adversarial testing:** Add prompt-injection and data-exfiltration test cases to the evaluation suite.

## 18. Six-week internship execution plan

| Week | Focus | Concrete outputs |
| --- | --- | --- |
| 1 | Problem formulation and architecture | Scope, user stories, module boundaries, repository, configuration plan |
| 2 | Ingestion and persistence | Extraction, normalization, chunking, stable IDs, SQLite storage, tests |
| 3 | Retrieval | Hashing embeddings, cosine similarity, BM25-style ranking, RRF, diagnostics |
| 4 | Generation and interface | Grounded prompt, offline/OpenAI/Ollama adapters, citations, FastAPI, UI |
| 5 | Evaluation and verification | JSONL corpus, metrics, retrieval ablation, unit tests, browser smoke test |
| 6 | Analysis and delivery | Results interpretation, limitations, ethics, report, presentation, demonstration |

The sequence demonstrates a defensible internship progression: build a deterministic baseline first, instrument it, evaluate it, then make external generation optional. This order prevents an API model from hiding weaknesses in ingestion or retrieval.

## 19. Glossary of important terms

**API (Application Programming Interface):** A defined interface through which one software component requests services or data from another. In this project, FastAPI exposes HTTP endpoints and the OpenAI/Ollama adapters call provider APIs.

**Approximate Nearest Neighbour (ANN):** A technique for finding close vectors efficiently at scale, trading a small amount of exactness for speed. It is not used in the current small-corpus baseline.

**BM25:** A widely used keyword-ranking formula that combines term frequency, inverse document frequency, and document-length normalization.

**Chunk:** A bounded passage created from a larger document so it can be indexed, retrieved, cited, and placed into a generation context.

**Chunk overlap:** Text repeated between adjacent chunks to reduce loss of meaning at chunk boundaries.

**Citation:** A reference connecting an answer statement to a retrieved source. In this project a citation includes source, chunk position, scores, and an excerpt.

**Citation accuracy:** The proportion of citations that actually support the specific claim they are attached to. This is stronger than merely showing a citation marker.

**Context window:** The maximum amount of input and output content an LLM can process in one request. RAG manages context by selecting a small number of relevant chunks.

**Cosine similarity:** A vector similarity measure based on the angle between vectors. Higher values normally indicate greater directional similarity.

**CORS (Cross-Origin Resource Sharing):** Browser rules that control which web origins may call an API. Production deployments should configure this narrowly.

**Embedding:** A numeric vector representation of text used for similarity search, classification, clustering, or recommendation.

**Embedding dimension:** The number of numeric values in an embedding vector. The offline baseline uses 384 dimensions.

**Evidence gate:** A pre-generation policy that refuses to answer when retrieval signals are too weak.

**FastAPI:** A Python framework for creating HTTP APIs with validation and asynchronous support.

**Groundedness:** The degree to which answer claims are supported by retrieved source content.

**Hallucination:** A generated statement that is incorrect, unsupported, or invented relative to the available evidence.

**Hybrid retrieval:** A retrieval strategy that combines semantic/vector and lexical/keyword signals.

**IDF (Inverse Document Frequency):** A weighting term that increases the influence of words that occur in fewer documents.

**Inference:** Running a trained model to obtain an output, as distinct from training or fine-tuning the model.

**JSON Lines (JSONL):** A file format where each line is a complete JSON object. It is convenient for evaluation examples and streaming records.

**Keyword retrieval:** Search based on literal term matches, often using methods such as BM25.

**Latency:** Time required for an operation, such as retrieval, generation, or an end-to-end answer.

**LLM (Large Language Model):** A model trained to predict and generate language tokens at scale.

**Metadata:** Data describing another item. Chunk metadata can include document title, source filename, position, character count, owner, date, or access policy.

**MRR (Mean Reciprocal Rank):** A ranking metric based on the reciprocal of the first relevant result's rank, averaged across queries.

**nDCG@k:** A ranking metric that rewards relevant results near the top of a ranked list and normalizes the score against an ideal ordering.

**OCR (Optical Character Recognition):** Technology that converts text contained in scanned images into machine-readable characters.

**Ollama:** A local model-serving runtime used here as an optional generation provider.

**Prompt:** The instructions and context submitted to an LLM for a generation request.

**Prompt injection:** An attack in which untrusted text tries to override system instructions or cause unsafe behavior.

**Provider abstraction:** A software interface that lets the application switch among generation providers without rewriting orchestration logic.

**RAG (Retrieval-Augmented Generation):** A pattern that retrieves relevant external content and uses it as context for generation.

**Recall@k:** The proportion of queries for which a relevant result appears within the first `k` returned results.

**Reciprocal Rank Fusion (RRF):** A method for combining several rankings using reciprocal rank contributions.

**Refusal:** A transparent response that says the answer cannot be supported from available evidence.

**Reranker:** A second-stage model that receives an initial candidate list and produces a more accurate order, often at greater computational cost.

**Retrieval:** The process of finding and ranking source passages that may answer a query.

**SQLite:** A serverless relational database stored in a local file.

**Stopword:** A very common language word that may be removed during keyword processing because it adds little discriminative value.

**Token:** A unit of text used by a tokenizer or language model.

**Tokenization:** The process of breaking text into tokens for analysis, retrieval, or model input.

**Top-k:** The number of highest-ranked items kept after retrieval. This application defaults to five chunks.

**Traceability:** The ability to connect an output back to the data, processing steps, configuration, and sources that produced it.

**Uvicorn:** An ASGI server used to run the FastAPI application during development.

**Vector database:** A storage system optimized for embedding vectors and similarity search. The current project uses SQLite instead because the corpus is small.

**Vector search:** Retrieval that compares a query embedding to stored chunk embeddings, usually through cosine similarity or a related distance function.

<!-- pagebreak -->

## 20. Conclusion

This project demonstrates that an advanced AI application is more than an LLM API call. A credible RAG assistant requires controlled data ingestion, traceable chunking, measurable retrieval, an explicit evidence boundary, generation-provider abstraction, citations, usable interfaces, repeatable tests, and careful interpretation of metrics. The system implemented here provides all of those components in a compact, locally runnable form.

The key result is not simply that the application returns answers. It is that the application makes evidence, uncertainty, retrieval trade-offs, and limitations visible. Next steps are held-out evaluation, stronger embeddings, claim-level validation, provider measurement, and security controls.
