# Development of Advanced AI Applications using RAG and LLM APIs

## Abstract

Large language models (LLMs) can produce fluent answers, but a standalone model does not guarantee that an answer is based on an organisation's current documents, that its factual claims are verifiable, or that it will decline to answer when evidence is absent. This project investigates Retrieval-Augmented Generation (RAG) as an engineering approach to those limitations. A complete, locally runnable knowledge assistant was designed and implemented around a controlled document corpus. The system extracts and normalises text, creates overlapping chunks, generates deterministic local embeddings, persists document and vector data in SQLite, performs hybrid vector and BM25-style retrieval, and returns source-aware answers through a FastAPI service and responsive web interface. The generation layer supports an offline extractive baseline and configurable OpenAI and Ollama adapters, allowing retrieval behaviour to be studied without requiring credentials while retaining a path to API-backed generation.

The project also includes an evaluation harness, automated core tests, a 41-case university internship corpus, retrieval ablations, and a browser-level workflow. In the recorded offline evaluation, the system achieved average heuristic fact recall of **0.8241**, citation-or-refusal coverage of **1.0000**, answerable non-refusal of **1.0000**, and unanswerable-refusal accuracy of **0.8000**. A deterministic retrieval comparison found hybrid retrieval to have the strongest combined score (**0.9126**) across 36 evidence-seeking and five unanswerable cases. These are development measurements, not evidence of production readiness: fact recall is approximated by term matching, the refusal threshold was calibrated on the same corpus, citation coverage measures presence rather than evidential correctness, and no paid or remote LLM was involved in the measured run. The principal contribution is therefore not a claim of state-of-the-art accuracy, but a transparent and extensible RAG application in which ingestion, retrieval, generation, citation handling, refusal, and evaluation are separated and inspectable.

## 1. Introduction and Problem Definition

LLMs encode broad linguistic and factual patterns in model parameters. This makes them useful general-purpose interfaces, but it creates practical difficulties for knowledge-intensive applications. Parametric knowledge can be incomplete or outdated; a model may produce plausible unsupported text; private organisational data may not have been available during training; and users may be unable to trace an answer to its source. Fine-tuning can alter behaviour, but it is not an efficient mechanism for continuously updating documentary knowledge and does not by itself provide citations.

RAG addresses this problem by retrieving passages from an external corpus at query time and placing them in the generation context. The model is instructed to answer from those passages and cite them. This changes the application from a single probabilistic model call into a pipeline whose reliability depends on document extraction, chunk boundaries, representations, retrieval ranking, prompt construction, provider behaviour, and answer validation. A failure at any stage can affect the final answer. For example, a correct generator cannot use evidence that retrieval failed to return, while a high-ranking chunk can still be misleading if it is semantically similar but factually irrelevant.

The project treats RAG as an information system and examines four questions:

1. How can heterogeneous documents be converted into stable, retrievable evidence units?
2. How can lexical and vector signals be combined in a lightweight local retriever?
3. How can generated answers be constrained, cited, and refused when evidence is weak?
4. How should quality claims be measured without confusing convenient proxy metrics with correctness?

## 2. Objectives and Scope

The primary objective was to build an end-to-end RAG application that answers questions over an indexed corpus and exposes enough evidence and diagnostics for a user or evaluator to inspect the result. This objective was decomposed into the following technical goals:

- implement document ingestion, normalisation, deterministic identifiers, chunking, and metadata retention;
- construct a reproducible embedding and indexing baseline that can run without external services;
- combine vector similarity and keyword relevance rather than relying on one retrieval signal;
- implement provider-independent generation with an offline baseline and optional LLM APIs;
- require inline citations, remove invalid citation numbers, and refuse queries with insufficient retrieval evidence;
- expose ingestion, corpus management, question answering, health, and evaluation through an HTTP API;
- provide a usable web interface for uploading documents, asking questions, viewing sources, and running evaluation;
- create repeatable automated evaluation and test workflows; and
- document limitations, security implications, and the experimental work needed before deployment.

The scope is a research and demonstration baseline. It excludes user accounts, document-level authorisation, OCR, managed vector infrastructure, learned reranking, and production deployment. OpenAI and Ollama generation adapters are implemented, but recorded quantitative results are for offline extractive mode; remote-provider comparisons require credentials and a separate cost-controlled experiment.

## 3. System Architecture

The architecture separates ingestion from online question answering while sharing a persistent SQLite index:

```text
Documents / Uploads
        |
        v
Extraction -> Normalisation -> Overlapping chunks -> Hash embeddings
        |                                                |
        +------------ metadata and vectors --------------+
                              |
                           SQLite
                              |
User question -> Query embedding -> Hybrid retrieval -> Evidence threshold
                                                       |          |
                                                  insufficient   sufficient
                                                       |          |
                                                    refusal   prompt/generation
                                                                  |
                                                   citation validation
                                                                  |
                                             answer + citations + diagnostics
                                                                  |
                                                    FastAPI + Web UI
```

### 3.1 Ingestion and preprocessing

The ingestion layer accepts plain text, Markdown, CSV, JSON, HTML, and Python source files. PDF extraction is supported with optional `pypdf`. Text is decoded as UTF-8 with replacement, nulls are removed, and whitespace is normalised. PDF pages receive text markers, though page-level metadata is not a dedicated field.

A document identifier is derived from a SHA-256 digest of the filename and extracted text, truncated to 16 hexadecimal characters. This makes repeated ingestion of unchanged content idempotent at the document level. Chunk identifiers combine the document identifier and a zero-padded position.

Default chunking targets 900 characters with 160 characters of overlap and prefers paragraph boundaries. Oversized paragraphs are split by words; otherwise paragraphs accumulate to the limit and a tail is carried forward. Metadata includes document ID, title, source, position, and character count. Character counts only approximate model tokens, and chunking is unaware of headings, tables, semantics, or provider context windows.

### 3.2 Embeddings and persistent storage

The baseline embedder is a deterministic 384-dimensional feature-hashing representation. It tokenises terms, removes fixed stop words, and hashes each token with BLAKE2b into a signed vector dimension. Log-scaled counts are accumulated and L2-normalised; cosine similarity is the resulting dot product.

This representation provides reproducibility, zero API cost, and offline operation. It should not be interpreted as a learned semantic embedding: it primarily preserves token identity and cannot reliably capture paraphrase, polysemy, or domain-specific semantic relationships. The class boundary is intentionally replaceable so a learned local or API embedding model can be evaluated later.

SQLite stores document and chunk rows, with embeddings and metadata serialised as JSON and queries parameterised. Re-ingestion replaces prior chunks. This removes an infrastructure dependency, but loading and scoring all chunks per query does not scale.

### 3.3 Hybrid retrieval

For each question, cosine similarity compares hashed vectors, a BM25-style score applies term-frequency saturation and length normalisation (`k1 = 1.5`, `b = 0.75`), and lexical overlap measures the proportion of query terms present in a chunk.

Each score list is independently ranked and Reciprocal Rank Fusion is applied as `1/(60 + vector_rank) + 1/(60 + keyword_rank)`, followed by `0.15 * overlap`. The top five are returned. Fusion avoids calibrating raw score scales, but constants are untuned and hashing remains lexically driven; this is hybrid scoring without the semantic depth of a learned dense retriever.

### 3.4 Evidence gating, generation, and citations

The orchestration service refuses an answer when no result exists or when the best hybrid result falls below a configurable score threshold (`0.10`) while also showing low vector similarity, limited lexical overlap, and a weak keyword score. A refusal returns the fixed statement that the answer is unavailable from indexed documents. The threshold was calibrated against the included development corpus and reached 80% accuracy on five unanswerable cases; it is not a calibrated probability of answerability and must be retuned on a held-out set.

When evidence passes the gate, the provider layer offers three modes. The default offline provider selects the sentence with greatest query-term overlap from each retrieved chunk and returns at most three distinct evidence sentences with inline citation numbers. The OpenAI adapter uses the Responses API with a configurable model, output budget, and GPT-5 reasoning effort; the default is `gpt-5.6-luna` with low effort. The Ollama adapter sends a non-streaming generation request to a configurable local endpoint. All modes share instructions to answer only from supplied context, cite sources as `[1]`, `[2]`, and so forth, and decline when evidence is insufficient.

Out-of-range citation numbers are removed, while valid numbers map to source, position, score, and excerpt. If an accepted answer has no citation, `[1]` is appended. This guarantees a marker, not evidential entailment, so citation correctness needs separate evaluation.

### 3.5 API and user interface

FastAPI exposes endpoints for health status, document listing and clearing, file upload, filesystem-path ingestion, question answering, and sample evaluation. Static HTML, CSS, and JavaScript provide a responsive interface with corpus controls, chat, latency information, cited-source inspection, and evaluation output. The interface is served by the same application, simplifying local deployment and avoiding a separate frontend build pipeline.

The API returns answer text, cited sources, all retrieved sources, and diagnostics including provider, indexed chunk count, retrieved count, configured top-k, refusal state, and elapsed pipeline time. This response structure supports both user presentation and engineering investigation.

## 4. Methodology

Development followed an incremental, baseline-first process. Preprocessing, retrieval, generation, and evaluation were separated so they could change independently. Deterministic local components removed dependence on credentials, rate limits, or changing model outputs; optional APIs were placed behind a common generation interface.

Traceability was preserved through chunk source and position, component retrieval scores, citation-range checks, and structured diagnostics. Verification operates at three levels: core tests cover chunking, ingestion-to-answer behaviour, and evaluation execution; the JSON Lines harness records answer proxies; and a Playwright workflow exercises health, index management, upload, ingestion, answering, citations, evaluation, console errors, and mobile overflow. These checks verify integration but do not replace an empirical quality study.

## 5. Implementation

The application is implemented in Python 3.9 or later. FastAPI provides the HTTP layer, Uvicorn runs the development server, Pydantic validates request objects, and `python-multipart` supports uploads. The frontend uses browser-native HTML, CSS, and JavaScript. SQLite and most preprocessing, retrieval, and offline generation functions use the Python standard library. Pytest and Playwright support automated verification. The OpenAI SDK and `python-dotenv` are installed so `.env` configuration is loaded consistently; `pypdf` remains optional.

Environment variables configure data location, chunking, embedding dimension, top-k, provider, model, and endpoint. The main service composes these settings with embedding, storage, retrieval, and generation. Ingestion extracts, identifies, chunks, embeds, and replaces document state. Question answering loads chunks, ranks them, applies the evidence gate, and returns either refusal or generation. The API maps operational errors to HTTP 400 responses.

The evaluation module loads one JSON object per line. Each case includes a stable identifier, question, expected answer, expected sources, atomic required-fact phrases, category, and difficulty. The current automatic fact-recall implementation lowercases the answer and treats a required fact as matched when at least half of its terms longer than three characters occur in the answer. This is intentionally lightweight, but it is lexical rather than semantic and can produce both false positives and false negatives. Citation-or-refusal coverage is true when at least one citation is returned or the system refuses. Average latency is computed from service diagnostics.

## 6. Evaluation Methodology

RAG should be evaluated as two coupled subsystems. Retrieval evaluation asks whether the necessary evidence appears in the ranked context. Generation evaluation asks whether the final answer is correct, complete, faithful to that context, appropriately cited, and willing to refuse unsupported questions. End-to-end latency and API cost are operational measures rather than correctness measures.

The current evaluation dataset contains 41 cases over 13 university AI research internship documents: 30 answerable cases, three ambiguous cases, three conflicting-policy cases, and five genuinely unanswerable cases. Each case carries expected source documents and, where appropriate, required fact phrases. The set exercises multi-document lookup, current-versus-archived policy conflict, clarification scenarios, and unsupported requests. It is still a development corpus rather than a held-out benchmark, so its results should guide iteration rather than establish generalisation.

The following metrics form the intended full evaluation protocol:

- **Retrieval precision at k:** proportion of returned chunks judged relevant to the question.
- **Context recall:** proportion of necessary evidence represented in retrieved context.
- **Answer correctness:** degree to which the answer matches the reference facts.
- **Groundedness:** proportion of substantive answer claims supported by retrieved evidence.
- **Citation accuracy:** proportion of citations that actually support the associated claim.
- **Refusal quality:** correct refusal on unanswerable questions without excessive refusal on answerable ones.
- **Latency:** ingestion, retrieval, provider, and end-to-end timing reported separately.
- **Cost per query:** embedding and generation tokens multiplied by recorded provider prices at experiment time.

For a defensible final experiment, retrieval judgements and answer judgements should be stored separately. At least two human reviewers should score correctness, groundedness, citation accuracy, and refusal on a 0-2 rubric, with disagreements reconciled and inter-rater agreement reported. Dataset cases should be partitioned before tuning so that constants are not selected on the same examples used for the final claim. Each stochastic provider configuration should be repeated, while deterministic components should be versioned with their settings and corpus snapshot.

## 7. Current Measured Results

### 7.1 Recorded offline baseline

The table below reproduces the current offline-extractive run stored in `evals/results.json`. No OpenAI or Ollama result is included.

| Metric | Recorded value | Interpretation |
| --- | ---: | --- |
| Evaluation cases | 41 | 30 answerable, 3 ambiguous, 3 conflicting, 5 unanswerable |
| Fact-scored cases | 36 | Unanswerable cases are excluded from fact recall |
| Average heuristic fact recall | 0.8241 | Mean lexical required-fact coverage |
| Citation-or-refusal rate | 1.0000 | Every response had a citation or an explicit refusal |
| Answerable non-refusal rate | 1.0000 | No answerable case was refused in this run |
| Unanswerable-refusal accuracy | 0.8000 | Four of five unsupported cases were refused |
| Average measured pipeline latency | 4.60 ms | Local timing across 29 indexed chunks |

### 7.2 Retrieval comparison

The same corpus and 41 cases were evaluated with vector-only, keyword-only, and hybrid ranking. The comparison uses source-level Recall@5, MRR, nDCG@5, and unanswerable abstention accuracy. The hybrid rule uses the same calibrated evidence gate as the application.

| Configuration | Recall@5 | MRR | nDCG@5 | Unanswerable abstention | Combined score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector-only | 0.9583 | 0.9028 | 0.9033 | 0.6000 | 0.8663 |
| Keyword-only | 1.0000 | 0.9306 | 0.9441 | 0.0000 | 0.8290 |
| Hybrid | 1.0000 | 0.9097 | 0.9283 | 0.8000 | **0.9126** |

Keyword-only ranking produced the strongest pure retrieval positions: it records the highest nDCG@5 (**0.9441** versus hybrid's **0.9283**) as well as the top Recall@5. Hybrid does not win on ranking quality at all; its leading combined score (**0.9126**) comes entirely from folding in unanswerable-abstention accuracy, where its calibrated evidence gate refused four of five unsupported requests while keyword-only refused none. It is also worth noting that the vector and keyword signals are strongly correlated here, because the default hashing embedder is lexical rather than a learned semantic embedding, so the "hybrid" gain is dominated by the abstention component rather than complementary dense retrieval. This is a useful engineering trade-off rather than proof of semantic superiority: the corpus and threshold are shared with tuning.

### 7.3 Interpretation and validity boundaries

The 0.8241 value is not a general answer-accuracy estimate. It is an average over 36 in-domain fact-scored examples and uses a permissive term-overlap function: a phrase is counted when at least half of its words longer than three characters occur somewhere in the answer. It does not test meaning, negation, contradiction, or whether matched words belong to the same claim. Similarly, the 1.0000 citation-or-refusal rate proves only structural coverage. The offline extractive generator emits a `[n]` marker on every sentence it returns, so citation *presence* is near-guaranteed for the recorded offline runs; the metric therefore cannot establish that citations support the claims, only that a marker exists. (The orchestration service no longer fabricates a `[1]` when a generated answer omits citations; an uncited answer is now surfaced through an `uncited` diagnostic flag rather than being silently patched.)

The 4.60 ms latency should be interpreted as a local, small-corpus, offline pipeline measurement. It excludes browser and network round-trip effects, document ingestion time, remote provider latency, and API rate limiting. It also cannot predict scaling because every query loads and scores all stored chunks. No cost was incurred by the offline generator, but no API cost measurement has yet been performed.

Automated project checks cover nine unit and evaluation tests, and the browser smoke workflow covers upload, retrieval, citation/source rendering, evaluation, console errors, and mobile layout. These tests provide regression confidence for the implemented workflow, not statistical evidence of RAG quality.

### 7.4 Learned-embedding ablation

The hashing baseline was re-run with a learned sentence-embedding model (`sentence-transformers/all-MiniLM-L6-v2`, same 384 dimensions) swapped in behind the embedder interface via `EMBEDDING_PROVIDER=sentence-transformers`, on the identical corpus and 41 cases. Retrieval comparison:

| Configuration | Recall@5 | MRR | nDCG@5 | Unanswerable abstention | Combined score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector-only | 0.9861 | 0.9398 | 0.9365 | 0.0000 | 0.8223 |
| Keyword-only | 1.0000 | 0.9306 | 0.9441 | 0.0000 | 0.8290 |
| Hybrid | 1.0000 | 0.9236 | 0.9408 | 0.0000 | 0.8260 |

End-to-end offline-extractive evaluation under the same swap recorded average fact recall **0.8056** (versus **0.8241** for hashing), unanswerable-refusal accuracy **0.0000** (versus **0.8000**), citation-or-refusal coverage 1.0000, and mean latency 29.2 ms (versus 4.6 ms). Three findings follow, and all three are more instructive than a headline number:

1. **Learned dense retrieval genuinely improves ranking.** Vector-only nDCG@5 rises 0.9033 → 0.9365, Recall@5 0.9583 → 0.9861, and MRR 0.9028 → 0.9398. This confirms that under hashing the "vector" and "keyword" channels were near-duplicate lexical signals; a learned model makes the dense channel a distinct, stronger ranker.
2. **The gain does not reach the answer metric on this corpus.** Extractive fact recall actually slips (0.8241 → 0.8056) and keyword-only still leads on nDCG@5. This hand-written corpus shares vocabulary between questions and evidence, so lexical matching is already near-ceiling; dense retrieval pays off most under paraphrase and vocabulary mismatch, which this corpus underrepresents.
3. **The evidence gate does not transfer across embedders.** Unanswerable abstention collapses to 0.0000 for every mode, because the refusal thresholds (`min_hybrid_score`, `vector_score < 0.20`, and related constants) were calibrated to the hashing score scale; a learned model's cosine magnitudes occupy a different range, so the gate never fires. Refusal thresholds are therefore embedder-specific and must be re-tuned on a validation split for each embedding model, not treated as portable constants.

The practical conclusion is to keep hashing as the zero-dependency deterministic default and treat learned embeddings as an opt-in that requires threshold recalibration (plus a ~6× query-latency cost and a heavyweight optional dependency) before its retrieval gains translate end to end. Raw results are in `evals/comparison_results_sbert.json` and `evals/results_sbert.json`.

### 7.5 Remaining comparison work

Human-reviewed groundedness, claim-level citation accuracy, a cross-encoder reranker, and remote OpenAI/Ollama generation results remain pending. Those experiments should use a held-out split and report latency distribution, tokens, cost, and provider failures separately. The learned-embedding ablation above should be repeated after the refusal thresholds are recalibrated to the learned model's score scale, since the current 0.0 abstention figures reflect uncalibrated gating rather than a retrieval limitation.

## 8. Limitations

The most important limitation is empirical coverage. Although 41 cases are substantially stronger than a smoke test, the corpus is synthetic, close to the system domain, and reused to calibrate the evidence threshold. There is no held-out test set, human judgement, inter-rater analysis, statistical uncertainty, stress test, or regression history across model versions. The 80% unanswerable-refusal result is therefore an iteration signal, not a deployment-quality estimate.

The hashing embedder is deterministic but weak for semantic retrieval. Hybrid ranking partly compensates through BM25 and overlap, yet all signals remain strongly lexical. Retrieval constants and refusal thresholds are hand-selected rather than calibrated. The system has no reranker, query expansion, metadata filters, conversation-aware retrieval, or method for resolving conflicting sources.

Document handling is also limited. Scanned PDFs require OCR, structural information such as headings and tables is not modelled, and overlap may duplicate evidence. All vectors are stored as JSON and all chunks are scanned in memory per query. This is acceptable for a demonstration but inefficient for large or concurrent workloads.

Generation safeguards are syntactic rather than entailment-based. Invalid citation numbers are removed, but valid citations are not checked against individual claims: a present `[n]` marker guarantees formatting rather than support. Remote provider adapters do not yet implement retries, backoff, token accounting, streaming, structured output validation, or provider-specific error classification. OpenAI and Ollama integration paths have not been included in the recorded quality comparison.

Finally, the application is configured for trusted local use. It lacks authentication, tenancy, authorisation, quotas, audit logs, deployment hardening, and operational monitoring. In particular, destructive operations are unauthenticated: `DELETE /api/documents` and the ingest reset both replace or clear the corpus for any caller, which is acceptable only for a local single-user demo. These omissions prevent responsible use with sensitive or regulated data.

## 9. Ethics, Privacy, and Security

Grounded answers can improve traceability, but RAG does not remove ethical risk. A corpus may contain bias, outdated policies, personal data, copyrighted material, or malicious instructions. Retrieval can make sensitive passages visible to users who should not receive them, and a cited answer may be trusted too readily even when context is incomplete. Human oversight remains necessary for consequential decisions.

The API has no authentication or document-level access control. CORS allows all origins, and path ingestion accepts a server-side path. An exposed deployment therefore requires an authorised path allow-list, strict origins, identity and access checks, file and request limits, malware scanning, and rate limiting. Files and extracted content must be treated as untrusted; parsers should be isolated and logs should avoid confidential content.

Prompt injection is a specific threat: indexed text can instruct a generator to ignore rules or disclose data. The current prompt formats context separately but does not detect malicious content. Controls should include provenance, filtering, least-privilege retrieval, policy tests, and output validation.

Provider use introduces additional data-governance questions. Before sending content to a remote API, the operator must determine whether the corpus and queries may leave the local environment, how long the provider retains data, and which region and contractual controls apply. API keys must remain in environment variables or a secret manager, never documents, frontend code, logs, or source control. The offline provider and a locally controlled Ollama deployment offer stronger data locality, but local execution still requires host, model, and access security.

Evaluation should examine quality across language varieties and specialised terminology. Users should see sources and insufficient-evidence notices, be able to correct or remove indexed information, and treat consequential outputs as decision support.

## 10. Six-Week Internship Timeline

- **Week 1 - Problem formulation and design:** Define RAG reliability questions; specify module boundaries, data contracts, metrics, repository structure, and a reproducible environment.
- **Week 2 - Ingestion and storage:** Implement text/PDF extraction, normalisation, stable identifiers, paragraph-aware chunking, metadata, SQLite storage, and ingestion tests.
- **Week 3 - Retrieval baseline:** Implement hashing embeddings, cosine similarity, BM25-style scoring, RRF hybrid ranking, diagnostics, and evidence-threshold refusal.
- **Week 4 - Generation and application integration:** Implement the shared grounded prompt, offline provider, OpenAI and Ollama adapters, citation validation, FastAPI endpoints, and responsive web interface.
- **Week 5 - Evaluation and verification:** Create JSONL cases and automatic metrics; add unit and browser workflows; run the offline baseline; inspect failure cases and document validity limits.
- **Week 6 - Analysis and reporting:** Consolidate architecture and results; run retrieval ablations; document ethics, security, and limitations; prepare demonstration, final report, and presentation material.

This timeline reflects a logical six-week execution sequence. A subsequent experimental phase is still required for held-out, human-reviewed, and remote-provider comparisons.

## 11. Future Work

The immediate priority is a held-out domain dataset with chunk-level relevance labels. The implemented 41-case set covers lookup, conflict, ambiguity, and unanswerable requests, but a separate validation and test split is needed before further threshold tuning. Human review should establish correctness, groundedness, citation accuracy, and refusal quality.

The next experiment should tune chunk size, overlap, fusion weights, and refusal thresholds on a validation split, then compare a learned embedding model and optional cross-encoder reranker against the documented vector-only, keyword-only, and hybrid baselines. Quality, latency, memory, and cost should be measured separately.

API-backed generation should be tested only after the retrieval dataset is fixed. OpenAI and Ollama runs should record model identity, model version when available, temperature, prompt version, token usage, latency distribution, failure rate, and cost. Answers should be stored with retrieved context to permit blinded review. Streaming can improve perceived responsiveness, but it should not obscure final citation validation.

Production work includes approximate-nearest-neighbour indexing, incremental ingestion, structured extraction, OCR, query rewriting, metadata filters, and conflict-aware answers. Security requires identity, role-based access, safe paths, strict CORS, secrets management, quotas, auditing, deletion, and injection tests. Observability should separate stage latency and track refusals and unsupported claims.

## 12. Conclusion

This project implemented a complete RAG knowledge assistant that joins document engineering, hybrid retrieval, constrained generation, citations, API delivery, a web interface, and evaluation in one inspectable system. Its offline baseline makes the pipeline reproducible and demonstrates how evidence can be retrieved, surfaced, and linked to answers without requiring an external model. Optional OpenAI and Ollama adapters show how the same orchestration boundary can support API-backed generation.

The recorded results show that the 41-case development set is executable and that the offline system returns substantial required-fact coverage, citation-or-refusal structure, and four correct refusals out of five unsupported requests at low local latency. More importantly, analysis of the metrics shows why those numbers must not be overstated. Citation presence is not citation correctness, lexical overlap is not semantic correctness, local baseline latency is not remote-service latency, and a synthetic development corpus is not production evidence.

The central learning is that advanced AI application development is an exercise in controlled system design. Reliable RAG requires traceable data, measurable retrieval, explicit evidence boundaries, refusal behaviour, provider abstraction, security controls, and evaluation that tests claims rather than presentation. The implemented project provides a sound internship-level foundation; held-out and human evaluation define the next path from a functional prototype toward a defensible applied research result.
