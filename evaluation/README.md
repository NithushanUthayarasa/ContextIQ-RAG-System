# ContextIQ Advanced RAG Evaluation Framework

This directory contains the ground-truth benchmark dataset and evaluation artifacts used to measure the end-to-end performance of **ContextIQ-RAG** across retrieval, ranking, context compression, and answer provenance.

---

## 1. Evaluation Purpose

Phase 14 Step 5 benchmarks the advanced RAG pipeline layers:
- **Layer A: Retrieval Quality**: Hit@K, Recall@K, and MRR@K ($K \in \{3, 5\}$).
- **Layer B: Context Efficiency & Evidence Preservation**: Character reduction % and deterministic Evidence Retention Rate (ERR).
- **Layer C: Provenance & Robustness**: Citation Precision/Recall and Unanswerable/Refusal Accuracy.

---

## 2. Dataset Format (`retrieval_dataset.json`)

The dataset comprises 15 curated questions verified against the indexed lecture document (`SE3090 Lecture 04 Database Auth Integration.pdf`).

### Schema:
```json
{
  "id": "q01",
  "query_type": "conceptual",
  "question": "What is the difference between authentication and authorization?",
  "expected_source": "SE3090 Lecture 04 Database Auth Integration.pdf",
  "expected_pages": [40],
  "expected_evidence": [
    "Who are you",
    "What may you do",
    "401 Unauthorized"
  ],
  "unanswerable": false
}
```

### Query Categories:
- **Conceptual** (3): Broad architectural concepts (AuthN vs AuthZ, ACID principles, 2NF).
- **Technical Entity** (3): System components (JWT structure, ORM with Npgsql, Connection Pooling).
- **Exact Detail** (3): Schema attributes and syntax (password_hash column, Authorize attribute, HasIndex).
- **Multi-Page** (3): Multi-hop comparisons (Surrogate vs Natural keys, Access vs Refresh tokens, SQL injection defense).
- **Unanswerable / Negative** (3): Out-of-domain queries testing refusal behavior (Redis clustering, Kubernetes HPA, OAuth PKCE).

---

## 3. Metrics

| Metric | Layer | Formula / Logic |
|---|---|---|
| **Hit@K** | Retrieval | 1.0 if at least one relevant chunk appears within top-K, else 0.0. |
| **Recall@K** | Retrieval | Fraction of expected pages covered in top-K. |
| **MRR@K** | Retrieval | Reciprocal rank ($1 / \text{rank}$) of the first relevant chunk in top-K. |
| **Context Reduction %** | Context Efficiency | Percentage of characters saved by extractive context compression. |
| **Evidence Retention Rate (ERR)** | Context Preservation | Fraction of expected evidence phrases preserved in compressed text relative to uncompressed context. |
| **Citation Precision / Recall** | Provenance | Set overlap of cited `(source, page)` pairs vs ground truth. |
| **Refusal Accuracy** | Robustness | Correct generation of insufficient-information statement on unanswerable queries. |

---

## 4. Ablation Configurations

The benchmark evaluates five pipeline configurations:
1. **C1 (Hybrid Baseline)**: Semantic vector + BM25 keyword search with Reciprocal Rank Fusion ($k=60$).
2. **C2 (Hybrid + Reranking)**: Hybrid retrieval + candidate pool reranked by TF-IDF term overlap.
3. **C3 (Hybrid + Reranking + Query Expansion)**: Multi-query candidate retrieval + reranking.
4. **C4 (Hybrid + Reranking + Parent/Child)**: Small child chunk matching expanded to full parent sections.
5. **C5 (Full Advanced RAG)**: Child matching + Reranking + Parent expansion + Extractive Context Compression.

---

## 5. How to Run Evaluation

### Deterministic Offline Benchmark (Zero API Calls):
```powershell
python scripts/evaluate_retrieval.py --ablation
```

### Options:
- `--top-k 5`: Number of chunks passed to generator (default: 5).
- `--dataset path/to/dataset.json`: Custom dataset path.
- `--output-json evaluation/results.json`: Machine-readable results export.
- `--output-report evaluation/REPORT.md`: Comprehensive Markdown report export.
- `--live-gemini`: Enable live Gemini API query expansion (default: False, uses deterministic offline expander).

---

## 6. Deterministic vs. Live API Evaluation

- **Automated Tests & Default CLI**: Runs 100% offline using deterministic mocks and offline query expansion. Zero Gemini API calls, reproducible in CI/CD.
- **Live API Run**: When `--live-gemini` is passed, live LLM calls are made for query expansion and generation.

---

## 7. Key Findings & Empirical Scope

1. **C1 Hybrid Baseline is Strongest**: C1 Hybrid Baseline (Semantic + BM25 + RRF) is currently the highest-performing retrieval configuration on this 15-query benchmark (Hit@5 = 1.000, MRR@5 = 0.938).
2. **TF-IDF Reranking on Sparse Slides**: TF-IDF reranking reduced retrieval metrics on this particular sparse slide-deck corpus (MRR@5 dropped to 0.708). On presentation slides where title/overview pages repeat query keywords, lexical term frequency can rank structural overview pages above concise content slides. This is an empirical limitation of lexical TF-IDF on presentation slides, not a software bug.
3. **Evaluation vs. Production Query Expansion**: Offline deterministic query expansion is used strictly for reproducible, offline evaluation and is NOT equivalent to production Gemini query expansion. On this test set, rule-based expansion did not improve retrieval, whereas production Gemini expansion uses LLM semantic reasoning to generate rich natural language reformulations.
4. **Context Compression Trade-Off**: Context compression provides substantial context reduction (~70.3% character reduction, down from 3,298 to 973 average characters) but currently exhibits an evidence-retention trade-off (~61.1% strict evidence retention). While it saves significant prompt budget, short slide bullets without direct query term overlap can be filtered out.
5. **Corpus Scope**: This evaluation uses one current document/corpus (`SE3090 Lecture 04 Database Auth Integration.pdf`) and 15 curated questions; results should not be generalized to all datasets or diverse document formats.
