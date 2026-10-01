# Retrieval Evaluation Dataset

This dataset is used to benchmark the three retrieval strategies implemented in **ContextIQ‑RAG**:

- **semantic** – vector similarity search with ChromaDB.
- **bm25** – keyword search using the in‑memory Okapi BM25 index.
- **hybrid** – Reciprocal Rank Fusion of the above two.

The file `retrieval_dataset.json` contains a list of queries together with the IDs of chunks that are considered **relevant** for each query.  The IDs refer to the `chunk_id` values stored in ChromaDB (and mirrored in the BM25 index).

The dataset is deliberately small (≈15 queries) and mixes different query types:

| Type | Description |
|------|-------------|
| A – Semantic / conceptual | Questions that require understanding of concepts. |
| B – Exact terminology | Queries that contain the exact term appearing in the documents. |
| C – Technical entity | Questions about specific components (e.g., ChromaDB). |
| D – Multi‑word concept | Longer, more descriptive queries. |
| E – Specific detail | Queries that ask for a concrete configuration value. |

The evaluation scripts read this file, invoke the `Retriever` in the requested mode, and compute standard IR metrics (Hit@K, Recall@K, MRR@K) for K = 3, 5, 10.

**Important:** The dataset is tied to the current document version.  If the underlying documents change (e.g., re‑ingestion), the `chunk_id`s will also change and the ground‑truth will need to be regenerated.
