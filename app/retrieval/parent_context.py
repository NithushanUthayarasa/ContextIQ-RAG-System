"""
ContextIQ - Parent Context Resolver Module
Expands retrieved and reranked child chunks into their corresponding parent contexts
while eliminating duplicate parents and preserving the strongest child relevance metadata.
"""

from dataclasses import replace
from typing import Any, Dict, List, Optional

from app.retrieval.models import RetrievalResult, RetrievedChunk


def _is_stronger_child(candidate: RetrievedChunk, current_best: RetrievedChunk) -> bool:
    """
    Determines whether candidate chunk has stronger retrieval/reranking relevance
    evidence than current_best.

    Comparison hierarchy:
    1. rerank_score (higher is better)
    2. distance (lower is better)
    3. cosine_similarity (higher is better)
    4. rrf_score (higher is better)
    5. bm25_score (higher is better)
    """
    # 1. Rerank score comparison
    c_rerank = getattr(candidate, "rerank_score", None)
    b_rerank = getattr(current_best, "rerank_score", None)
    if c_rerank is not None and b_rerank is not None:
        if c_rerank != b_rerank:
            return c_rerank > b_rerank
    elif c_rerank is not None and b_rerank is None:
        return True
    elif c_rerank is None and b_rerank is not None:
        return False

    # 2. Vector distance comparison (smaller is better)
    if candidate.distance is not None and current_best.distance is not None:
        if candidate.distance != current_best.distance:
            return candidate.distance < current_best.distance
    elif candidate.distance is not None and current_best.distance is None:
        return True
    elif candidate.distance is None and current_best.distance is not None:
        return False

    # 3. Cosine similarity comparison (higher is better)
    if candidate.cosine_similarity is not None and current_best.cosine_similarity is not None:
        if candidate.cosine_similarity != current_best.cosine_similarity:
            return candidate.cosine_similarity > current_best.cosine_similarity

    # 4. RRF hybrid score comparison (higher is better)
    c_rrf = getattr(candidate, "rrf_score", None)
    b_rrf = getattr(current_best, "rrf_score", None)
    if c_rrf is not None and b_rrf is not None:
        if c_rrf != b_rrf:
            return c_rrf > b_rrf
    elif c_rrf is not None and b_rrf is None:
        return True
    elif c_rrf is None and b_rrf is not None:
        return False

    # 5. BM25 score comparison (higher is better)
    c_bm25 = getattr(candidate, "bm25_score", None)
    b_bm25 = getattr(current_best, "bm25_score", None)
    if c_bm25 is not None and b_bm25 is not None:
        if c_bm25 != b_bm25:
            return c_bm25 > b_bm25
    elif c_bm25 is not None and b_bm25 is None:
        return True
    elif c_bm25 is None and b_bm25 is not None:
        return False

    return False


def _fetch_parents_map(
    parent_ids: List[str],
    parent_store: Optional[Any],
) -> Dict[str, Any]:
    """
    Fetches parent records from parent_store supporting various store interfaces:
    - Dictionary or Mapping: {pid: record}
    - Object with get_parents(ids): returns dict of {pid: record}
    - Object with get_parent(id): returns record
    - Object with get(ids=...): returns ChromaDB query dict
    """
    if not parent_ids or parent_store is None:
        return {}

    # Case 1: Custom parent store with get_parents method
    if hasattr(parent_store, "get_parents") and callable(parent_store.get_parents):
        try:
            return parent_store.get_parents(parent_ids)
        except Exception:
            return {}

    # Case 2: In-memory dictionary
    if isinstance(parent_store, dict):
        return {pid: parent_store[pid] for pid in parent_ids if pid in parent_store}

    # Case 3: Store with single get_parent method
    if hasattr(parent_store, "get_parent") and callable(parent_store.get_parent):
        res = {}
        for pid in parent_ids:
            try:
                p = parent_store.get_parent(pid)
                if p:
                    res[pid] = p
            except Exception:
                pass
        return res

    # Case 4: ChromaDB-style collection or vector store with get(ids=...)
    if hasattr(parent_store, "get") and callable(parent_store.get):
        try:
            raw = parent_store.get(ids=parent_ids)
            ids = raw.get("ids", [])
            docs = raw.get("documents", [])
            metas = raw.get("metadatas", [])
            res = {}
            for i, pid in enumerate(ids):
                doc_text = docs[i] if i < len(docs) else ""
                meta = metas[i] if i < len(metas) and metas[i] else {}
                res[pid] = {
                    "parent_id": pid,
                    "text": doc_text,
                    "source": meta.get("source", "unknown"),
                    "page_number": meta.get("page_number", 1),
                    "parent_index": meta.get("parent_index", 0),
                    "document_id": meta.get("document_id"),
                }
            return res
        except Exception:
            return {}

    return {}


def resolve_parent_context(
    child_chunks: List[RetrievedChunk],
    parent_store: Optional[Any] = None,
) -> List[RetrievedChunk]:
    """
    Resolves a list of retrieved child chunks to their corresponding parent contexts.

    Guarantees:
    1. Returns empty list if child_chunks is empty.
    2. Multiple child chunks that reference the same parent_id are merged into ONE parent chunk.
    3. The merged parent preserves the strongest relevance metrics (distance, cosine_similarity,
       bm25_score, rrf_score, rerank_score) from its constituent children.
    4. Sets `original_child_id` on the resolved chunk pointing to the best child chunk.
    5. Gracefully falls back to the child chunk itself if:
       - The child has no parent_id (e.g. legacy indexed chunk), OR
       - The parent_id is not found in parent_store, OR
       - parent_store is None.
    6. Preserves the ordering established by the retrieval/reranking step.
    """
    if not child_chunks:
        return []

    # Collect unique parent IDs needed
    needed_pids = list(
        {
            getattr(c, "parent_id", None)
            for c in child_chunks
            if getattr(c, "parent_id", None) and str(getattr(c, "parent_id", None)).strip()
        }
    )

    parents_map = _fetch_parents_map(needed_pids, parent_store)

    ordered_keys: List[str] = []
    groups: Dict[str, Dict[str, Any]] = {}

    for chunk in child_chunks:
        pid = getattr(chunk, "parent_id", None)
        parent_record = parents_map.get(pid) if pid else None

        if pid and parent_record is not None:
            group_key = f"parent::{pid}"
            if group_key not in groups:
                ordered_keys.append(group_key)
                groups[group_key] = {
                    "type": "parent",
                    "parent_id": pid,
                    "parent_record": parent_record,
                    "best_child": chunk,
                }
            else:
                # Merge duplicate parent reference, keeping strongest child
                if _is_stronger_child(chunk, groups[group_key]["best_child"]):
                    groups[group_key]["best_child"] = chunk
        else:
            # Standalone child chunk fallback
            group_key = f"child::{chunk.chunk_id}"
            if group_key not in groups:
                ordered_keys.append(group_key)
                groups[group_key] = {
                    "type": "child",
                    "chunk": chunk,
                }

    resolved_results: List[RetrievedChunk] = []

    for key in ordered_keys:
        item = groups[key]
        if item["type"] == "parent":
            pid = item["parent_id"]
            p_rec = item["parent_record"]
            best_child = item["best_child"]

            # Extract parent fields whether p_rec is ParentChunk or dict
            p_text = (
                getattr(p_rec, "text", None)
                if hasattr(p_rec, "text")
                else p_rec.get("text", "")
            )
            p_source = (
                getattr(p_rec, "source", None)
                if hasattr(p_rec, "source")
                else p_rec.get("source", best_child.source)
            )
            p_page = (
                getattr(p_rec, "page_number", None)
                if hasattr(p_rec, "page_number")
                else p_rec.get("page_number", best_child.page_number)
            )
            p_index = (
                getattr(p_rec, "parent_index", None)
                if hasattr(p_rec, "parent_index")
                else p_rec.get("parent_index", 0)
            )
            p_doc_id = (
                getattr(p_rec, "document_id", None)
                if hasattr(p_rec, "document_id")
                else p_rec.get("document_id", best_child.document_id)
            )

            parent_chunk = RetrievalResult(
                chunk_id=pid,
                text=p_text,
                source=p_source,
                page_number=int(p_page),
                chunk_index=int(p_index),
                distance=best_child.distance,
                document_id=p_doc_id,
                bm25_score=best_child.bm25_score,
                semantic_rank=best_child.semantic_rank,
                bm25_rank=best_child.bm25_rank,
                rrf_score=best_child.rrf_score,
                retrieval_method=best_child.retrieval_method,
                rerank_score=best_child.rerank_score,
                original_rank=best_child.original_rank,
                parent_id=pid,
                parent_index=int(p_index),
                original_child_id=best_child.chunk_id,
            )
            resolved_results.append(parent_chunk)
        else:
            c = item["chunk"]
            fallback = replace(
                c,
                original_child_id=getattr(c, "original_child_id", None) or c.chunk_id,
            )
            resolved_results.append(fallback)

    return resolved_results
