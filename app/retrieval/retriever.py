"""
ContextIQ - Document Retriever Module
Coordinates semantic vector search (ChromaDB), keyword search (BM25), and hybrid fusion (RRF).
"""

from typing import Any, Dict, List, Optional, Sequence, Set

from app.config import (
    DEFAULT_HYBRID_CANDIDATE_MULTIPLIER,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
)
from app.ingestion.embedder import GeminiEmbedder
from app.retrieval.bm25 import BM25Index, BM25Result
from app.retrieval.models import RetrievalResult, RetrievedChunk
from app.vectorstore.chroma_store import ChromaVectorStore


class RetrieverError(Exception):
    """Base exception for retrieval failures."""
    pass


class EmptyQueryError(RetrieverError, ValueError):
    """Raised when an empty or whitespace query is submitted for retrieval."""
    pass


class InvalidTopKError(RetrieverError, ValueError):
    """Raised when top_k is invalid (<= 0 or not an integer)."""
    pass


class InvalidSimilarityThresholdError(RetrieverError, ValueError):
    """Raised when similarity_threshold is invalid (not a numeric in [0.0, 1.0])."""
    pass


class InvalidDocumentFilterError(RetrieverError, ValueError):
    """Raised when document_ids filter argument is malformed or invalid."""
    pass


class InvalidRetrievalModeError(RetrieverError, ValueError):
    """Raised when an unsupported retrieval mode is specified."""
    pass


VALID_RETRIEVAL_MODES = {"semantic", "bm25", "hybrid"}


def _validate_similarity_threshold(threshold: Any) -> float:
    """Validates that a similarity threshold is a valid float/int in [0.0, 1.0]."""
    if threshold is None or isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise InvalidSimilarityThresholdError(
            f"similarity_threshold must be a numeric value between 0.0 and 1.0, got: {threshold}"
        )
    val = float(threshold)
    if val < 0.0 or val > 1.0:
        raise InvalidSimilarityThresholdError(
            f"similarity_threshold must be between 0.0 and 1.0, got: {threshold}"
        )
    return val


def _validate_and_normalize_document_ids(
    document_ids: Optional[Any],
) -> Optional[List[str]]:
    """
    Validates and normalizes document_ids input for document filtering.

    Rules:
    - None -> None (no filtering, search all documents)
    - [] (empty list or empty collection) -> None (treated as no restriction, search all documents)
    - string -> raise InvalidDocumentFilterError (prevent strings being iterated character-by-character)
    - non-iterable / non-sequence -> raise InvalidDocumentFilterError
    - elements must be non-empty strings
    - preserves deduplicated unique document IDs
    """
    if document_ids is None:
        return None

    if isinstance(document_ids, (str, bytes)):
        raise InvalidDocumentFilterError(
            f"document_ids must be a sequence of strings, not {type(document_ids).__name__}. Pass a list like ['{document_ids}']."
        )

    if not hasattr(document_ids, "__iter__"):
        raise InvalidDocumentFilterError(
            f"document_ids must be an iterable sequence of strings, got: {type(document_ids).__name__}"
        )

    clean_ids: List[str] = []
    seen = set()
    for item in document_ids:
        if not isinstance(item, str) or not item.strip():
            raise InvalidDocumentFilterError(
                f"Each document_id must be a non-empty string, got: {repr(item)}"
            )
        cleaned = item.strip()
        if cleaned not in seen:
            seen.add(cleaned)
            clean_ids.append(cleaned)

    if not clean_ids:
        return None

    return clean_ids


class Retriever:
    """
    Coordinates multi-modal document retrieval:
    - Semantic retrieval using Gemini Embeddings and ChromaDB cosine similarity.
    - Keyword retrieval using Okapi BM25.
    - Hybrid retrieval fusing semantic and keyword results using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        embedder: GeminiEmbedder,
        vector_store: ChromaVectorStore,
        default_top_k: int = DEFAULT_TOP_K,
        default_min_similarity: float = DEFAULT_MIN_SIMILARITY,
        bm25_index: Optional[BM25Index] = None,
        default_retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_multiplier: int = DEFAULT_HYBRID_CANDIDATE_MULTIPLIER,
    ):
        if embedder is None:
            raise RetrieverError("An embedder instance must be provided to Retriever.")
        if vector_store is None:
            raise RetrieverError("A vector_store instance must be provided to Retriever.")

        self.embedder = embedder
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.default_min_similarity = _validate_similarity_threshold(default_min_similarity)
        self.bm25_index = bm25_index if bm25_index is not None else BM25Index()
        self.default_retrieval_mode = default_retrieval_mode
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def sync_bm25_index(self, force: bool = False) -> int:
        """
        Synchronizes the in-memory BM25 index with chunks from the Chroma vector store.

        Returns:
            The number of chunks indexed in BM25.
        """
        vs_count = self.vector_store.count()
        if not force and self.bm25_index.chunk_count == vs_count and vs_count > 0:
            return self.bm25_index.chunk_count

        if vs_count == 0:
            self.bm25_index.clear()
            return 0

        chunks = self.vector_store.get_all_chunks()
        return self.bm25_index.add_records(chunks)

    def _retrieve_semantic(
        self,
        query: str,
        top_k: int,
        where_filter: Optional[Dict[str, Any]],
        thresh: float,
    ) -> List[RetrievedChunk]:
        """Executes vector search against ChromaDB and filters by similarity threshold."""
        try:
            query_vector = self.embedder.embed_query(query)
        except Exception as e:
            raise RetrieverError(f"Failed to generate query embedding: {str(e)}") from e

        try:
            raw_results = self.vector_store.query(
                query_embedding=query_vector,
                top_k=top_k,
                where=where_filter,
            )
        except Exception as e:
            raise RetrieverError(f"Failed to execute vector store query: {str(e)}") from e

        ids_list = raw_results.get("ids", [[]])
        docs_list = raw_results.get("documents", [[]])
        metas_list = raw_results.get("metadatas", [[]])
        dists_list = raw_results.get("distances", [[]])

        if not ids_list or not ids_list[0]:
            return []

        ids = ids_list[0]
        documents = docs_list[0] if docs_list else []
        metadatas = metas_list[0] if metas_list else []
        distances = dists_list[0] if dists_list else []

        results: List[RetrievedChunk] = []
        rank = 1
        for cid, doc, meta, dist in zip(ids, documents, metadatas, distances):
            meta_dict = meta or {}
            chunk = RetrievedChunk(
                chunk_id=cid,
                text=doc,
                source=meta_dict.get("source", "unknown"),
                page_number=int(meta_dict.get("page_number", 1)),
                chunk_index=int(meta_dict.get("chunk_index", 0)),
                distance=float(dist),
                document_id=meta_dict.get("document_id"),
                semantic_rank=rank,
                retrieval_method="semantic",
            )
            # Filter by minimum cosine similarity threshold
            if chunk.cosine_similarity >= thresh:
                results.append(chunk)
                rank += 1

        return results

    def _retrieve_bm25(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]],
    ) -> List[RetrievedChunk]:
        """Executes BM25 keyword search against in-memory index."""
        self.sync_bm25_index()
        if self.bm25_index.chunk_count == 0:
            return []

        bm25_results = self.bm25_index.search(
            query=query,
            top_k=top_k,
            document_ids=document_ids,
        )

        results: List[RetrievedChunk] = []
        for rank, r in enumerate(bm25_results, start=1):
            chunk = RetrievedChunk(
                chunk_id=r.chunk_id,
                text=r.text,
                source=r.source,
                page_number=r.page_number,
                chunk_index=r.chunk_index,
                distance=None,
                document_id=r.document_id,
                bm25_score=r.score,
                bm25_rank=rank,
                retrieval_method="bm25",
            )
            results.append(chunk)

        return results

    def _fuse_hybrid(
        self,
        semantic_chunks: List[RetrievedChunk],
        bm25_chunks: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """Fuses semantic and keyword candidates using Reciprocal Rank Fusion (RRF)."""
        combined_chunks: Dict[str, RetrievedChunk] = {}

        for chunk in semantic_chunks:
            sem_rank = chunk.semantic_rank or 1
            rrf_sem = 1.0 / (self.rrf_k + sem_rank)
            chunk.rrf_score = rrf_sem
            chunk.retrieval_method = "semantic"
            combined_chunks[chunk.chunk_id] = chunk

        for chunk in bm25_chunks:
            bm_rank = chunk.bm25_rank or 1
            rrf_bm = 1.0 / (self.rrf_k + bm_rank)
            cid = chunk.chunk_id

            if cid in combined_chunks:
                existing = combined_chunks[cid]
                existing.bm25_score = chunk.bm25_score
                existing.bm25_rank = chunk.bm25_rank
                existing.rrf_score = (existing.rrf_score or 0.0) + rrf_bm
                existing.retrieval_method = "hybrid"
            else:
                chunk.rrf_score = rrf_bm
                chunk.retrieval_method = "bm25"
                combined_chunks[cid] = chunk

        fused = list(combined_chunks.values())
        fused.sort(key=lambda c: (-(c.rrf_score or 0.0), c.chunk_id))
        return fused[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        document_ids: Optional[List[str]] = None,
        retrieval_mode: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieves relevant document chunks using the specified retrieval_mode:
        - "semantic": ChromaDB vector similarity search with cosine similarity threshold filtering.
        - "bm25": Okapi BM25 keyword search.
        - "hybrid": Reciprocal Rank Fusion of semantic and BM25 candidates.

        Args:
            query: The user's search query or question.
            top_k: Maximum number of candidate chunks to return.
            where: Optional base metadata filter dictionary for vector store.
            similarity_threshold: Optional override for minimum cosine similarity threshold (0.0 to 1.0).
            document_ids: Optional list of document_id strings to restrict retrieval scope.
            retrieval_mode: Optional retrieval mode ("semantic", "bm25", "hybrid").

        Returns:
            List of RetrievedChunk objects ordered by relevance.

        Raises:
            EmptyQueryError: If query is empty or whitespace-only.
            InvalidTopKError: If top_k is <= 0 or not an integer.
            InvalidSimilarityThresholdError: If similarity_threshold is outside [0.0, 1.0].
            InvalidDocumentFilterError: If document_ids is malformed.
            InvalidRetrievalModeError: If retrieval_mode is not one of {"semantic", "bm25", "hybrid"}.
            RetrieverError: If retrieval fails.
        """
        if not query or not query.strip():
            raise EmptyQueryError("Retrieval query must not be empty.")

        cleaned_query = query.strip()

        k = top_k if top_k is not None else self.default_top_k
        if not isinstance(k, int) or k <= 0:
            raise InvalidTopKError(f"top_k must be a positive integer, got: {top_k}")

        thresh = (
            _validate_similarity_threshold(similarity_threshold)
            if similarity_threshold is not None
            else self.default_min_similarity
        )

        norm_doc_ids = _validate_and_normalize_document_ids(document_ids)

        mode_str = retrieval_mode if retrieval_mode is not None else self.default_retrieval_mode
        if not isinstance(mode_str, str) or mode_str.strip().lower() not in VALID_RETRIEVAL_MODES:
            raise InvalidRetrievalModeError(
                f"Invalid retrieval_mode '{mode_str}'. Allowed modes: {', '.join(sorted(VALID_RETRIEVAL_MODES))}."
            )
        mode = mode_str.strip().lower()

        # If vector store is empty, return empty list gracefully
        if self.vector_store.count() == 0:
            return []

        # Build ChromaDB metadata filter
        where_filter = where
        if norm_doc_ids is not None:
            if len(norm_doc_ids) == 1:
                doc_filter: Dict[str, Any] = {"document_id": norm_doc_ids[0]}
            else:
                doc_filter = {"document_id": {"$in": norm_doc_ids}}

            if where is not None:
                where_filter = {"$and": [where, doc_filter]}
            else:
                where_filter = doc_filter

        if mode == "semantic":
            return self._retrieve_semantic(cleaned_query, k, where_filter, thresh)
        elif mode == "bm25":
            return self._retrieve_bm25(cleaned_query, k, norm_doc_ids)
        elif mode == "hybrid":
            candidate_k = k * self.candidate_multiplier
            sem_candidates = self._retrieve_semantic(cleaned_query, candidate_k, where_filter, thresh)
            bm_candidates = self._retrieve_bm25(cleaned_query, candidate_k, norm_doc_ids)
            return self._fuse_hybrid(sem_candidates, bm_candidates, k)
        else:
            raise InvalidRetrievalModeError(f"Unsupported retrieval mode: {mode}")
