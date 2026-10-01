"""
ContextIQ - Document Retriever Module
Coordinates query vectorization, similarity search against ChromaDB, and formatting of retrieved chunks.
"""

from typing import Any, Dict, List, Optional

from app.config import DEFAULT_MIN_SIMILARITY, DEFAULT_TOP_K
from app.ingestion.embedder import GeminiEmbedder
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


class Retriever:
    """
    Coordinates semantic retrieval using an injected GeminiEmbedder and ChromaVectorStore,
    with configurable minimum similarity threshold filtering.
    """

    def __init__(
        self,
        embedder: GeminiEmbedder,
        vector_store: ChromaVectorStore,
        default_top_k: int = DEFAULT_TOP_K,
        default_min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ):
        if embedder is None:
            raise RetrieverError("An embedder instance must be provided to Retriever.")
        if vector_store is None:
            raise RetrieverError("A vector_store instance must be provided to Retriever.")

        self.embedder = embedder
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.default_min_similarity = _validate_similarity_threshold(default_min_similarity)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """
        Embeds a user query, queries ChromaDB, and returns the top_k most relevant chunks
        that satisfy the minimum cosine similarity threshold.

        Args:
            query: The user's search query or question.
            top_k: Maximum number of candidate chunks to retrieve from vector store.
            where: Optional metadata filter dictionary.
            similarity_threshold: Optional override for minimum cosine similarity threshold (0.0 to 1.0).

        Returns:
            List of RetrievedChunk objects ordered by cosine distance (nearest first),
            retaining only those with cosine_similarity >= threshold.

        Raises:
            EmptyQueryError: If query is empty or whitespace-only.
            InvalidTopKError: If top_k is <= 0 or not an integer.
            InvalidSimilarityThresholdError: If similarity_threshold is outside [0.0, 1.0].
            RetrieverError: If embedding or vector store query fails.
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

        # If vector store is empty, return empty list gracefully
        if self.vector_store.count() == 0:
            return []

        try:
            query_vector = self.embedder.embed_query(cleaned_query)
        except Exception as e:
            raise RetrieverError(f"Failed to generate query embedding: {str(e)}") from e

        try:
            raw_results = self.vector_store.query(
                query_embedding=query_vector,
                top_k=k,
                where=where,
            )
        except Exception as e:
            raise RetrieverError(f"Failed to execute vector store query: {str(e)}") from e

        retrieved_chunks: List[RetrievedChunk] = []

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
            )
            # Filter by minimum cosine similarity threshold
            if chunk.cosine_similarity >= thresh:
                retrieved_chunks.append(chunk)

        return retrieved_chunks
