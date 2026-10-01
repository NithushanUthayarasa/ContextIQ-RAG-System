"""
ContextIQ - Document Retriever Module
Coordinates query vectorization, similarity search against ChromaDB, and formatting of retrieved chunks.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.config import DEFAULT_TOP_K
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


class Retriever:
    """
    Coordinates semantic retrieval using an injected GeminiEmbedder and ChromaVectorStore.
    """

    def __init__(
        self,
        embedder: GeminiEmbedder,
        vector_store: ChromaVectorStore,
        default_top_k: int = DEFAULT_TOP_K,
    ):
        if embedder is None:
            raise RetrieverError("An embedder instance must be provided to Retriever.")
        if vector_store is None:
            raise RetrieverError("A vector_store instance must be provided to Retriever.")

        self.embedder = embedder
        self.vector_store = vector_store
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """
        Embeds a user query, queries ChromaDB, and returns the top_k most relevant chunks.

        Args:
            query: The user's search query or question.
            top_k: Maximum number of chunks to return (defaults to configured top_k).
            where: Optional metadata filter dictionary.

        Returns:
            List of RetrievedChunk objects ordered by cosine distance (nearest first).

        Raises:
            EmptyQueryError: If query is empty or whitespace-only.
            InvalidTopKError: If top_k is <= 0 or not an integer.
            RetrieverError: If embedding or vector store query fails.
        """
        if not query or not query.strip():
            raise EmptyQueryError("Retrieval query must not be empty.")

        cleaned_query = query.strip()

        k = top_k if top_k is not None else self.default_top_k
        if not isinstance(k, int) or k <= 0:
            raise InvalidTopKError(f"top_k must be a positive integer, got: {top_k}")

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
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=doc,
                    source=meta_dict.get("source", "unknown"),
                    page_number=int(meta_dict.get("page_number", 1)),
                    chunk_index=int(meta_dict.get("chunk_index", 0)),
                    distance=float(dist),
                    document_id=meta_dict.get("document_id"),
                )
            )

        return retrieved_chunks
