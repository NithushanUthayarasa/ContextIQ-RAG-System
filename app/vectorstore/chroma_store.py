"""
ContextIQ - ChromaDB Vector Store Module
Manages persistent local vector storage with cosine distance metric using ChromaDB.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import chromadb

from app.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, EMBEDDING_DIMENSION
from app.ingestion.chunker import DocumentChunk


class VectorStoreError(Exception):
    """Base exception for ChromaVectorStore errors."""
    pass


class EmptyVectorStoreInputError(VectorStoreError, ValueError):
    """Raised when empty chunks or embeddings are passed for insertion."""
    pass


class MismatchedInputError(VectorStoreError, ValueError):
    """Raised when the number of chunks does not match the number of embeddings."""
    pass


class InvalidEmbeddingError(VectorStoreError, ValueError):
    """Raised when an embedding is malformed or has an unexpected dimension."""
    pass


class ChromaVectorStore:
    """
    Persistent ChromaDB vector store configured with cosine similarity for document chunks.
    """

    def __init__(
        self,
        persist_dir: Optional[Union[str, Path]] = None,
        collection_name: Optional[str] = None,
        expected_dimension: Optional[int] = None,
    ):
        self.persist_dir = Path(persist_dir or CHROMA_PERSIST_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        self.expected_dimension = (
            expected_dimension if expected_dimension is not None else EMBEDDING_DIMENSION
        )

        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self._get_or_create_collection()
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB client: {str(e)}") from e

    def _get_or_create_collection(self):
        """
        Creates or retrieves the ChromaDB collection explicitly configured for cosine distance.
        """
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Upserts document chunks and their corresponding embedding vectors into ChromaDB.

        Args:
            chunks: List of DocumentChunk objects.
            embeddings: List of embedding vectors (float lists).

        Returns:
            The number of chunks added/updated.

        Raises:
            EmptyVectorStoreInputError: If chunks or embeddings are empty.
            MismatchedInputError: If chunk and embedding counts differ.
            InvalidEmbeddingError: If an embedding has incorrect dimension or invalid types.
        """
        if not chunks or not embeddings:
            raise EmptyVectorStoreInputError("Chunks and embeddings must not be empty.")

        if len(chunks) != len(embeddings):
            raise MismatchedInputError(
                f"Number of chunks ({len(chunks)}) does not match number of embeddings ({len(embeddings)})."
            )

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        clean_embeddings: List[List[float]] = []

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            if not isinstance(chunk, DocumentChunk):
                raise VectorStoreError(
                    f"Item at index {idx} is not an instance of DocumentChunk."
                )

            if not chunk.chunk_id or not chunk.chunk_id.strip():
                raise VectorStoreError(f"Chunk at index {idx} has an invalid or empty chunk_id.")

            if not emb or not isinstance(emb, (list, tuple)):
                raise InvalidEmbeddingError(
                    f"Embedding at index {idx} must be a non-empty sequence of floats."
                )

            if self.expected_dimension is not None and len(emb) != self.expected_dimension:
                raise InvalidEmbeddingError(
                    f"Embedding dimension {len(emb)} at index {idx} does not match expected dimension {self.expected_dimension}."
                )

            try:
                float_emb = [float(v) for v in emb]
            except (ValueError, TypeError) as e:
                raise InvalidEmbeddingError(
                    f"Embedding at index {idx} contains non-numeric values."
                ) from e

            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "source": chunk.source,
                    "page_number": int(chunk.page_number),
                    "chunk_index": int(chunk.chunk_index),
                }
            )
            clean_embeddings.append(float_emb)

        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=clean_embeddings,
                metadatas=metadatas,
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert chunks into ChromaDB: {str(e)}") from e

        return len(ids)

    def count(self) -> int:
        """Returns the total number of document chunks currently stored in the collection."""
        return self.collection.count()

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fetches raw documents and metadata from the collection."""
        kwargs: Dict[str, Any] = {}
        if ids is not None:
            kwargs["ids"] = ids
        if where is not None:
            kwargs["where"] = where
        return self.collection.get(**kwargs)

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Queries the vector store for the nearest chunks using cosine distance.

        Args:
            query_embedding: Dense float vector of the query.
            top_k: Number of nearest chunks to retrieve.
            where: Optional metadata filter.

        Returns:
            ChromaDB query response dictionary containing ids, documents, metadatas, distances.
        """
        if self.count() == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        actual_k = min(top_k, self.count())
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": actual_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where

        try:
            return self.collection.query(**kwargs)
        except Exception as e:
            raise VectorStoreError(f"ChromaDB similarity query failed: {str(e)}") from e

    def delete_by_source(self, source: str) -> None:
        """Deletes all chunks associated with a specific document source filename."""
        self.collection.delete(where={"source": source})

    def reset(self) -> None:
        """
        Clears the collection completely and re-initializes an empty collection.
        Useful for re-indexing documents or testing.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self._get_or_create_collection()

    @property
    def stats(self) -> Dict[str, Any]:
        """Provides high-level statistics about the vector store."""
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.count(),
            "persist_dir": str(self.persist_dir),
            "distance_metric": "cosine",
            "expected_dimension": self.expected_dimension,
        }
