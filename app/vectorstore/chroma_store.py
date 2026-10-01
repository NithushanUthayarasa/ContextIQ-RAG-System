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
            meta: Dict[str, Any] = {
                "source": chunk.source,
                "page_number": int(chunk.page_number),
                "chunk_index": int(chunk.chunk_index),
            }
            if chunk.document_id is not None:
                meta["document_id"] = chunk.document_id
            metadatas.append(meta)
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

    def has_document(self, document_id: str) -> bool:
        """
        Checks whether at least one chunk exists with the given document_id.

        Args:
            document_id: Unique content hash of the document to check.

        Returns:
            True if the document exists in the store, False otherwise.
        """
        if not document_id or not isinstance(document_id, str) or not document_id.strip():
            return False

        clean_doc_id = document_id.strip()
        try:
            res = self.collection.get(where={"document_id": clean_doc_id}, limit=1)
            ids = res.get("ids", [])
            return len(ids) > 0
        except Exception as e:
            raise VectorStoreError(f"Failed to check document existence: {str(e)}") from e

    def delete_by_document_id(self, document_id: str) -> int:
        """
        Deletes all chunks belonging to a specific document_id.

        Args:
            document_id: Unique content hash of the document to delete.

        Returns:
            The number of chunks deleted.
        """
        if not document_id or not isinstance(document_id, str) or not document_id.strip():
            return 0

        clean_doc_id = document_id.strip()
        try:
            matching = self.collection.get(where={"document_id": clean_doc_id})
            chunk_ids = matching.get("ids", [])
            if not chunk_ids:
                return 0

            self.collection.delete(where={"document_id": clean_doc_id})
            return len(chunk_ids)
        except Exception as e:
            raise VectorStoreError(f"Failed to delete document chunks: {str(e)}") from e

    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        """
        Returns document-level summary information derived from ChromaDB metadata.
        Groups by unique document_id.

        Returns:
            List of dictionaries with keys:
            - document_id: Unique document identifier.
            - source: Original filename or document label.
            - page_count: Total distinct page numbers in this document.
            - chunk_count: Total number of chunks indexed for this document.
        """
        if self.count() == 0:
            return []

        try:
            data = self.collection.get(include=["metadatas"])
        except Exception as e:
            raise VectorStoreError(f"Failed to retrieve metadata from ChromaDB: {str(e)}") from e

        metadatas = data.get("metadatas", [])
        if not metadatas:
            return []

        docs_map: Dict[str, Dict[str, Any]] = {}

        for meta in metadatas:
            if not meta:
                continue

            doc_id = meta.get("document_id")
            source = meta.get("source", "unknown")
            if not doc_id:
                # Graceful fallback for legacy V1 records lacking document_id
                doc_id = f"legacy_{source}"

            page_num = meta.get("page_number")

            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "document_id": doc_id,
                    "source": source,
                    "pages": set(),
                    "chunk_count": 0,
                }

            if page_num is not None:
                docs_map[doc_id]["pages"].add(page_num)
            docs_map[doc_id]["chunk_count"] += 1

        result = [
            {
                "document_id": doc_id,
                "source": info["source"],
                "page_count": len(info["pages"]),
                "chunk_count": info["chunk_count"],
            }
            for doc_id, info in docs_map.items()
        ]

        # Deterministic stable ordering: source + document_id
        result.sort(key=lambda d: (d["source"], d["document_id"]))
        return result


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
