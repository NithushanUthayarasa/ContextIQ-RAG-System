"""
ContextIQ - Retrieval Result Models
Defines explicit data models for retrieved document chunks, distances, and similarity metrics.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalResult:
    """
    Represents a retrieved document chunk with similarity distance, BM25 score, and fusion ranks.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        text: Text content of the chunk.
        source: Filename or document title.
        page_number: 1-indexed page number where the chunk originates.
        chunk_index: Zero-indexed position of chunk within the page.
        distance: Raw distance score returned by the vector store (e.g., Cosine Distance). None for BM25-only chunks.
        document_id: Optional unique SHA-256 identifier of the source document.
        bm25_score: Raw BM25 score if retrieved via keyword search.
        semantic_rank: 1-based rank in semantic candidate pool if retrieved via vector search.
        bm25_rank: 1-based rank in BM25 candidate pool if retrieved via keyword search.
        rrf_score: Reciprocal Rank Fusion score if retrieved via hybrid search.
        retrieval_method: Retrieval mechanism that identified this chunk ("semantic", "bm25", "hybrid").
    """
    chunk_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int
    distance: Optional[float] = None
    document_id: Optional[str] = None
    bm25_score: Optional[float] = None
    semantic_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    retrieval_method: Optional[str] = None
    # Reranking metadata — populated by a Reranker; never set by the Retriever.
    rerank_score: Optional[float] = None
    original_rank: Optional[int] = None

    @property
    def cosine_similarity(self) -> Optional[float]:
        """
        Calculates cosine similarity from cosine distance:
        cosine_similarity = 1.0 - cosine_distance.

        Returns None if distance is None (e.g. BM25-only retrieval result).
        """
        if self.distance is None:
            return None
        return 1.0 - self.distance

    @property
    def similarity(self) -> Optional[float]:
        """
        Derived similarity metric under cosine distance space.
        Maintains clear separation between raw distance and derived similarity.
        """
        return self.cosine_similarity


# Alias for backward compatibility across existing code and test suites
RetrievedChunk = RetrievalResult
