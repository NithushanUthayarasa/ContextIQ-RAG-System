"""
ContextIQ - Retrieval Result Models
Defines explicit data models for retrieved document chunks, distances, and similarity metrics.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalResult:
    """
    Represents a retrieved document chunk with similarity distance and metadata.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        text: Text content of the chunk.
        source: Filename or document title.
        page_number: 1-indexed page number where the chunk originates.
        chunk_index: Zero-indexed position of chunk within the page.
        distance: Raw distance score returned by the vector store (e.g., Cosine Distance).
        document_id: Optional unique SHA-256 identifier of the source document.
    """
    chunk_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int
    distance: float
    document_id: Optional[str] = None

    @property
    def cosine_similarity(self) -> float:
        """
        Calculates cosine similarity from cosine distance:
        cosine_similarity = 1.0 - cosine_distance.

        Valid for Chroma collections configured with cosine space (hnsw:space = 'cosine').
        """
        return 1.0 - self.distance

    @property
    def similarity(self) -> float:
        """
        Derived similarity metric under cosine distance space.
        Maintains clear separation between raw distance and derived similarity.
        """
        return self.cosine_similarity


# Alias for backward compatibility across existing code and test suites
RetrievedChunk = RetrievalResult
