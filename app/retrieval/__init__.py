"""Retrieval package: Top-K similarity search and ranking."""

from app.retrieval.models import RetrievalResult
from app.retrieval.retriever import (
    RetrievedChunk,
    Retriever,
    RetrieverError,
    EmptyQueryError,
    InvalidTopKError,
    InvalidSimilarityThresholdError,
    InvalidDocumentFilterError,
)

__all__ = [
    "RetrievalResult",
    "RetrievedChunk",
    "Retriever",
    "RetrieverError",
    "EmptyQueryError",
    "InvalidTopKError",
    "InvalidSimilarityThresholdError",
    "InvalidDocumentFilterError",
]
