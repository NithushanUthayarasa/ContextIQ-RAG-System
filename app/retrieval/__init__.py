"""Retrieval package: Top-K similarity search, BM25 keyword search, and hybrid ranking."""

from app.retrieval.bm25 import BM25Index, BM25Result, tokenize
from app.retrieval.models import RetrievalResult, RetrievedChunk
from app.retrieval.retriever import (
    Retriever,
    RetrieverError,
    EmptyQueryError,
    InvalidTopKError,
    InvalidSimilarityThresholdError,
    InvalidDocumentFilterError,
    InvalidRetrievalModeError,
    VALID_RETRIEVAL_MODES,
)

__all__ = [
    "RetrievalResult",
    "RetrievedChunk",
    "BM25Index",
    "BM25Result",
    "tokenize",
    "Retriever",
    "RetrieverError",
    "EmptyQueryError",
    "InvalidTopKError",
    "InvalidSimilarityThresholdError",
    "InvalidDocumentFilterError",
    "InvalidRetrievalModeError",
    "VALID_RETRIEVAL_MODES",
]
