"""Retrieval package: Top-K similarity search, BM25 keyword search, hybrid ranking, and reranking."""

from app.retrieval.bm25 import BM25Index, BM25Result, tokenize
from app.retrieval.models import RetrievalResult, RetrievedChunk
from app.retrieval.reranker import BaseReranker, TFIDFReranker, RerankerError, InvalidTopKError as RerankerInvalidTopKError
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
    "BaseReranker",
    "TFIDFReranker",
    "RerankerError",
    "RerankerInvalidTopKError",
    "Retriever",
    "RetrieverError",
    "EmptyQueryError",
    "InvalidTopKError",
    "InvalidSimilarityThresholdError",
    "InvalidDocumentFilterError",
    "InvalidRetrievalModeError",
    "VALID_RETRIEVAL_MODES",
]
