"""
ContextIQ - Reranker Module

Provides a clean abstraction for reranking retrieved candidate chunks based on
deeper query-document relevance AFTER initial retrieval.

Architecture position:
    Retriever → Reranker → Generator

The Reranker's job is to reorder a candidate pool and return the top-K most
relevant chunks without modifying their original retrieval metadata (distance,
bm25_score, rrf_score, etc.).

Available implementations
--------------------------
TFIDFReranker (default)
    A deterministic, dependency-free baseline that scores query-document
    relevance using TF-IDF term overlap between the query and each candidate.
    No API calls, no GPU, fully reproducible.

    Limitation: TF-IDF overlap does not capture semantic similarity.
    A cross-encoder (e.g. ms-marco-MiniLM) will outperform it on paraphrase
    queries.  The BaseReranker interface makes swapping straightforward.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import replace
from typing import List, Optional

from app.retrieval.models import RetrievedChunk


class RerankerError(Exception):
    """Base exception for reranker failures."""


class InvalidTopKError(RerankerError, ValueError):
    """Raised when top_k is invalid (<= 0 or not an integer)."""


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BaseReranker(ABC):
    """
    Abstract reranker interface.

    Subclasses must implement ``rerank``.  The contract:
    - Accept a query string and a list of candidate RetrievedChunk objects.
    - Return at most ``top_k`` chunks, ordered by descending relevance.
    - Preserve ALL original retrieval metadata (distance, bm25_score, …).
    - Set ``rerank_score`` and ``original_rank`` on returned chunks.
    - Never call external generative APIs.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """
        Reranks ``candidates`` with respect to ``query``.

        Args:
            query: The retrieval query (typically the rewritten standalone query).
            candidates: Candidate chunks from the retriever (may be larger than top_k).
            top_k: Maximum number of chunks to return.

        Returns:
            List of at most top_k RetrievedChunk objects ordered by rerank_score
            descending.  Original metadata is preserved unchanged; rerank_score
            and original_rank are added.

        Raises:
            InvalidTopKError: If top_k <= 0.
        """

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise InvalidTopKError(
                f"top_k must be a positive integer, got: {top_k!r}"
            )


# ---------------------------------------------------------------------------
# TF-IDF term-overlap baseline reranker
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercased alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def _tfidf_score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """
    Lightweight TF-IDF query-document relevance score.

    For each unique query token present in the document, computes:
        tf  = (count in doc) / (total doc tokens)
        idf = log(1 + 1/query_term_frequency_in_query)  [smoothed]
        contribution = tf * idf

    Returns the sum over all matching query tokens.  Result is in [0, +inf).
    """
    if not query_tokens or not doc_tokens:
        return 0.0

    doc_len = len(doc_tokens)
    doc_tf = Counter(doc_tokens)
    query_tf = Counter(query_tokens)

    score = 0.0
    for term, q_count in query_tf.items():
        if term in doc_tf:
            tf = doc_tf[term] / doc_len
            # Smoothed IDF using query term frequency (purely local signal)
            idf = math.log(1.0 + 1.0 / q_count)
            score += tf * idf

    return score


class TFIDFReranker(BaseReranker):
    """
    Deterministic TF-IDF term-overlap reranker.

    Scores each candidate chunk by the weighted term overlap between the
    query tokens and the chunk tokens.  No external dependencies, no API
    calls, fully reproducible.

    Suitable as a baseline for development and testing.  For production
    quality, replace with a cross-encoder (e.g. sentence-transformers
    cross-encoder/ms-marco-MiniLM-L-6-v2) via a subclass of BaseReranker.
    """

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """
        Scores and reorders candidates by TF-IDF term overlap.

        Duplicate chunk_ids are deduplicated before scoring (first occurrence
        wins to preserve original retrieval ordering when scores are equal).
        """
        self._validate_top_k(top_k)

        if not candidates:
            return []

        # Deduplicate by chunk_id, keeping first occurrence
        seen_ids: set = set()
        unique_candidates: List[RetrievedChunk] = []
        for chunk in candidates:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_candidates.append(chunk)

        query_tokens = _tokenize(query)

        scored: List[tuple] = []
        for original_rank, chunk in enumerate(unique_candidates, start=1):
            score = _tfidf_score(query_tokens, _tokenize(chunk.text))
            scored.append((score, original_rank, chunk))

        # Sort: descending score, then ascending original_rank as tie-breaker
        scored.sort(key=lambda t: (-t[0], t[1]))

        result: List[RetrievedChunk] = []
        for score, original_rank, chunk in scored[:top_k]:
            # Use dataclasses.replace so we never mutate the input objects
            reranked_chunk = replace(
                chunk,
                rerank_score=score,
                original_rank=original_rank,
            )
            result.append(reranked_chunk)

        return result
