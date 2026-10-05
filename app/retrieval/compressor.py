"""
ContextIQ - Context Compression Module

Provides deterministic, local, extractive context compression to condense retrieved
chunks down to their most query-relevant sentences before passing them to the generator.

Architecture position:
    Retriever → Reranker → Parent Resolution → Compressor → Generator

Key properties:
- Extractive: Selects verbatim sentences strictly from retrieved chunks rather than generating new text.
- Deterministic: Zero external API calls, pure Python token scoring, 100% reproducible.
- Preserves Order: Selected sentences are re-sorted by their original chronological
  occurrence to preserve narrative and syntactic coherence.
- Safe Fallback: If no sentences pass the relevance threshold, or text splitting yields
  empty candidates, the original chunk text is preserved intact.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import replace
import math
import re
from typing import List, Optional, Sequence, Tuple

from app.config import (
    COMPRESSION_MAX_SENTENCES,
    COMPRESSION_MIN_SENTENCE_LEN,
    COMPRESSION_SIMILARITY_THRESHOLD,
)
from app.retrieval.bm25 import tokenize
from app.retrieval.models import RetrievedChunk


class ContextCompressorError(Exception):
    """Base exception for context compression failures."""


class InvalidCompressorConfigError(ContextCompressorError, ValueError):
    """Raised when compressor configuration parameters are invalid."""


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BaseContextCompressor(ABC):
    """
    Abstract context compressor interface.

    Subclasses must implement ``compress``. The contract:
    - Accept a query string and a list of RetrievedChunk objects.
    - Return a list of RetrievedChunk objects with compressed text.
    - Populate ``original_text``, ``compression_ratio``, ``sentences_kept``,
      and ``sentences_total`` on returned chunks.
    - Preserve all original provenance and retrieval metadata unchanged.
    - Never call external generative APIs.
    """

    @abstractmethod
    def compress(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        Compresses ``chunks`` with respect to ``query``.

        Args:
            query: The retrieval query string.
            chunks: List of retrieved candidate chunks.

        Returns:
            List of compressed RetrievedChunk objects.
        """


# ---------------------------------------------------------------------------
# Sentence tokenization and scoring utilities
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "can", "could", "should", "would",
    "will", "shall", "may", "might", "must", "it", "its", "this", "that",
    "these", "those", "what", "which", "who", "whom", "whose", "when",
    "where", "why", "how",
}


def _split_into_sentences(text: str) -> List[str]:
    """
    Splits text into discrete, non-empty sentences.
    Preserves sentence content and strips trailing whitespace.
    """
    if not text or not isinstance(text, str):
        return []

    raw = _SENTENCE_SPLIT_PATTERN.split(text.strip())
    sentences = [s.strip() for s in raw if s and s.strip()]
    if not sentences and text.strip():
        sentences = [text.strip()]
    return sentences


def _score_sentence(query_tokens: List[str], sentence_tokens: List[str]) -> float:
    """
    Computes lexical cosine similarity between query and sentence tokens.
    Filters out common stop words if content tokens are present in the query.
    Returns a float score in [0.0, 1.0].
    """
    if not query_tokens or not sentence_tokens:
        return 0.0

    content_query = [t for t in query_tokens if t not in _STOP_WORDS]
    active_query = content_query if content_query else query_tokens

    if content_query:
        active_sent = [t for t in sentence_tokens if t not in _STOP_WORDS]
    else:
        active_sent = sentence_tokens

    if not active_query or not active_sent:
        return 0.0

    q_counts = Counter(active_query)
    s_counts = Counter(active_sent)

    dot = sum(q_counts[t] * s_counts[t] for t in q_counts if t in s_counts)
    if dot == 0:
        return 0.0

    q_norm = math.sqrt(sum(c * c for c in q_counts.values()))
    s_norm = math.sqrt(sum(c * c for c in s_counts.values()))

    if q_norm == 0.0 or s_norm == 0.0:
        return 0.0

    return float(dot / (q_norm * s_norm))


# ---------------------------------------------------------------------------
# Extractive Context Compressor Implementation
# ---------------------------------------------------------------------------

class ExtractiveContextCompressor(BaseContextCompressor):
    """
    Deterministic, extractive context compressor.

    Identifies and extracts the most query-relevant sentences from each chunk,
    omitting irrelevant background sentences while preserving document order.
    """

    def __init__(
        self,
        max_sentences: int = COMPRESSION_MAX_SENTENCES,
        similarity_threshold: float = COMPRESSION_SIMILARITY_THRESHOLD,
        min_sentence_len: int = COMPRESSION_MIN_SENTENCE_LEN,
    ):
        if not isinstance(max_sentences, int) or isinstance(max_sentences, bool) or max_sentences < 1:
            raise InvalidCompressorConfigError(
                f"max_sentences must be an integer >= 1, got {max_sentences!r}"
            )
        if (
            not isinstance(similarity_threshold, (int, float))
            or isinstance(similarity_threshold, bool)
            or not (0.0 <= similarity_threshold <= 1.0)
        ):
            raise InvalidCompressorConfigError(
                f"similarity_threshold must be a float in [0.0, 1.0], got {similarity_threshold!r}"
            )
        if not isinstance(min_sentence_len, int) or isinstance(min_sentence_len, bool) or min_sentence_len < 1:
            raise InvalidCompressorConfigError(
                f"min_sentence_len must be an integer >= 1, got {min_sentence_len!r}"
            )

        self.max_sentences = max_sentences
        self.similarity_threshold = float(similarity_threshold)
        self.min_sentence_len = min_sentence_len

    def compress(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        max_sentences: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """
        Compresses each chunk in ``chunks`` by selecting sentences matching ``query``.

        Args:
            query: The query text used to score sentence relevance.
            chunks: Retrieved chunks to be compressed.
            max_sentences: Optional per-call override for maximum sentences per chunk.
            similarity_threshold: Optional per-call override for minimum similarity score.

        Returns:
            List of compressed RetrievedChunk objects with updated metadata.
        """
        if not chunks:
            return []

        eff_max_sentences = max_sentences if max_sentences is not None else self.max_sentences
        eff_threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold

        if not isinstance(eff_max_sentences, int) or isinstance(eff_max_sentences, bool) or eff_max_sentences < 1:
            raise InvalidCompressorConfigError(
                f"max_sentences must be an integer >= 1, got {eff_max_sentences!r}"
            )
        if (
            not isinstance(eff_threshold, (int, float))
            or isinstance(eff_threshold, bool)
            or not (0.0 <= eff_threshold <= 1.0)
        ):
            raise InvalidCompressorConfigError(
                f"similarity_threshold must be a float in [0.0, 1.0], got {eff_threshold!r}"
            )

        query_tokens = tokenize(query.strip()) if query and query.strip() else []

        compressed_chunks: List[RetrievedChunk] = []

        for chunk in chunks:
            if not chunk.text or not chunk.text.strip():
                compressed_chunks.append(chunk)
                continue

            original_text = chunk.text
            orig_len = len(original_text)
            sentences = _split_into_sentences(original_text)
            total_sentences = len(sentences)

            # If no query tokens or cannot split into sentences, fallback to original text
            if not query_tokens or total_sentences == 0:
                compressed_chunks.append(
                    replace(
                        chunk,
                        original_text=original_text,
                        compression_ratio=1.0,
                        sentences_kept=total_sentences,
                        sentences_total=total_sentences,
                    )
                )
                continue

            # Score each sentence
            scored_sentences: List[Tuple[int, str, float]] = []
            for idx, sent in enumerate(sentences):
                if len(sent.strip()) < self.min_sentence_len:
                    score = 0.0
                else:
                    score = _score_sentence(query_tokens, tokenize(sent))
                scored_sentences.append((idx, sent, score))

            # Filter sentences meeting the threshold
            passing = [item for item in scored_sentences if item[2] >= eff_threshold]

            # SAFE FALLBACK: If no sentences pass threshold, keep original chunk text
            if not passing:
                compressed_chunks.append(
                    replace(
                        chunk,
                        original_text=original_text,
                        compression_ratio=1.0,
                        sentences_kept=total_sentences,
                        sentences_total=total_sentences,
                    )
                )
                continue

            # If more than max_sentences qualify, select top by relevance score
            if len(passing) > eff_max_sentences:
                # Sort descending by score, tie-breaker ascending by index
                passing.sort(key=lambda item: (-item[2], item[0]))
                selected = passing[:eff_max_sentences]
            else:
                selected = passing

            # Re-sort selected sentences by original chronological index to preserve fluency
            selected.sort(key=lambda item: item[0])

            selected_texts = [item[1] for item in selected]
            compressed_text = " ".join(selected_texts).strip()

            # Safeguard: if joined text is empty, fallback to original text
            if not compressed_text:
                compressed_chunks.append(
                    replace(
                        chunk,
                        original_text=original_text,
                        compression_ratio=1.0,
                        sentences_kept=total_sentences,
                        sentences_total=total_sentences,
                    )
                )
                continue

            ratio = round(len(compressed_text) / orig_len, 4) if orig_len > 0 else 1.0

            compressed_chunk = replace(
                chunk,
                text=compressed_text,
                original_text=original_text,
                compression_ratio=ratio,
                sentences_kept=len(selected),
                sentences_total=total_sentences,
            )
            compressed_chunks.append(compressed_chunk)

        return compressed_chunks
