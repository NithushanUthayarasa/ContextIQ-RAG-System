"""
ContextIQ - BM25 Keyword Search Module
Implements a lightweight, pure-Python Okapi BM25 index with deterministic tokenization
and metadata-aware document filtering.
"""

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Union

_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b")


def tokenize(text: str) -> List[str]:
    """
    Extracts lowercased alphanumeric and hyphenated tokens from input text.
    Also splits hyphenated terms so that sub-words can match individually.

    Examples:
        tokenize("Retrieval-Augmented Generation (RAG) 2.0!")
        -> ["retrieval-augmented", "retrieval", "augmented", "generation", "rag", "2", "0"]
    """
    if not text or not isinstance(text, str):
        return []
    tokens: List[str] = []
    for match in _TOKEN_PATTERN.finditer(text):
        token = match.group(0).lower()
        tokens.append(token)
        if "-" in token:
            for part in token.split("-"):
                if part and part != token:
                    tokens.append(part)
    return tokens


@dataclass
class BM25Result:
    """Represents a scored chunk returned from BM25 keyword search."""
    chunk_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int
    score: float
    document_id: Optional[str] = None
    parent_id: Optional[str] = None
    parent_index: Optional[int] = None


@dataclass
class _IndexedDocument:
    """Internal indexed representation of a single document chunk."""
    chunk_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int
    document_id: Optional[str]
    length: int
    term_frequencies: Dict[str, int]
    parent_id: Optional[str] = None
    parent_index: Optional[int] = None


class BM25Index:
    """
    In-memory Okapi BM25 search index for document chunks.

    Parameters:
        k1: BM25 term frequency saturation parameter (default 1.5).
        b: Document length normalization parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, _IndexedDocument] = {}
        self.doc_frequencies: Dict[str, int] = Counter()
        self.total_docs: int = 0
        self.total_length: int = 0
        self.avg_doc_len: float = 0.0

    @property
    def chunk_count(self) -> int:
        """Returns the total number of indexed document chunks."""
        return self.total_docs

    def clear(self) -> None:
        """Resets the index completely."""
        self.documents.clear()
        self.doc_frequencies.clear()
        self.total_docs = 0
        self.total_length = 0
        self.avg_doc_len = 0.0

    def add_records(self, records: Sequence[Union[Dict[str, Any], Any]]) -> int:
        """
        Indexes a collection of chunk records (dictionaries or DocumentChunk instances).
        Rebuilds corpus-level statistics and IDF tables.

        Returns:
            The number of chunks indexed.
        """
        self.clear()
        if not records:
            return 0

        for item in records:
            if isinstance(item, dict):
                chunk_id = str(item.get("chunk_id", ""))
                text = str(item.get("text", ""))
                meta = item.get("metadata", {}) or {}
                source = str(meta.get("source", "unknown"))
                page_number = int(meta.get("page_number", 1))
                chunk_index = int(meta.get("chunk_index", 0))
                doc_id = meta.get("document_id")
                parent_id = meta.get("parent_id")
                parent_index = int(meta["parent_index"]) if meta.get("parent_index") is not None else None
            elif hasattr(item, "chunk_id"):
                chunk_id = str(item.chunk_id)
                text = str(getattr(item, "text", ""))
                source = str(getattr(item, "source", "unknown"))
                page_number = int(getattr(item, "page_number", 1))
                chunk_index = int(getattr(item, "chunk_index", 0))
                doc_id = getattr(item, "document_id", None)
                parent_id = getattr(item, "parent_id", None)
                parent_index = getattr(item, "parent_index", None)
            else:
                continue

            if not chunk_id:
                continue

            tokens = tokenize(text)
            term_freqs = Counter(tokens)
            doc_len = len(tokens)

            doc = _IndexedDocument(
                chunk_id=chunk_id,
                text=text,
                source=source,
                page_number=page_number,
                chunk_index=chunk_index,
                document_id=doc_id,
                length=doc_len,
                term_frequencies=term_freqs,
                parent_id=parent_id,
                parent_index=parent_index,
            )
            self.documents[chunk_id] = doc
            self.total_length += doc_len

            for term in term_freqs.keys():
                self.doc_frequencies[term] += 1

        self.total_docs = len(self.documents)
        self.avg_doc_len = (
            (self.total_length / self.total_docs) if self.total_docs > 0 else 0.0
        )
        return self.total_docs

    def _idf(self, term: str) -> float:
        """
        Computes non-negative Robertson-Spärck Jones IDF:
        IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
        Always strictly positive for any n(q) <= N.
        """
        n_q = self.doc_frequencies.get(term, 0)
        return math.log(1.0 + (self.total_docs - n_q + 0.5) / (n_q + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[Sequence[str]] = None,
    ) -> List[BM25Result]:
        """
        Scores indexed chunks against query terms using the Okapi BM25 formula.

        Args:
            query: User search query or question.
            top_k: Maximum number of ranked results to return.
            document_ids: Optional document identifier filter.

        Returns:
            List of BM25Result objects sorted descending by score (> 0).
        """
        if not query or not query.strip() or self.total_docs == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        filter_set = set(document_ids) if document_ids else None
        scored_results: List[BM25Result] = []

        for chunk_id, doc in self.documents.items():
            if filter_set is not None:
                if doc.document_id not in filter_set:
                    continue

            score = 0.0
            doc_len = doc.length
            len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len) if self.avg_doc_len > 0 else 1.0

            for term in query_tokens:
                freq = doc.term_frequencies.get(term, 0)
                if freq == 0:
                    continue

                idf_val = self._idf(term)
                # Okapi BM25 TF saturation formula
                tf_component = (freq * (self.k1 + 1.0)) / (freq + self.k1 * len_norm)
                score += idf_val * tf_component

            if score > 0.0:
                scored_results.append(
                    BM25Result(
                        chunk_id=doc.chunk_id,
                        text=doc.text,
                        source=doc.source,
                        page_number=doc.page_number,
                        chunk_index=doc.chunk_index,
                        score=score,
                        document_id=doc.document_id,
                        parent_id=doc.parent_id,
                        parent_index=doc.parent_index,
                    )
                )

        # Deterministic sorting: score descending, then chunk_id ascending
        scored_results.sort(key=lambda r: (-r.score, r.chunk_id))
        return scored_results[:top_k]
