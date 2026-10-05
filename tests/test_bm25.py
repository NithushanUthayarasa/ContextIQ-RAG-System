"""
Tests for Okapi BM25 Keyword Search Index (app/retrieval/bm25.py).
Verifies deterministic tokenization, Okapi scoring, ranking properties, and metadata filtering.
"""

import pytest
from app.retrieval.bm25 import BM25Index, BM25Result, tokenize


class TestTokenize:
    """Tests the deterministic tokenizer used by BM25."""

    def test_tokenize_standard_sentence(self):
        text = "Retrieval-Augmented Generation improves LLM accuracy."
        tokens = tokenize(text)
        assert tokens == [
            "retrieval-augmented",
            "retrieval",
            "augmented",
            "generation",
            "improves",
            "llm",
            "accuracy",
        ]

    def test_tokenize_preserves_numbers_and_hyphens(self):
        text = "Model GPT-4 scored 98.5% on RAG-2024 benchmark."
        tokens = tokenize(text)
        assert "gpt-4" in tokens
        assert "rag-2024" in tokens
        assert "98" in tokens
        assert "5" in tokens

    def test_tokenize_empty_and_whitespace(self):
        assert tokenize("") == []
        assert tokenize("   \t\n  ") == []
        assert tokenize(None) == []  # type: ignore

    def test_tokenize_punctuation_only(self):
        assert tokenize("!@#$%^&*()_+=~`{}[]:;'<>,.?/") == []


class TestBM25Index:
    """Tests BM25Index indexing, scoring, ranking, and filtering."""

    @pytest.fixture
    def sample_records(self):
        return [
            {
                "chunk_id": "c1",
                "text": "Deep Learning models require large amounts of computational power and GPU hardware.",
                "metadata": {
                    "source": "deep_learning.pdf",
                    "page_number": 1,
                    "chunk_index": 0,
                    "document_id": "doc_dl",
                },
            },
            {
                "chunk_id": "c2",
                "text": "Retrieval-Augmented Generation combines semantic vector search with large language models.",
                "metadata": {
                    "source": "rag_paper.pdf",
                    "page_number": 2,
                    "chunk_index": 0,
                    "document_id": "doc_rag",
                },
            },
            {
                "chunk_id": "c3",
                "text": "Vector databases like ChromaDB store embeddings for similarity search in RAG pipelines.",
                "metadata": {
                    "source": "vector_dbs.pdf",
                    "page_number": 5,
                    "chunk_index": 1,
                    "document_id": "doc_chroma",
                },
            },
            {
                "chunk_id": "c4",
                "text": "Okapi BM25 is a probabilistic keyword ranking function widely used in information retrieval.",
                "metadata": {
                    "source": "ir_textbook.pdf",
                    "page_number": 12,
                    "chunk_index": 3,
                    "document_id": "doc_ir",
                },
            },
        ]

    def test_empty_index_returns_empty(self):
        index = BM25Index()
        assert index.chunk_count == 0
        results = index.search("retrieval")
        assert results == []

    def test_empty_query_returns_empty(self, sample_records):
        index = BM25Index()
        index.add_records(sample_records)
        assert index.search("") == []
        assert index.search("   ") == []

    def test_query_no_match_returns_empty(self, sample_records):
        index = BM25Index()
        index.add_records(sample_records)
        results = index.search("quantum thermodynamics photosynthesis")
        assert results == []

    def test_exact_keyword_retrieval(self, sample_records):
        index = BM25Index()
        index.add_records(sample_records)
        results = index.search("BM25 ranking")
        assert len(results) > 0
        assert results[0].chunk_id == "c4"
        assert results[0].score > 0.0
        assert results[0].source == "ir_textbook.pdf"
        assert results[0].page_number == 12

    def test_term_frequency_saturation(self):
        index = BM25Index()
        records = [
            {
                "chunk_id": "single",
                "text": "The transformer architecture is powerful.",
                "metadata": {"source": "a.pdf", "page_number": 1, "chunk_index": 0},
            },
            {
                "chunk_id": "repeated",
                "text": "The transformer architecture is a transformer transformer transformer.",
                "metadata": {"source": "b.pdf", "page_number": 1, "chunk_index": 0},
            },
        ]
        index.add_records(records)
        results = index.search("transformer")
        assert len(results) == 2
        # 'repeated' has higher term frequency, so it should score higher than 'single'
        assert results[0].chunk_id == "repeated"
        assert results[1].chunk_id == "single"
        assert results[0].score > results[1].score

    def test_document_length_normalization(self):
        """Shorter document containing the query term receives a higher BM25 score than longer doc with same count."""
        index = BM25Index(k1=1.5, b=0.75)
        records = [
            {
                "chunk_id": "short",
                "text": "Supervised fine-tuning is effective.",
                "metadata": {"source": "short.pdf", "page_number": 1, "chunk_index": 0},
            },
            {
                "chunk_id": "long",
                "text": "Supervised fine-tuning is effective but requires massive amounts of data and compute and labels and evaluation setups and verification steps.",
                "metadata": {"source": "long.pdf", "page_number": 1, "chunk_index": 0},
            },
        ]
        index.add_records(records)
        results = index.search("fine-tuning")
        assert len(results) == 2
        assert results[0].chunk_id == "short"
        assert results[1].chunk_id == "long"
        assert results[0].score > results[1].score

    def test_top_k_limiting(self, sample_records):
        index = BM25Index()
        index.add_records(sample_records)
        # "models", "retrieval", "search" appear across multiple chunks
        results = index.search("models retrieval search", top_k=2)
        assert len(results) <= 2

    def test_document_id_filtering(self, sample_records):
        index = BM25Index()
        index.add_records(sample_records)

        # Search for "retrieval" which appears in c2 and c4
        all_results = index.search("retrieval", top_k=5)
        all_chunk_ids = {r.chunk_id for r in all_results}
        assert "c2" in all_chunk_ids
        assert "c4" in all_chunk_ids

        # Filter strictly to doc_rag (chunk c2)
        filtered = index.search("retrieval", top_k=5, document_ids=["doc_rag"])
        assert len(filtered) == 1
        assert filtered[0].chunk_id == "c2"
        assert filtered[0].document_id == "doc_rag"

    def test_document_id_filtering_no_match(self, sample_records):
        index = BM25Index()
        index.add_records(sample_records)
        filtered = index.search("retrieval", top_k=5, document_ids=["non_existent_doc"])
        assert filtered == []

    def test_index_clear_and_reindex(self, sample_records):
        index = BM25Index()
        index.add_records(sample_records)
        assert index.chunk_count == 4

        index.clear()
        assert index.chunk_count == 0
        assert index.search("BM25") == []

        index.add_records(sample_records[:2])
        assert index.chunk_count == 2
        assert len(index.search("Deep Learning")) == 1
