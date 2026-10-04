"""
ContextIQ - Context Compression Test Suite
Comprehensive tests verifying deterministic extractive compression, sentence splitting,
lexical relevance scoring, chronological order preservation, safe fallback for low-relevance
or empty candidates, metadata preservation, pipeline integration, and backward compatibility.
All tests use deterministic mocks; zero real Gemini API calls.
"""

from dataclasses import replace
import os
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
import pytest

from app.rag.conversation import ChatMessage, Conversation
from app.rag.pipeline import RAGPipeline, RAGResponse
from app.retrieval.compressor import (
    BaseContextCompressor,
    ContextCompressorError,
    ExtractiveContextCompressor,
    InvalidCompressorConfigError,
    _score_sentence,
    _split_into_sentences,
)
from app.retrieval.models import RetrievalResult, RetrievedChunk
from app.retrieval.reranker import TFIDFReranker


# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------

class MockRetriever:
    """Mock retriever returning predetermined chunks."""
    def __init__(self, chunks: Optional[List[RetrievedChunk]] = None):
        self.chunks = chunks or []
        self.default_top_k = 5
        self.default_min_similarity = 0.50
        self.default_retrieval_mode = "semantic"
        self.vector_store = None

    def retrieve(self, **kwargs) -> List[RetrievedChunk]:
        top_k = kwargs.get("top_k", self.default_top_k)
        thresh = kwargs.get("similarity_threshold", self.default_min_similarity)
        doc_ids = kwargs.get("document_ids")

        filtered = self.chunks
        if doc_ids:
            filtered = [c for c in filtered if c.document_id in doc_ids]
        if thresh is not None:
            filtered = [
                c for c in filtered
                if c.cosine_similarity is None or c.cosine_similarity >= thresh
            ]
        return filtered[:top_k]


class MockGenerator:
    """Mock generator that records the chunks passed to generate()."""
    def __init__(self):
        self.last_retrieved_chunks = []
        self.last_question = None

    def generate(self, question: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        self.last_question = question
        self.last_retrieved_chunks = list(retrieved_chunks)
        return "Deterministic answer based on provided context."


def make_chunk(
    chunk_id: str,
    text: str,
    source: str = "doc.pdf",
    page: int = 1,
    index: int = 0,
    distance: float = 0.20,
    parent_id: Optional[str] = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page_number=page,
        chunk_index=index,
        distance=distance,
        document_id="doc-sha256-abc",
        parent_id=parent_id,
    )


# ---------------------------------------------------------------------------
# 1. Configuration & Validation Tests
# ---------------------------------------------------------------------------

class TestCompressorConfigValidation:
    def test_valid_default_initialization(self):
        compressor = ExtractiveContextCompressor()
        assert compressor.max_sentences >= 1
        assert 0.0 <= compressor.similarity_threshold <= 1.0
        assert compressor.min_sentence_len >= 1

    def test_custom_valid_initialization(self):
        compressor = ExtractiveContextCompressor(
            max_sentences=3,
            similarity_threshold=0.25,
            min_sentence_len=20,
        )
        assert compressor.max_sentences == 3
        assert compressor.similarity_threshold == 0.25
        assert compressor.min_sentence_len == 20

    def test_invalid_max_sentences_raises(self):
        with pytest.raises(InvalidCompressorConfigError):
            ExtractiveContextCompressor(max_sentences=0)
        with pytest.raises(InvalidCompressorConfigError):
            ExtractiveContextCompressor(max_sentences=-5)
        with pytest.raises(InvalidCompressorConfigError):
            ExtractiveContextCompressor(max_sentences=True)  # type: ignore

    def test_invalid_similarity_threshold_raises(self):
        with pytest.raises(InvalidCompressorConfigError):
            ExtractiveContextCompressor(similarity_threshold=-0.1)
        with pytest.raises(InvalidCompressorConfigError):
            ExtractiveContextCompressor(similarity_threshold=1.5)
        with pytest.raises(InvalidCompressorConfigError):
            ExtractiveContextCompressor(similarity_threshold=True)  # type: ignore

    def test_invalid_min_sentence_len_raises(self):
        with pytest.raises(InvalidCompressorConfigError):
            ExtractiveContextCompressor(min_sentence_len=0)
        with pytest.raises(InvalidCompressorConfigError):
            ExtractiveContextCompressor(min_sentence_len=-10)


# ---------------------------------------------------------------------------
# 2. Sentence Splitting & Scoring Unit Tests
# ---------------------------------------------------------------------------

class TestSentenceSplittingAndScoring:
    def test_split_into_sentences_standard(self):
        text = "ContextIQ is an intelligent RAG system. It indexes documents into ChromaDB. It uses Gemini embeddings."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "ContextIQ is an intelligent RAG system."
        assert sentences[1] == "It indexes documents into ChromaDB."
        assert sentences[2] == "It uses Gemini embeddings."

    def test_split_into_sentences_with_newlines(self):
        text = "Heading text\nSecond sentence here.\n\nThird sentence follows."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "Heading text"
        assert sentences[1] == "Second sentence here."
        assert sentences[2] == "Third sentence follows."

    def test_split_into_sentences_empty_or_whitespace(self):
        assert _split_into_sentences("") == []
        assert _split_into_sentences("   \n\t  ") == []

    def test_score_sentence_relevant(self):
        query_tokens = ["chromadb", "vector", "store"]
        sent_tokens = ["chromadb", "is", "a", "high", "performance", "vector", "store"]
        score = _score_sentence(query_tokens, sent_tokens)
        assert score > 0.40

    def test_score_sentence_irrelevant(self):
        query_tokens = ["chromadb", "vector"]
        sent_tokens = ["today", "weather", "forecast", "predicts", "rain"]
        score = _score_sentence(query_tokens, sent_tokens)
        assert score == 0.0

    def test_score_sentence_stop_words_only(self):
        query_tokens = ["what", "is", "chromadb"]
        sent_tokens = ["this", "is", "a", "cup", "of", "tea"]
        score = _score_sentence(query_tokens, sent_tokens)
        # "chromadb" is the content word in query; "is" is ignored, so tea sentence scores 0.0
        assert score == 0.0


# ---------------------------------------------------------------------------
# 3. Extractive Context Compressor Behavior Tests
# ---------------------------------------------------------------------------

class TestExtractiveContextCompressor:
    def setup_method(self):
        self.compressor = ExtractiveContextCompressor(
            max_sentences=3,
            similarity_threshold=0.10,
            min_sentence_len=15,
        )

    def test_compress_empty_chunks(self):
        assert self.compressor.compress("query", []) == []

    def test_compress_empty_query_fallback(self):
        chunk = make_chunk("c1", "ContextIQ uses ChromaDB for vector retrieval. It supports BM25 too.")
        results = self.compressor.compress("", [chunk])
        assert len(results) == 1
        assert results[0].text == chunk.text
        assert results[0].compression_ratio == 1.0

    def test_compress_filters_irrelevant_sentences(self):
        text = (
            "ContextIQ utilizes ChromaDB as its embedded vector database. "
            "The cafeteria menu features delicious sandwiches on Tuesdays. "
            "ChromaDB stores document embeddings and performs similarity search. "
            "The office air conditioner was serviced last Thursday."
        )
        chunk = make_chunk("c1", text)
        results = self.compressor.compress("How does ChromaDB store embeddings?", [chunk])

        assert len(results) == 1
        compressed = results[0]

        # Relevant sentences kept
        assert "ChromaDB" in compressed.text
        # Irrelevant sentences omitted
        assert "cafeteria" not in compressed.text
        assert "air conditioner" not in compressed.text

        assert compressed.sentences_kept == 2
        assert compressed.sentences_total == 4
        assert compressed.original_text == text
        assert compressed.compression_ratio is not None
        assert 0.0 < compressed.compression_ratio < 1.0

    def test_chronological_order_is_preserved(self):
        text = (
            "First sentence introduces the ChromaDB architecture. "
            "Second irrelevant sentence about garden flowers. "
            "Third sentence concludes the ChromaDB storage overview."
        )
        chunk = make_chunk("c1", text)
        results = self.compressor.compress("ChromaDB architecture overview", [chunk])

        assert len(results) == 1
        comp_text = results[0].text
        idx_first = comp_text.find("First sentence")
        idx_third = comp_text.find("Third sentence")
        assert idx_first != -1
        assert idx_third != -1
        assert idx_first < idx_third

    def test_safe_fallback_when_no_sentences_pass_threshold(self):
        text = (
            "The weather in Tokyo is mild and pleasant in spring. "
            "Mount Fuji is visible from Tokyo on clear mornings."
        )
        chunk = make_chunk("c1", text)
        results = self.compressor.compress("How does ChromaDB indexing work?", [chunk])

        assert len(results) == 1
        # Fallback preserves full text
        assert results[0].text == text
        assert results[0].original_text == text
        assert results[0].compression_ratio == 1.0
        assert results[0].sentences_kept == 2
        assert results[0].sentences_total == 2

    def test_max_sentences_limit(self):
        text = (
            "Sentence 1 about ChromaDB features and storage. "
            "Sentence 2 about ChromaDB performance benchmarks. "
            "Sentence 3 about ChromaDB integration with Python. "
            "Sentence 4 about ChromaDB configuration options."
        )
        chunk = make_chunk("c1", text)
        # Limit to max 2 sentences
        results = self.compressor.compress("ChromaDB features", [chunk], max_sentences=2)
        assert len(results) == 1
        assert results[0].sentences_kept == 2
        assert results[0].sentences_total == 4

    def test_preserves_provenance_and_metadata(self):
        chunk = make_chunk("c-42", "ChromaDB stores embeddings. Coffee is hot.", page=7, index=3, distance=0.15)
        chunk.bm25_score = 4.2
        chunk.parent_id = "parent-99"
        chunk.original_child_id = "child-88"

        results = self.compressor.compress("ChromaDB embeddings", [chunk])
        c = results[0]
        assert c.chunk_id == "c-42"
        assert c.source == "doc.pdf"
        assert c.page_number == 7
        assert c.chunk_index == 3
        assert c.distance == 0.15
        assert c.document_id == "doc-sha256-abc"
        assert c.bm25_score == 4.2
        assert c.parent_id == "parent-99"
        assert c.original_child_id == "child-88"

    def test_call_time_overrides_validation(self):
        chunk = make_chunk("c1", "Sentence one with ChromaDB. Sentence two with ChromaDB.")
        with pytest.raises(InvalidCompressorConfigError):
            self.compressor.compress("query", [chunk], max_sentences=0)
        with pytest.raises(InvalidCompressorConfigError):
            self.compressor.compress("query", [chunk], similarity_threshold=1.5)


# ---------------------------------------------------------------------------
# 4. Pipeline Integration Tests
# ---------------------------------------------------------------------------

class TestPipelineCompressionIntegration:
    def test_pipeline_compression_disabled_by_default(self):
        chunk = make_chunk(
            "c1",
            "ContextIQ uses ChromaDB. Lunch is served at noon. ChromaDB performs similarity searches."
        )
        retriever = MockRetriever([chunk])
        generator = MockGenerator()
        compressor = ExtractiveContextCompressor()

        pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            compressor=compressor,
            context_compression_enabled=False,
        )

        resp = pipeline.ask("What database does ContextIQ use?")
        assert resp.context_compression_enabled is False
        assert resp.total_chars_original is None
        assert resp.total_chars_compressed is None
        # Generator received original uncompressed chunk
        assert generator.last_retrieved_chunks[0].text == chunk.text

    def test_pipeline_compression_enabled_compresses_chunks(self):
        chunk = make_chunk(
            "c1",
            "ContextIQ uses ChromaDB. Lunch is served at noon. ChromaDB performs similarity searches."
        )
        retriever = MockRetriever([chunk])
        generator = MockGenerator()
        compressor = ExtractiveContextCompressor()

        pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            compressor=compressor,
            context_compression_enabled=True,
        )

        resp = pipeline.ask("What database does ContextIQ use?")
        assert resp.context_compression_enabled is True
        assert resp.total_chars_original is not None
        assert resp.total_chars_compressed is not None
        assert resp.total_chars_compressed < resp.total_chars_original
        # Generator received compressed chunk
        assert "Lunch is served" not in generator.last_retrieved_chunks[0].text
        assert "ContextIQ uses ChromaDB" in generator.last_retrieved_chunks[0].text

    def test_pipeline_ask_runtime_override(self):
        chunk = make_chunk(
            "c1",
            "ContextIQ uses ChromaDB. Lunch is served at noon. ChromaDB performs similarity searches."
        )
        retriever = MockRetriever([chunk])
        generator = MockGenerator()
        compressor = ExtractiveContextCompressor()

        pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            compressor=compressor,
            context_compression_enabled=False,
        )

        # Override context_compression_enabled=True for this single turn
        resp = pipeline.ask("What database does ContextIQ use?", context_compression_enabled=True)
        assert resp.context_compression_enabled is True
        assert resp.total_chars_compressed < resp.total_chars_original

    def test_pipeline_full_order_rerank_parent_and_compression(self):
        """
        Verify pipeline order: Retrieval → Dedup → Reranker → Parent Context → Compression → Generator
        """
        parent_text = (
            "Parent Section Header. "
            "ContextIQ utilizes ChromaDB for document storage. "
            "The weather outside is cold and rainy. "
            "ChromaDB stores high dimensional vectors efficiently."
        )
        parent_chunk = make_chunk("p1", parent_text)
        child_chunk = make_chunk("c1", "ContextIQ utilizes ChromaDB for document storage.", parent_id="p1")

        mock_store = MagicMock()
        mock_store.get_parents.return_value = {
            "p1": {
                "parent_id": "p1",
                "text": parent_text,
                "source": "doc.pdf",
                "page_number": 1,
                "parent_index": 0,
            }
        }

        retriever = MockRetriever([child_chunk])
        generator = MockGenerator()
        reranker = TFIDFReranker()
        compressor = ExtractiveContextCompressor()

        pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            reranker=reranker,
            parent_store=mock_store,
            parent_child_enabled=True,
            compressor=compressor,
            context_compression_enabled=True,
        )

        resp = pipeline.ask("How does ChromaDB store vectors?")
        assert resp.reranking_enabled is True
        assert resp.parent_child_enabled is True
        assert resp.context_compression_enabled is True

        gen_chunk = generator.last_retrieved_chunks[0]
        assert gen_chunk.parent_id == "p1"
        assert "weather outside is cold" not in gen_chunk.text
        assert "ChromaDB" in gen_chunk.text


# ---------------------------------------------------------------------------
# 5. Conversation Memory Tests
# ---------------------------------------------------------------------------

class TestConversationMemoryCompression:
    def test_chat_message_compression_fields(self):
        msg = ChatMessage(
            role="assistant",
            content="Answer text",
            context_compression_enabled=True,
            total_chars_original=500,
            total_chars_compressed=200,
        )
        assert msg.context_compression_enabled is True
        assert msg.total_chars_original == 500
        assert msg.total_chars_compressed == 200

    def test_conversation_add_assistant_message_stores_compression(self):
        conv = Conversation()
        msg = conv.add_assistant_message(
            content="Generated answer",
            context_compression_enabled=True,
            total_chars_original=1000,
            total_chars_compressed=450,
        )
        assert msg.context_compression_enabled is True
        assert msg.total_chars_original == 1000
        assert msg.total_chars_compressed == 450
