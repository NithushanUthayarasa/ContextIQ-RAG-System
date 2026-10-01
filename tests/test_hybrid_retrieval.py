"""
Tests for Hybrid Retrieval (app/retrieval/retriever.py and app/retrieval/bm25.py).
Verifies semantic, BM25, and hybrid (RRF) retrieval modes, score fusion, threshold gating,
document filtering, and RAG pipeline integration.
"""

from unittest.mock import MagicMock
import pytest

from app.generation.generator import GeminiGenerator
from app.ingestion.chunker import DocumentChunk
from app.rag.conversation import Conversation
from app.rag.pipeline import RAGPipeline
from app.retrieval.bm25 import BM25Index
from app.retrieval.models import RetrievedChunk
from app.retrieval.retriever import (
    EmptyQueryError,
    InvalidRetrievalModeError,
    InvalidTopKError,
    Retriever,
)
from app.vectorstore.chroma_store import ChromaVectorStore


@pytest.fixture
def sample_chunk_records():
    return [
        {
            "chunk_id": "doc1_c1",
            "text": "Retrieval-Augmented Generation (RAG) grounds LLM outputs in verified external knowledge.",
            "metadata": {
                "source": "rag_overview.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "document_id": "doc_rag",
            },
        },
        {
            "chunk_id": "doc1_c2",
            "text": "Hybrid search combines dense vector embeddings with sparse BM25 keyword matching.",
            "metadata": {
                "source": "rag_overview.pdf",
                "page_number": 2,
                "chunk_index": 1,
                "document_id": "doc_rag",
            },
        },
        {
            "chunk_id": "doc2_c1",
            "text": "ChromaDB is a lightweight open-source embedding database using cosine distance metrics.",
            "metadata": {
                "source": "chroma_guide.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "document_id": "doc_chroma",
            },
        },
        {
            "chunk_id": "doc3_c1",
            "text": "BM25 uses term frequency and inverse document frequency to score document relevance.",
            "metadata": {
                "source": "bm25_paper.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "document_id": "doc_bm25",
            },
        },
    ]


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 768
    return embedder


@pytest.fixture
def mock_vector_store(sample_chunk_records):
    store = MagicMock()
    store.count.return_value = len(sample_chunk_records)
    store.get_all_chunks.return_value = sample_chunk_records

    # Default query returns doc1_c1 (distance 0.20 -> sim 0.80) and doc1_c2 (distance 0.40 -> sim 0.60)
    def mock_query(query_embedding, top_k=5, where=None):
        return {
            "ids": [["doc1_c1", "doc1_c2"]],
            "documents": [[
                sample_chunk_records[0]["text"],
                sample_chunk_records[1]["text"],
            ]],
            "metadatas": [[
                sample_chunk_records[0]["metadata"],
                sample_chunk_records[1]["metadata"],
            ]],
            "distances": [[0.20, 0.40]],
        }

    store.query.side_effect = mock_query
    return store


class TestRetrieverModes:
    """Verifies retrieval mode dispatch, candidate pools, and score handling."""

    def test_default_mode_is_semantic(self, mock_embedder, mock_vector_store):
        retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store)
        assert retriever.default_retrieval_mode == "semantic"

        results = retriever.retrieve("What is RAG?", top_k=2)
        mock_embedder.embed_query.assert_called_once()
        mock_vector_store.query.assert_called_once()
        assert len(results) == 2
        assert results[0].retrieval_method == "semantic"
        assert results[0].distance == 0.20
        assert results[0].cosine_similarity == pytest.approx(0.80)
        assert results[0].bm25_score is None

    def test_bm25_mode_does_not_call_embedder_or_chroma_query(
        self, mock_embedder, mock_vector_store, sample_chunk_records
    ):
        retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store)
        results = retriever.retrieve("BM25 term frequency", top_k=2, retrieval_mode="bm25")

        mock_embedder.embed_query.assert_not_called()
        mock_vector_store.query.assert_not_called()
        assert len(results) > 0
        assert results[0].chunk_id == "doc3_c1"
        assert results[0].retrieval_method == "bm25"
        assert results[0].distance is None
        assert results[0].cosine_similarity is None
        assert results[0].bm25_score is not None
        assert results[0].bm25_score > 0.0

    def test_invalid_retrieval_mode_raises(self, mock_embedder, mock_vector_store):
        retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store)
        with pytest.raises(InvalidRetrievalModeError):
            retriever.retrieve("test", retrieval_mode="unsupported_mode")

    def test_empty_query_raises_in_all_modes(self, mock_embedder, mock_vector_store):
        retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store)
        for mode in ["semantic", "bm25", "hybrid"]:
            with pytest.raises(EmptyQueryError):
                retriever.retrieve("", retrieval_mode=mode)
            with pytest.raises(EmptyQueryError):
                retriever.retrieve("   ", retrieval_mode=mode)

    def test_invalid_top_k_raises_in_all_modes(self, mock_embedder, mock_vector_store):
        retriever = Retriever(embedder=mock_embedder, vector_store=mock_vector_store)
        for mode in ["semantic", "bm25", "hybrid"]:
            with pytest.raises(InvalidTopKError):
                retriever.retrieve("valid query", top_k=0, retrieval_mode=mode)
            with pytest.raises(InvalidTopKError):
                retriever.retrieve("valid query", top_k=-1, retrieval_mode=mode)


class TestHybridRRFMath:
    """Verifies Reciprocal Rank Fusion math and properties."""

    def test_rrf_scoring_and_candidate_multiplier(self, mock_embedder, sample_chunk_records):
        vector_store = MagicMock()
        vector_store.count.return_value = 4
        vector_store.get_all_chunks.return_value = sample_chunk_records

        # When queried with top_k=2, hybrid queries top_k*2 = 4 candidates
        def mock_query(query_embedding, top_k=4, where=None):
            assert top_k == 4
            return {
                "ids": [["doc1_c1", "doc1_c2"]],
                "documents": [[sample_chunk_records[0]["text"], sample_chunk_records[1]["text"]]],
                "metadatas": [[sample_chunk_records[0]["metadata"], sample_chunk_records[1]["metadata"]]],
                "distances": [[0.20, 0.35]],
            }

        vector_store.query.side_effect = mock_query

        retriever = Retriever(
            embedder=mock_embedder,
            vector_store=vector_store,
            default_min_similarity=0.50,
            rrf_k=60,
        )

        # Query matches "BM25" strongly in BM25 (doc3_c1 rank 1) and matches doc1_c1, doc1_c2 in semantic
        results = retriever.retrieve("BM25 keyword scoring", top_k=2, retrieval_mode="hybrid")
        assert len(results) <= 2

        # Every returned chunk has an rrf_score
        for r in results:
            assert r.rrf_score is not None
            assert r.rrf_score > 0.0

        # Ordered strictly descending by rrf_score
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_dual_hit_chunk_scores_higher_than_single_hit(self, mock_embedder, sample_chunk_records):
        """Chunk appearing in both semantic and keyword candidates gets sum of RRF reciprocal ranks."""
        vector_store = MagicMock()
        vector_store.count.return_value = 4
        vector_store.get_all_chunks.return_value = sample_chunk_records

        # Semantic returns doc1_c2 (rank 1) and doc2_c1 (rank 2)
        vector_store.query.return_value = {
            "ids": [["doc1_c2", "doc2_c1"]],
            "documents": [[sample_chunk_records[1]["text"], sample_chunk_records[2]["text"]]],
            "metadatas": [[sample_chunk_records[1]["metadata"], sample_chunk_records[2]["metadata"]]],
            "distances": [[0.25, 0.30]],
        }

        retriever = Retriever(embedder=mock_embedder, vector_store=vector_store, rrf_k=60)

        # "hybrid search" appears explicitly in doc1_c2, so doc1_c2 will rank #1 in BM25 as well
        results = retriever.retrieve("hybrid search", top_k=2, retrieval_mode="hybrid")

        # doc1_c2 should be rank 1 overall and have retrieval_method == 'hybrid'
        top_chunk = results[0]
        assert top_chunk.chunk_id == "doc1_c2"
        assert top_chunk.retrieval_method == "hybrid"
        assert top_chunk.semantic_rank == 1
        assert top_chunk.bm25_rank == 1
        # RRF = 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.0327868
        assert top_chunk.rrf_score == pytest.approx((1.0 / 61) + (1.0 / 61), rel=1e-4)

    def test_semantic_threshold_filtering_in_hybrid_mode(
        self, mock_embedder, sample_chunk_records
    ):
        """Semantic candidates failing similarity threshold are dropped from semantic candidates."""
        vector_store = MagicMock()
        vector_store.count.return_value = 4
        vector_store.get_all_chunks.return_value = sample_chunk_records

        # doc1_c1 distance 0.10 (sim 0.90), doc1_c2 distance 0.70 (sim 0.30)
        vector_store.query.return_value = {
            "ids": [["doc1_c1", "doc1_c2"]],
            "documents": [[sample_chunk_records[0]["text"], sample_chunk_records[1]["text"]]],
            "metadatas": [[sample_chunk_records[0]["metadata"], sample_chunk_records[1]["metadata"]]],
            "distances": [[0.10, 0.70]],
        }

        retriever = Retriever(
            embedder=mock_embedder,
            vector_store=vector_store,
            default_min_similarity=0.60,
        )

        results = retriever.retrieve(
            "retrieval knowledge",
            top_k=5,
            similarity_threshold=0.60,
            retrieval_mode="hybrid",
        )

        # doc1_c2 has similarity 0.30 (< 0.60), so it failed the semantic gate
        # If it appeared in results via BM25, its distance would not be treated as passing semantic
        semantic_chunk_ids = [r.chunk_id for r in results if r.retrieval_method in ("semantic", "hybrid")]
        assert "doc1_c1" in semantic_chunk_ids
        assert "doc1_c2" not in semantic_chunk_ids


class TestDocumentFilteringInHybrid:
    """Verifies that document_ids filter restricts both semantic and BM25 search."""

    def test_document_filter_restricts_both_engines(
        self, mock_embedder, sample_chunk_records
    ):
        vector_store = MagicMock()
        vector_store.count.return_value = 4
        vector_store.get_all_chunks.return_value = sample_chunk_records

        def mock_query(query_embedding, top_k, where=None):
            # Verify where filter restricted to doc_chroma
            assert where is not None
            assert where.get("document_id") == "doc_chroma"
            return {
                "ids": [["doc2_c1"]],
                "documents": [[sample_chunk_records[2]["text"]]],
                "metadatas": [[sample_chunk_records[2]["metadata"]]],
                "distances": [[0.15]],
            }

        vector_store.query.side_effect = mock_query

        retriever = Retriever(embedder=mock_embedder, vector_store=vector_store)
        results = retriever.retrieve(
            "database search",
            top_k=5,
            document_ids=["doc_chroma"],
            retrieval_mode="hybrid",
        )

        # All returned chunks must belong exclusively to doc_chroma
        assert len(results) > 0
        for r in results:
            assert r.document_id == "doc_chroma"


class TestRAGPipelineHybridIntegration:
    """Tests RAGPipeline end-to-end execution with hybrid retrieval mode."""

    def test_pipeline_hybrid_mode_propagation(
        self, mock_embedder, sample_chunk_records
    ):
        vector_store = MagicMock()
        vector_store.count.return_value = 4
        vector_store.get_all_chunks.return_value = sample_chunk_records
        vector_store.query.return_value = {
            "ids": [["doc1_c1"]],
            "documents": [[sample_chunk_records[0]["text"]]],
            "metadatas": [[sample_chunk_records[0]["metadata"]]],
            "distances": [[0.20]],
        }

        retriever = Retriever(embedder=mock_embedder, vector_store=vector_store)
        generator = MagicMock(spec=GeminiGenerator)
        generator.generate.return_value = "Grounded response about RAG."

        pipeline = RAGPipeline(retriever=retriever, generator=generator)
        response = pipeline.ask("Explain RAG", top_k=2, retrieval_mode="hybrid")

        assert response.answer == "Grounded response about RAG."
        assert response.retrieval_mode == "hybrid"
        assert len(response.retrieved_chunks) > 0

    def test_conversation_stores_retrieval_mode(self):
        conv = Conversation()
        msg = conv.add_assistant_message(
            content="Answer text",
            retrieval_mode="hybrid",
        )
        assert msg.retrieval_mode == "hybrid"
        assert conv.get_messages()[0].retrieval_mode == "hybrid"


class TestChromaGetAllChunks:
    """Verifies ChromaVectorStore.get_all_chunks functionality."""

    def test_get_all_chunks_empty(self, tmp_path):
        store = ChromaVectorStore(persist_dir=tmp_path, collection_name="test_empty")
        assert store.get_all_chunks() == []

    def test_get_all_chunks_populated(self, tmp_path):
        store = ChromaVectorStore(persist_dir=tmp_path, collection_name="test_pop", expected_dimension=3)
        chunk = DocumentChunk(
            chunk_id="chunk_1",
            text="Hello world test",
            source="test.pdf",
            page_number=1,
            chunk_index=0,
            document_id="doc_123",
        )
        store.add_chunks([chunk], [[0.1, 0.2, 0.3]])
        chunks = store.get_all_chunks()
        assert len(chunks) == 1
        assert chunks[0]["chunk_id"] == "chunk_1"
        assert chunks[0]["text"] == "Hello world test"
        assert chunks[0]["metadata"]["source"] == "test.pdf"
        assert chunks[0]["metadata"]["document_id"] == "doc_123"


class TestRetrieverEdgeCases:
    """Verifies edge cases in hybrid retrieval."""

    def test_empty_vector_store_returns_empty(self, mock_embedder):
        store = MagicMock()
        store.count.return_value = 0
        retriever = Retriever(embedder=mock_embedder, vector_store=store)
        for mode in ["semantic", "bm25", "hybrid"]:
            assert retriever.retrieve("query", retrieval_mode=mode) == []

    def test_bm25_fallback_when_semantic_empty(self, mock_embedder, sample_chunk_records):
        store = MagicMock()
        store.count.return_value = 4
        store.get_all_chunks.return_value = sample_chunk_records
        store.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        retriever = Retriever(embedder=mock_embedder, vector_store=store)
        results = retriever.retrieve("ChromaDB embeddings", top_k=2, retrieval_mode="hybrid")
        assert len(results) > 0
        assert results[0].retrieval_method == "bm25"
        assert results[0].bm25_score is not None

    def test_semantic_fallback_when_bm25_empty(self, mock_embedder, sample_chunk_records):
        store = MagicMock()
        store.count.return_value = 4
        store.get_all_chunks.return_value = sample_chunk_records
        store.query.return_value = {
            "ids": [["doc1_c1"]],
            "documents": [[sample_chunk_records[0]["text"]]],
            "metadatas": [[sample_chunk_records[0]["metadata"]]],
            "distances": [[0.20]],
        }

        retriever = Retriever(embedder=mock_embedder, vector_store=store)
        # Query with words not appearing in any document
        results = retriever.retrieve("xyzabc completelyunseenquery", top_k=2, retrieval_mode="hybrid")
        assert len(results) == 1
        assert results[0].chunk_id == "doc1_c1"
        assert results[0].retrieval_method == "semantic"
        assert results[0].rrf_score is not None

    def test_sync_bm25_index(self, mock_embedder, sample_chunk_records):
        store = MagicMock()
        store.count.return_value = len(sample_chunk_records)
        store.get_all_chunks.return_value = sample_chunk_records

        retriever = Retriever(embedder=mock_embedder, vector_store=store)
        assert retriever.bm25_index.chunk_count == 0

        indexed_count = retriever.sync_bm25_index()
        assert indexed_count == 4
        assert retriever.bm25_index.chunk_count == 4

        # Second call with force=False skips redundant reindex
        assert retriever.sync_bm25_index(force=False) == 4

