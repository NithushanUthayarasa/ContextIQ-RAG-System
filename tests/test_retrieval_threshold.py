"""
Unit and integration tests for similarity threshold and irrelevant-result filtering (Phase 13 Step 2).
"""

from unittest.mock import MagicMock
import pytest

from app.ingestion.embedder import GeminiEmbedder
from app.generation.generator import GeminiGenerator
from app.rag.pipeline import RAGPipeline, RAGResponse
from app.retrieval.models import RetrievalResult, RetrievedChunk
from app.retrieval.retriever import (
    InvalidSimilarityThresholdError,
    Retriever,
)
from app.vectorstore.chroma_store import ChromaVectorStore


@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=GeminiEmbedder)
    embedder.embed_query.return_value = [0.1] * 768
    return embedder


def create_fake_store_with_distances(distances: list, doc_ids=None):
    """Helper to create a mocked ChromaVectorStore returning specific distances and metadata."""
    count = len(distances)
    ids = [f"c_{i}" for i in range(count)]
    docs = [f"Content for chunk {i}" for i in range(count)]
    metadatas = [
        {
            "source": f"doc_{i}.pdf",
            "page_number": i + 1,
            "chunk_index": 0,
            "document_id": (doc_ids[i] if doc_ids else f"hash_{i}"),
        }
        for i in range(count)
    ]
    store = MagicMock(spec=ChromaVectorStore)
    store.count.return_value = count
    store.query.return_value = {
        "ids": [ids],
        "documents": [docs],
        "metadatas": [metadatas],
        "distances": [distances],
    }
    return store


def test_high_similarity_results_retained(mock_embedder):
    """Test 1: Similarities 0.95, 0.85, 0.75 with threshold 0.70 retain all 3 results."""
    # Cosine distances: 0.05, 0.15, 0.25 -> similarities: 0.95, 0.85, 0.75
    store = create_fake_store_with_distances([0.05, 0.15, 0.25])
    retriever = Retriever(embedder=mock_embedder, vector_store=store, default_min_similarity=0.70)

    results = retriever.retrieve("query", top_k=3)
    assert len(results) == 3
    assert [round(r.cosine_similarity, 2) for r in results] == [0.95, 0.85, 0.75]


def test_low_similarity_results_removed(mock_embedder):
    """Test 2: Similarities 0.95, 0.65, 0.40 with threshold 0.70 retain only 1 result."""
    # Cosine distances: 0.05, 0.35, 0.60 -> similarities: 0.95, 0.65, 0.40
    store = create_fake_store_with_distances([0.05, 0.35, 0.60])
    retriever = Retriever(embedder=mock_embedder, vector_store=store, default_min_similarity=0.70)

    results = retriever.retrieve("query", top_k=3)
    assert len(results) == 1
    assert results[0].chunk_id == "c_0"
    assert round(results[0].cosine_similarity, 2) == 0.95


def test_boundary_accepted(mock_embedder):
    """Test 3: Similarity exactly equal to threshold (0.70) is accepted."""
    # distance = 0.30 -> cosine_similarity = 0.70
    store = create_fake_store_with_distances([0.30])
    retriever = Retriever(embedder=mock_embedder, vector_store=store, default_min_similarity=0.70)

    results = retriever.retrieve("boundary query", top_k=1)
    assert len(results) == 1
    assert round(results[0].cosine_similarity, 2) == 0.70


def test_just_below_boundary_rejected(mock_embedder):
    """Test 4: Similarity 0.6999 (just below 0.70) is rejected."""
    # distance = 0.3001 -> cosine_similarity = 0.6999
    store = create_fake_store_with_distances([0.3001])
    retriever = Retriever(embedder=mock_embedder, vector_store=store, default_min_similarity=0.70)

    results = retriever.retrieve("query", top_k=1)
    assert len(results) == 0


def test_all_results_rejected(mock_embedder):
    """Test 5: Similarities 0.40, 0.30, 0.20 with threshold 0.70 return empty list."""
    # distances = 0.60, 0.70, 0.80 -> similarities = 0.40, 0.30, 0.20
    store = create_fake_store_with_distances([0.60, 0.70, 0.80])
    retriever = Retriever(embedder=mock_embedder, vector_store=store, default_min_similarity=0.70)

    results = retriever.retrieve("unrelated query", top_k=3)
    assert results == []


def test_ordering_preserved_after_filtering(mock_embedder):
    """Test 6: Ordering 0.92, 0.81, 0.73, 0.60 with threshold 0.70 preserves relative order."""
    # distances = 0.08, 0.19, 0.27, 0.40 -> similarities = 0.92, 0.81, 0.73, 0.60
    store = create_fake_store_with_distances([0.08, 0.19, 0.27, 0.40])
    retriever = Retriever(embedder=mock_embedder, vector_store=store)

    results = retriever.retrieve("query", top_k=4, similarity_threshold=0.70)
    assert len(results) == 3
    assert [round(r.cosine_similarity, 2) for r in results] == [0.92, 0.81, 0.73]
    assert results[0].distance < results[1].distance < results[2].distance


def test_distance_and_similarity_unaltered_by_filtering(mock_embedder):
    """Test 7: Filtering does not round or mutate distance and similarity values."""
    raw_dist = 0.1842391
    store = create_fake_store_with_distances([raw_dist])
    retriever = Retriever(embedder=mock_embedder, vector_store=store)

    results = retriever.retrieve("query", top_k=1, similarity_threshold=0.50)
    assert len(results) == 1
    assert results[0].distance == raw_dist
    assert results[0].cosine_similarity == (1.0 - raw_dist)


def test_metadata_survives_filtering(mock_embedder):
    """Test 8: Retained chunks preserve document_id, source, page_number, chunk_index, and text."""
    store = create_fake_store_with_distances([0.15], doc_ids=["sha256_full_hash"])
    retriever = Retriever(embedder=mock_embedder, vector_store=store)

    results = retriever.retrieve("query", top_k=1, similarity_threshold=0.50)
    assert len(results) == 1
    chunk = results[0]
    assert chunk.document_id == "sha256_full_hash"
    assert chunk.source == "doc_0.pdf"
    assert chunk.page_number == 1
    assert chunk.chunk_index == 0
    assert chunk.chunk_id == "c_0"
    assert chunk.text == "Content for chunk 0"


def test_top_k_and_threshold_interaction(mock_embedder):
    """Test 9: Vector store retrieves top_k=3 candidates; only those meeting threshold are retained."""
    fake_store = MagicMock(spec=ChromaVectorStore)
    fake_store.count.return_value = 4
    # Chroma returns only top 3 candidates: distances 0.05, 0.10, 0.15 (similarities 0.95, 0.90, 0.85)
    # The 4th candidate (similarity 0.80) is NOT returned by Chroma when top_k=3
    fake_store.query.return_value = {
        "ids": [["c_0", "c_1", "c_2"]],
        "documents": [["doc 0", "doc 1", "doc 2"]],
        "metadatas": [[
            {"source": "a.pdf", "page_number": 1, "chunk_index": 0},
            {"source": "a.pdf", "page_number": 1, "chunk_index": 1},
            {"source": "a.pdf", "page_number": 2, "chunk_index": 0},
        ]],
        "distances": [[0.05, 0.10, 0.15]],
    }
    retriever = Retriever(embedder=mock_embedder, vector_store=fake_store)
    results = retriever.retrieve("query", top_k=3, similarity_threshold=0.90)

    # Candidates were 0.95, 0.90, 0.85; with threshold 0.90, exactly 2 remain
    assert len(results) == 2
    assert [round(r.cosine_similarity, 2) for r in results] == [0.95, 0.90]
    fake_store.query.assert_called_once_with(query_embedding=mock_embedder.embed_query.return_value, top_k=3, where=None)


def test_invalid_threshold_raises_error(mock_embedder):
    """Test 10: Negative, > 1.0, and non-numeric thresholds raise InvalidSimilarityThresholdError."""
    store = create_fake_store_with_distances([0.1])
    retriever = Retriever(embedder=mock_embedder, vector_store=store)

    with pytest.raises(InvalidSimilarityThresholdError, match="between 0.0 and 1.0"):
        retriever.retrieve("q", similarity_threshold=-0.1)

    with pytest.raises(InvalidSimilarityThresholdError, match="between 0.0 and 1.0"):
        retriever.retrieve("q", similarity_threshold=1.1)

    with pytest.raises(InvalidSimilarityThresholdError, match="numeric value"):
        retriever.retrieve("q", similarity_threshold="high")

    with pytest.raises(InvalidSimilarityThresholdError, match="numeric value"):
        retriever.retrieve("q", similarity_threshold=True)

    with pytest.raises(InvalidSimilarityThresholdError, match="between 0.0 and 1.0"):
        Retriever(embedder=mock_embedder, vector_store=store, default_min_similarity=-0.05)


def test_pipeline_all_rejected_generates_empty_context_message(mock_embedder):
    """
    Step 16: Pipeline integration test.
    When all candidates are rejected by the similarity threshold:
    - Generator is not given weak chunks
    - Generator returns insufficient context message
    """
    # Vector store produces weak match: distance = 0.70 -> similarity = 0.30
    store = create_fake_store_with_distances([0.70])
    retriever = Retriever(embedder=mock_embedder, vector_store=store, default_min_similarity=0.60)

    generator = MagicMock(spec=GeminiGenerator)
    generator.EMPTY_CONTEXT_MESSAGE = "I couldn't find relevant information in the indexed documents."
    generator.generate.return_value = "I couldn't find relevant information in the indexed documents."

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.ask("What is the temperature on Neptune?", similarity_threshold=0.60)

    assert isinstance(response, RAGResponse)
    assert response.retrieved_chunks == []
    assert response.sources == []
    assert response.answer == "I couldn't find relevant information in the indexed documents."
    # Verify generator.generate was called with empty list of retrieved chunks
    generator.generate.assert_called_once_with(
        question="What is the temperature on Neptune?",
        retrieved_chunks=[],
    )
