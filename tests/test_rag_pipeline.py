"""
Unit tests for ContextIQ RAGPipeline and RAGResponse.
Tests use mocked Retriever and GeminiGenerator instances to verify decoupled orchestration.
"""

from unittest.mock import MagicMock
import pytest

from app.generation.generator import GeminiGenerationError, EmptyQuestionError
from app.rag.pipeline import RAGPipeline, RAGResponse, RAGPipelineError
from app.retrieval.retriever import RetrievedChunk, Retriever, RetrieverError


@pytest.fixture
def mock_retriever():
    """Mock Retriever returning sample RetrievedChunk objects."""
    retriever = MagicMock(spec=Retriever)
    retriever.retrieve.return_value = [
        RetrievedChunk("c1", "RAG is a retrieval technique.", "paper.pdf", 3, 0, 0.15),
        RetrievedChunk("c2", "Chunking is critical for RAG.", "paper.pdf", 3, 1, 0.22),
        RetrievedChunk("c3", "Embeddings map text to vectors.", "paper.pdf", 7, 0, 0.35),
    ]
    return retriever


@pytest.fixture
def mock_generator():
    """Mock GeminiGenerator returning synthetic answer."""
    generator = MagicMock()
    generator.generate.return_value = "Retrieval-Augmented Generation combines retrieval with generation."
    return generator


def test_pipeline_initialization(mock_retriever, mock_generator):
    """Verify pipeline initialization and validation of required dependencies."""
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
    assert pipeline.retriever == mock_retriever
    assert pipeline.generator == mock_generator

    with pytest.raises(RAGPipelineError, match="valid Retriever instance"):
        RAGPipeline(retriever=None, generator=mock_generator)

    with pytest.raises(RAGPipelineError, match="valid GeminiGenerator instance"):
        RAGPipeline(retriever=mock_retriever, generator=None)


def test_successful_end_to_end_orchestration(mock_retriever, mock_generator):
    """Verify that ask() orchestrates retriever and generator returning RAGResponse."""
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
    resp = pipeline.ask("What is RAG?", top_k=3)

    assert isinstance(resp, RAGResponse)
    assert resp.answer == "Retrieval-Augmented Generation combines retrieval with generation."
    assert resp.query == "What is RAG?"
    assert len(resp.retrieved_chunks) == 3


def test_retriever_parameters_delegation(mock_retriever, mock_generator):
    """Verify that retriever receives trimmed question and top_k parameter."""
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
    pipeline.ask("  Explain embeddings  ", top_k=4)

    mock_retriever.retrieve.assert_called_once_with(query="Explain embeddings", top_k=4)


def test_generator_parameters_delegation(mock_retriever, mock_generator):
    """Verify that generator receives original question and retrieved chunks."""
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
    pipeline.ask("Explain embeddings")

    retrieved_chunks = mock_retriever.retrieve.return_value
    mock_generator.generate.assert_called_once_with(
        question="Explain embeddings",
        retrieved_chunks=retrieved_chunks,
    )


def test_source_deduplication_and_ordering():
    """
    Verify that duplicate (source, page) combinations are removed while preserving order:
    Input:
      Chunk 1: paper.pdf, page 3
      Chunk 2: paper.pdf, page 3 (duplicate page)
      Chunk 3: paper.pdf, page 7
      Chunk 4: other.pdf, page 1
    Output sources:
      1. paper.pdf - page 3
      2. paper.pdf - page 7
      3. other.pdf - page 1
    """
    chunks = [
        RetrievedChunk("c1", "text 1", "paper.pdf", 3, 0, 0.1),
        RetrievedChunk("c2", "text 2", "paper.pdf", 3, 1, 0.2),
        RetrievedChunk("c3", "text 3", "paper.pdf", 7, 0, 0.3),
        RetrievedChunk("c4", "text 4", "other.pdf", 1, 0, 0.4),
    ]
    sources = RAGPipeline.extract_sources(chunks)

    assert len(sources) == 3
    assert sources[0] == {"source": "paper.pdf", "page": 3}
    assert sources[1] == {"source": "paper.pdf", "page": 7}
    assert sources[2] == {"source": "other.pdf", "page": 1}


def test_empty_retrieval_handling(mock_retriever, mock_generator):
    """Verify handling when retriever returns no chunks."""
    mock_retriever.retrieve.return_value = []
    mock_generator.generate.return_value = "I couldn't find relevant information in the indexed documents."

    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
    resp = pipeline.ask("Unknown question")

    assert resp.sources == []
    assert resp.retrieved_chunks == []
    assert "couldn't find relevant information" in resp.answer


def test_empty_question_validation(mock_retriever, mock_generator):
    """Verify that empty questions raise EmptyQuestionError before calling retriever."""
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)

    with pytest.raises(EmptyQuestionError, match="Question must not be empty"):
        pipeline.ask("")

    with pytest.raises(EmptyQuestionError, match="Question must not be empty"):
        pipeline.ask("   \t\n  ")

    mock_retriever.retrieve.assert_not_called()


def test_retriever_error_propagation(mock_retriever, mock_generator):
    """Verify that RetrieverError propagates out cleanly."""
    mock_retriever.retrieve.side_effect = RetrieverError("ChromaDB connection dropped")
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)

    with pytest.raises(RetrieverError, match="ChromaDB connection dropped"):
        pipeline.ask("What is RAG?")


def test_generator_error_propagation(mock_retriever, mock_generator):
    """Verify that GeminiGenerationError propagates out cleanly."""
    mock_generator.generate.side_effect = GeminiGenerationError("Gemini quota exceeded")
    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)

    with pytest.raises(GeminiGenerationError, match="Gemini quota exceeded"):
        pipeline.ask("What is RAG?")


# ---------------------------------------------------------------------------
# Candidate Deduplication Regression Tests
# ---------------------------------------------------------------------------

def test_deduplication_semantic_smaller_distance_wins(mock_generator):
    """Verify that when distance is present for both duplicates, smaller distance wins."""
    c1_weak = RetrievedChunk("c1", "weak text", "doc.pdf", 1, 0, distance=0.5, retrieval_method="semantic")
    c1_strong = RetrievedChunk("c1", "strong text", "doc.pdf", 1, 0, distance=0.1, retrieval_method="semantic")

    retriever = MagicMock(spec=Retriever)
    retriever.retrieve.side_effect = [[c1_weak], [c1_strong]]

    mock_expander = MagicMock()
    mock_expander.expand.return_value = ["q1", "q2"]

    pipeline = RAGPipeline(retriever=retriever, generator=mock_generator, query_expander=mock_expander)
    resp = pipeline.ask("q1")

    assert len(resp.retrieved_chunks) == 1
    assert resp.retrieved_chunks[0].distance == 0.1
    assert resp.retrieved_chunks[0].text == "strong text"


def test_deduplication_bm25_none_distance_higher_bm25_wins(mock_generator):
    """
    Regression test: BM25 chunks have distance=None.
    Verify that comparing duplicate BM25 chunks does NOT raise TypeError: '<' not supported between instances of 'NoneType'.
    Higher bm25_score must win.
    """
    c1_low = RetrievedChunk("c1", "low score text", "doc.pdf", 1, 0, distance=None, bm25_score=2.5, retrieval_method="bm25")
    c1_high = RetrievedChunk("c1", "high score text", "doc.pdf", 1, 0, distance=None, bm25_score=8.5, retrieval_method="bm25")

    retriever = MagicMock(spec=Retriever)
    retriever.retrieve.side_effect = [[c1_low], [c1_high]]

    mock_expander = MagicMock()
    mock_expander.expand.return_value = ["q1", "q2"]

    pipeline = RAGPipeline(retriever=retriever, generator=mock_generator, query_expander=mock_expander)
    resp = pipeline.ask("q1", retrieval_mode="bm25")

    assert len(resp.retrieved_chunks) == 1
    assert resp.retrieved_chunks[0].distance is None
    assert resp.retrieved_chunks[0].bm25_score == 8.5
    assert resp.retrieved_chunks[0].text == "high score text"


def test_deduplication_rrf_none_distance_higher_rrf_wins(mock_generator):
    """
    Regression test: Hybrid chunks with distance=None must compare rrf_score without error.
    """
    c1_low = RetrievedChunk("c1", "low rrf", "doc.pdf", 1, 0, distance=None, rrf_score=0.015, retrieval_method="hybrid")
    c1_high = RetrievedChunk("c1", "high rrf", "doc.pdf", 1, 0, distance=None, rrf_score=0.032, retrieval_method="hybrid")

    retriever = MagicMock(spec=Retriever)
    retriever.retrieve.side_effect = [[c1_low], [c1_high]]

    mock_expander = MagicMock()
    mock_expander.expand.return_value = ["q1", "q2"]

    pipeline = RAGPipeline(retriever=retriever, generator=mock_generator, query_expander=mock_expander)
    resp = pipeline.ask("q1", retrieval_mode="hybrid")

    assert len(resp.retrieved_chunks) == 1
    assert resp.retrieved_chunks[0].rrf_score == 0.032
    assert resp.retrieved_chunks[0].text == "high rrf"


def test_deduplication_mixed_none_distance_graceful_fallback(mock_generator):
    """
    Regression test: When one duplicate has distance and another has distance=None (e.g. mixed streams),
    no TypeError is raised and deduplication completes safely.
    """
    c1_semantic = RetrievedChunk("c1", "semantic candidate", "doc.pdf", 1, 0, distance=0.2, retrieval_method="semantic")
    c1_bm25 = RetrievedChunk("c1", "bm25 candidate", "doc.pdf", 1, 0, distance=None, bm25_score=5.0, retrieval_method="bm25")

    retriever = MagicMock(spec=Retriever)
    # Return semantic first, then bm25 second
    retriever.retrieve.side_effect = [[c1_semantic], [c1_bm25]]

    mock_expander = MagicMock()
    mock_expander.expand.return_value = ["q1", "q2"]

    pipeline = RAGPipeline(retriever=retriever, generator=mock_generator, query_expander=mock_expander)
    resp = pipeline.ask("q1")

    assert len(resp.retrieved_chunks) == 1
    # First chunk preserved safely without raising TypeError
    assert resp.retrieved_chunks[0].chunk_id == "c1"


def test_deduplication_across_expanded_queries_hybrid_with_reranking(mock_generator):
    """
    Regression test: Full flow with hybrid retrieval, query expansion, and reranking
    handling duplicate chunks across expanded queries without error.
    """
    c1_first = RetrievedChunk("c1", "auth token overview", "doc.pdf", 2, 0, distance=None, rrf_score=0.02, bm25_score=4.0)
    c1_second = RetrievedChunk("c1", "auth token details", "doc.pdf", 2, 0, distance=None, rrf_score=0.03, bm25_score=6.0)
    c2 = RetrievedChunk("c2", "database connection pool", "doc.pdf", 4, 0, distance=None, rrf_score=0.015, bm25_score=3.0)

    retriever = MagicMock(spec=Retriever)
    retriever.retrieve.side_effect = [[c1_first], [c1_second, c2]]

    mock_expander = MagicMock()
    mock_expander.expand.return_value = ["auth tokens", "JWT authentication"]

    mock_reranker = MagicMock()
    # Reranker receives deduplicated candidates: [c1_second, c2]
    mock_reranker.rerank.side_effect = lambda query, candidates, top_k: candidates[:top_k]

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=mock_generator,
        reranker=mock_reranker,
        reranker_candidate_multiplier=2,
        query_expander=mock_expander,
    )
    resp = pipeline.ask("auth tokens", top_k=2, retrieval_mode="hybrid")

    assert resp.candidates_retrieved == 2
    assert len(resp.retrieved_chunks) == 2
    assert resp.retrieved_chunks[0].chunk_id == "c1"
    assert resp.retrieved_chunks[0].rrf_score == 0.03
