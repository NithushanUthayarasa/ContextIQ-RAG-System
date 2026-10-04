"""
Unit tests for ContextIQ Runtime Performance & Latency Tracking.
Tests PipelineTimings data model, RAGPipeline timing instrumentation,
RAGResponse backward-compatibility, Conversation message tracking,
and UI performance metrics rendering.
"""

from unittest.mock import MagicMock, patch
import pytest

from app.generation.generator import GeminiGenerator
from app.rag.conversation import ChatMessage, Conversation
from app.rag.pipeline import PipelineTimings, RAGPipeline, RAGResponse
from app.retrieval.compressor import BaseContextCompressor
from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import BaseReranker
from app.retrieval.retriever import Retriever
from app.ui.components import render_performance_metrics


# ---------------------------------------------------------------------------
# Fixtures & Mocks
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_chunk():
    return RetrievedChunk(
        chunk_id="chunk_1",
        text="Authentication verifies user identity. Authorization determines permissions.",
        source="security_lecture.pdf",
        page_number=14,
        chunk_index=0,
        distance=0.15,
    )


@pytest.fixture
def mock_retriever(sample_chunk):
    retriever = MagicMock(spec=Retriever)
    retriever.default_top_k = 3
    retriever.default_min_similarity = 0.5
    retriever.default_retrieval_mode = "semantic"
    retriever.retrieve.return_value = [sample_chunk]
    return retriever


@pytest.fixture
def mock_generator():
    generator = MagicMock(spec=GeminiGenerator)
    generator.generate.return_value = "Authentication checks credentials."
    return generator


# ---------------------------------------------------------------------------
# Test 1 — PipelineTimings Data Model
# ---------------------------------------------------------------------------
def test_pipeline_timings_initialization():
    """Verify initialization and default values of PipelineTimings."""
    t = PipelineTimings(
        total_ms=125.5,
        retrieval_ms=35.2,
        generation_ms=80.1,
    )
    assert t.total_ms == 125.5
    assert t.retrieval_ms == 35.2
    assert t.generation_ms == 80.1
    assert t.query_rewrite_ms is None
    assert t.query_expansion_ms is None
    assert t.reranking_ms is None
    assert t.parent_resolution_ms is None
    assert t.compression_ms is None

    # to_dict only includes non-null executed stages
    d = t.to_dict()
    assert d["total_ms"] == 125.5
    assert d["retrieval_ms"] == 35.2
    assert d["generation_ms"] == 80.1
    assert "query_rewrite_ms" not in d
    assert "compression_ms" not in d


def test_pipeline_timings_with_all_stages():
    """Verify to_dict includes optional stages when populated."""
    t = PipelineTimings(
        total_ms=250.0,
        query_rewrite_ms=20.0,
        query_expansion_ms=30.0,
        retrieval_ms=50.0,
        reranking_ms=15.0,
        parent_resolution_ms=10.0,
        compression_ms=25.0,
        generation_ms=100.0,
    )
    d = t.to_dict()
    assert len(d) == 8
    assert d["query_rewrite_ms"] == 20.0
    assert d["compression_ms"] == 25.0


# ---------------------------------------------------------------------------
# Test 2 — RAGPipeline Latency Tracking (Baseline Execution)
# ---------------------------------------------------------------------------
def test_pipeline_measures_baseline_latency(mock_retriever, mock_generator):
    """Verify RAGPipeline.ask measures retrieval, generation, and total wall-clock time."""
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
    )

    response = pipeline.ask("What is authentication?")

    assert response.timings is not None
    assert isinstance(response.timings, PipelineTimings)
    assert response.timings.total_ms >= 0.0
    assert response.timings.retrieval_ms >= 0.0
    assert response.timings.generation_ms >= 0.0
    # Optional stages should be None because they were not enabled/configured
    assert response.timings.query_rewrite_ms is None
    assert response.timings.query_expansion_ms is None
    assert response.timings.reranking_ms is None
    assert response.timings.parent_resolution_ms is None
    assert response.timings.compression_ms is None


# ---------------------------------------------------------------------------
# Test 3 — RAGPipeline Latency Tracking with Optional Stages
# ---------------------------------------------------------------------------
def test_pipeline_measures_reranking_and_compression_stages(mock_retriever, mock_generator, sample_chunk):
    """Verify optional stages (reranking, compression) record timings when active."""
    mock_reranker = MagicMock(spec=BaseReranker)
    mock_reranker.rerank.return_value = [sample_chunk]

    mock_compressor = MagicMock(spec=BaseContextCompressor)
    mock_compressor.compress.return_value = [sample_chunk]

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        reranker=mock_reranker,
        compressor=mock_compressor,
        context_compression_enabled=True,
    )

    response = pipeline.ask("What is authentication?")

    assert response.timings is not None
    assert response.timings.reranking_ms is not None
    assert response.timings.reranking_ms >= 0.0
    assert response.timings.compression_ms is not None
    assert response.timings.compression_ms >= 0.0
    assert response.timings.total_ms >= response.timings.retrieval_ms


def test_pipeline_measures_query_rewrite_stage(mock_retriever, mock_generator):
    """Verify query rewriting records latency when executed with conversation history."""
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite_query.return_value = "What is database authentication?"

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        query_rewriter=mock_rewriter,
    )

    history = [ChatMessage(role="user", content="Tell me about databases.")]
    response = pipeline.ask("What is authentication?", conversation_messages=history)

    assert response.timings is not None
    assert response.timings.query_rewrite_ms is not None
    assert response.timings.query_rewrite_ms >= 0.0
    mock_rewriter.rewrite_query.assert_called_once()


def test_pipeline_measures_query_expansion_stage(mock_retriever, mock_generator):
    """Verify query expansion records latency when executed."""
    mock_expander = MagicMock()
    mock_expander.expand.return_value = ["What is auth?", "How does authentication work?"]

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        query_expander=mock_expander,
    )

    response = pipeline.ask("What is authentication?")

    assert response.timings is not None
    assert response.timings.query_expansion_ms is not None
    assert response.timings.query_expansion_ms >= 0.0
    mock_expander.expand.assert_called_once()


# ---------------------------------------------------------------------------
# Test 4 — Backward Compatibility of RAGResponse & Conversation
# ---------------------------------------------------------------------------
def test_rag_response_backward_compatibility():
    """Verify RAGResponse can still be constructed without timings field."""
    resp = RAGResponse(
        answer="Grounded answer",
        sources=[{"source": "doc.pdf", "page": 1}],
        retrieved_chunks=[],
        query="test query",
    )
    assert resp.timings is None
    assert resp.answer == "Grounded answer"


def test_conversation_tracks_assistant_timings():
    """Verify Conversation.add_assistant_message stores timings and handles None."""
    conv = Conversation()
    t = PipelineTimings(total_ms=100.0, retrieval_ms=30.0, generation_ms=70.0)

    # 1. Message with timings
    msg1 = conv.add_assistant_message("Answer with timings", timings=t)
    assert msg1.timings == t

    # 2. Message without timings (backward-compatible)
    msg2 = conv.add_assistant_message("Legacy answer")
    assert msg2.timings is None


# ---------------------------------------------------------------------------
# Test 5 — UI Performance Metrics Renderer
# ---------------------------------------------------------------------------
@patch("app.ui.components.st")
def test_render_performance_metrics_none_safe(mock_st):
    """Verify render_performance_metrics safely ignores None timings without error."""
    render_performance_metrics(timings=None)
    mock_st.expander.assert_not_called()
    mock_st.metric.assert_not_called()


@patch("app.ui.components.st")
def test_render_performance_metrics_renders_executed_stages(mock_st):
    """Verify render_performance_metrics renders only executed stages and stats."""
    mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
    mock_st.expander.return_value.__enter__.return_value = MagicMock()

    timings = PipelineTimings(
        total_ms=1240.0,
        query_rewrite_ms=110.0,
        retrieval_ms=180.0,
        compression_ms=25.0,
        generation_ms=820.0,
    )

    render_performance_metrics(
        timings=timings,
        candidates_retrieved=15,
        final_chunk_count=5,
        total_chars_original=3200,
        total_chars_compressed=950,
    )

    mock_st.expander.assert_called_once_with("⚡ Performance", expanded=False)
    assert mock_st.metric.call_count == 5  # Total, Query Rewrite, Retrieval, Compression, Generation
    mock_st.caption.assert_called_once()
    caption_text = mock_st.caption.call_args[0][0]
    assert "Candidates retrieved: **15**" in caption_text
    assert "Final context chunks: **5**" in caption_text
    assert "reduction" in caption_text
