"""
Unit and integration tests for query expansion (app/rag/query_expander.py, app/rag/pipeline.py).
All tests are fast, deterministic, and mock all Gemini API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from app.rag.query_expander import BaseQueryExpander, GeminiQueryExpander, QueryExpanderError
from app.rag.pipeline import RAGPipeline, RAGResponse, EmptyQuestionError
from app.retrieval.models import RetrievedChunk


def make_chunk(chunk_id: str, text: str, distance: float = 0.2) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source="test.pdf",
        page_number=1,
        chunk_index=0,
        distance=distance,
        retrieval_method="semantic",
    )


# ---------------------------------------------------------------------------
# 1. Base Class Tests
# ---------------------------------------------------------------------------

def test_base_query_expander_not_implemented():
    class IncompleteExpander(BaseQueryExpander):
        pass

    with pytest.raises(TypeError):
        IncompleteExpander()  # Cannot instantiate abstract class without implementation


# ---------------------------------------------------------------------------
# 2. GeminiQueryExpander Initialization & Validation
# ---------------------------------------------------------------------------

def test_gemini_query_expander_init_defaults():
    expander = GeminiQueryExpander(api_key="")
    assert expander.client is None


def test_gemini_query_expander_invalid_max_queries():
    expander = GeminiQueryExpander(api_key="")
    with pytest.raises(QueryExpanderError, match="must be >= 1"):
        expander.expand("test query", max_queries=0)
    with pytest.raises(QueryExpanderError, match="must be >= 1"):
        expander.expand("test query", max_queries=-1)


def test_gemini_query_expander_empty_query():
    expander = GeminiQueryExpander(api_key="")
    assert expander.expand("", max_queries=3) == []
    assert expander.expand("   ", max_queries=3) == []


# ---------------------------------------------------------------------------
# 3. GeminiQueryExpander Disabled / Fallback Behavior
# ---------------------------------------------------------------------------

def test_gemini_query_expander_disabled_returns_original():
    with patch("app.rag.query_expander.QUERY_EXPANSION_ENABLED", False):
        mock_client = MagicMock()
        expander = GeminiQueryExpander(client=mock_client)
        result = expander.expand("what is RAG?", max_queries=3)
        assert result == ["what is RAG?"]
        mock_client.models.generate_content.assert_not_called()


def test_gemini_query_expander_no_client_returns_original():
    with patch("app.rag.query_expander.QUERY_EXPANSION_ENABLED", True):
        expander = GeminiQueryExpander(api_key="")
        result = expander.expand("what is RAG?", max_queries=3)
        assert result == ["what is RAG?"]


def test_gemini_query_expander_api_failure_fallback():
    with patch("app.rag.query_expander.QUERY_EXPANSION_ENABLED", True):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API rate limit exceeded")
        expander = GeminiQueryExpander(client=mock_client)
        result = expander.expand("what is RAG?", max_queries=3)
        assert result == ["what is RAG?"]


# ---------------------------------------------------------------------------
# 4. Parsing and Deduplication in GeminiQueryExpander
# ---------------------------------------------------------------------------

def test_gemini_query_expander_successful_expansion():
    with patch("app.rag.query_expander.QUERY_EXPANSION_ENABLED", True):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "retrieval augmented generation definition\nhow does RAG work in AI?\nretrieval augmented generation"
        mock_client.models.generate_content.return_value = mock_response

        expander = GeminiQueryExpander(client=mock_client)
        result = expander.expand("what is RAG?", max_queries=3)

        assert len(result) == 3
        assert result[0] == "what is RAG?"
        assert result[1] == "retrieval augmented generation definition"
        assert result[2] == "how does RAG work in AI?"


def test_gemini_query_expander_filters_duplicates_and_empty_lines():
    with patch("app.rag.query_expander.QUERY_EXPANSION_ENABLED", True):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "\n\nwhat is RAG?\nquery two\nquery two\nquery three\n"
        mock_client.models.generate_content.return_value = mock_response

        expander = GeminiQueryExpander(client=mock_client)
        result = expander.expand("what is RAG?", max_queries=3)

        assert result == ["what is RAG?", "query two", "query three"]


# ---------------------------------------------------------------------------
# 5. RAGPipeline Integration with Query Expansion
# ---------------------------------------------------------------------------

def test_rag_pipeline_without_expander():
    retriever = MagicMock()
    generator = MagicMock()
    generator.generate.return_value = "Answer"
    chunk = make_chunk("c1", "content 1", distance=0.1)
    retriever.retrieve.return_value = [chunk]

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.ask("test question")

    assert response.query_expansion_enabled is False
    assert response.expanded_queries == ["test question"]
    assert retriever.retrieve.call_count == 1
    assert generator.generate.call_args.kwargs["question"] == "test question"


def test_rag_pipeline_with_expander_retrieves_per_query():
    retriever = MagicMock()
    generator = MagicMock()
    generator.generate.return_value = "Generated answer"

    c1 = make_chunk("c1", "content 1", distance=0.2)
    c2 = make_chunk("c2", "content 2", distance=0.3)
    retriever.retrieve.side_effect = [[c1], [c2]]

    mock_expander = MagicMock()
    mock_expander.expand.return_value = ["original query", "expanded query 2"]

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        query_expander=mock_expander,
    )
    response = pipeline.ask("original query", top_k=5)

    assert response.query_expansion_enabled is True
    assert response.expanded_queries == ["original query", "expanded query 2"]
    assert retriever.retrieve.call_count == 2
    assert len(response.retrieved_chunks) == 2
    # Generator receives original question
    assert generator.generate.call_args.kwargs["question"] == "original query"


def test_rag_pipeline_deduplication_keeps_strongest_chunk():
    retriever = MagicMock()
    generator = MagicMock()
    generator.generate.return_value = "Answer"

    # Chunk c1 returned twice: once with distance 0.4, once with distance 0.1
    c1_weak = make_chunk("c1", "content 1 weak", distance=0.4)
    c1_strong = make_chunk("c1", "content 1 strong", distance=0.1)
    c2 = make_chunk("c2", "content 2", distance=0.3)

    retriever.retrieve.side_effect = [[c1_weak], [c1_strong, c2]]

    mock_expander = MagicMock()
    mock_expander.expand.return_value = ["q1", "q2"]

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        query_expander=mock_expander,
    )
    response = pipeline.ask("q1")

    # Should deduplicate c1 and keep c1_strong (distance 0.1)
    chunks = response.retrieved_chunks
    assert len(chunks) == 2
    chunk_map = {c.chunk_id: c for c in chunks}
    assert chunk_map["c1"].distance == 0.1
    assert chunk_map["c1"].text == "content 1 strong"


def test_rag_pipeline_query_expansion_with_reranking():
    retriever = MagicMock()
    generator = MagicMock()
    generator.generate.return_value = "Answer"
    reranker = MagicMock()

    c1 = make_chunk("c1", "chunk one", distance=0.2)
    c2 = make_chunk("c2", "chunk two", distance=0.3)
    retriever.retrieve.side_effect = [[c1], [c2]]
    reranker.rerank.return_value = [c2, c1]

    mock_expander = MagicMock()
    mock_expander.expand.return_value = ["query A", "query B"]

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        reranker=reranker,
        reranker_candidate_multiplier=3,
        query_expander=mock_expander,
    )
    response = pipeline.ask("query A", top_k=2)

    # Retriever called with candidate_k = top_k * multiplier = 2 * 3 = 6
    assert retriever.retrieve.call_args_list[0].kwargs["top_k"] == 6
    assert retriever.retrieve.call_args_list[1].kwargs["top_k"] == 6

    # Reranker receives deduplicated candidates and final_k=2
    reranker.rerank.assert_called_once()
    assert reranker.rerank.call_args.kwargs["top_k"] == 2
    assert response.retrieved_chunks == [c2, c1]
    assert response.candidates_retrieved == 2
    assert response.reranking_enabled is True
    assert response.query_expansion_enabled is True
