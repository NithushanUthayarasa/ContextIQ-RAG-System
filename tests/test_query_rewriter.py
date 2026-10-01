"""
Unit tests for ContextIQ QueryRewriter and context-aware query reformulation.
Tests verify standalone question handling, pronoun resolution, history formatting,
error fallback, and proper delegation through RAGPipeline without calling the real Gemini API.
"""

from unittest.mock import MagicMock
import pytest

from app.generation.generator import GeminiGenerator
from app.rag.conversation import ChatMessage, Conversation
from app.rag.pipeline import RAGPipeline, RAGResponse
from app.rag.query_rewriter import QueryRewriter
from app.retrieval.retriever import RetrievedChunk, Retriever
from app.main import handle_chat_turn


@pytest.fixture
def mock_gemini_client():
    """Provides a mock Google GenAI client returning a mock response."""
    client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Standalone query text"
    client.models.generate_content.return_value = mock_resp
    return client


# ---------------------------------------------------------------------------
# Test 1 — Standalone question
# ---------------------------------------------------------------------------
def test_standalone_question_rewrite(mock_gemini_client):
    """Verify standalone questions remain intact without unnecessary alteration."""
    mock_gemini_client.models.generate_content.return_value.text = "What is vector database indexing?"
    rewriter = QueryRewriter(client=mock_gemini_client)

    history = [ChatMessage(role="user", content="Hi"), ChatMessage(role="assistant", content="Hello!")]
    output = rewriter.rewrite_query("What is vector database indexing?", conversation_messages=history)

    assert output == "What is vector database indexing?"
    mock_gemini_client.models.generate_content.assert_called_once()


# ---------------------------------------------------------------------------
# Test 2 — Follow-up question
# ---------------------------------------------------------------------------
def test_follow_up_question_rewrite(mock_gemini_client):
    """Verify follow-up question resolves context from previous turns."""
    mock_gemini_client.models.generate_content.return_value.text = (
        "What are the benefits of Retrieval-Augmented Generation?"
    )
    rewriter = QueryRewriter(client=mock_gemini_client)

    history = [
        ChatMessage(role="user", content="What is RAG?"),
        ChatMessage(role="assistant", content="RAG retrieves relevant information before generation."),
    ]
    output = rewriter.rewrite_query("What are its benefits?", conversation_messages=history)

    assert output == "What are the benefits of Retrieval-Augmented Generation?"


# ---------------------------------------------------------------------------
# Test 3 — Pronoun resolution
# ---------------------------------------------------------------------------
def test_pronoun_resolution(mock_gemini_client):
    """Verify ambiguous pronouns (e.g. 'it') are resolved into entity names."""
    mock_gemini_client.models.generate_content.return_value.text = (
        "How does ChromaDB store embeddings?"
    )
    rewriter = QueryRewriter(client=mock_gemini_client)

    history = [
        ChatMessage(role="user", content="What is ChromaDB?"),
        ChatMessage(role="assistant", content="ChromaDB is a lightweight persistent vector database."),
    ]
    output = rewriter.rewrite_query("How does it store embeddings?", conversation_messages=history)

    assert output == "How does ChromaDB store embeddings?"


# ---------------------------------------------------------------------------
# Test 4 — No history bypasses rewriting call
# ---------------------------------------------------------------------------
def test_no_history_bypasses_gemini_call(mock_gemini_client):
    """Verify that when conversation messages list is empty, Gemini is not called."""
    rewriter = QueryRewriter(client=mock_gemini_client)

    output = rewriter.rewrite_query("What is cosine similarity?", conversation_messages=[])

    assert output == "What is cosine similarity?"
    mock_gemini_client.models.generate_content.assert_not_called()

    # Also test None
    output_none = rewriter.rewrite_query("What is cosine similarity?", conversation_messages=None)
    assert output_none == "What is cosine similarity?"
    mock_gemini_client.models.generate_content.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5 — Gemini failure fallback
# ---------------------------------------------------------------------------
def test_gemini_failure_fallback(mock_gemini_client):
    """Verify that when Gemini raises an exception, the original question is returned."""
    mock_gemini_client.models.generate_content.side_effect = Exception("API quota exceeded")
    rewriter = QueryRewriter(client=mock_gemini_client)

    history = [ChatMessage(role="user", content="Previous question")]
    output = rewriter.rewrite_query("What are its benefits?", conversation_messages=history)

    assert output == "What are its benefits?"


# ---------------------------------------------------------------------------
# Test 6 — Empty Gemini response fallback
# ---------------------------------------------------------------------------
def test_empty_gemini_response_fallback(mock_gemini_client):
    """Verify that when Gemini returns empty text, the original question is returned."""
    mock_gemini_client.models.generate_content.return_value.text = "   "
    rewriter = QueryRewriter(client=mock_gemini_client)

    history = [ChatMessage(role="user", content="Previous question")]
    output = rewriter.rewrite_query("What are its benefits?", conversation_messages=history)

    assert output == "What are its benefits?"


# ---------------------------------------------------------------------------
# Test 7 — Original user message preserved in conversation
# ---------------------------------------------------------------------------
def test_original_user_message_preserved():
    """
    Verify that in handle_chat_turn, the user message recorded in conversation
    remains the original submitted query, not the rewritten query.
    """
    conv = Conversation()
    conv.add_user_message("What is RAG?")
    conv.add_assistant_message("RAG is Retrieval-Augmented Generation.")

    mock_rewriter = MagicMock(spec=QueryRewriter)
    mock_rewriter.rewrite_query.return_value = "What are the benefits of Retrieval-Augmented Generation?"

    mock_retriever = MagicMock(spec=Retriever)
    mock_retriever.retrieve.return_value = [
        RetrievedChunk("c1", "RAG benefits text", "paper.pdf", 1, 0, 0.1)
    ]

    mock_generator = MagicMock(spec=GeminiGenerator)
    mock_generator.generate.return_value = "The benefits are improved accuracy and grounding."

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        query_rewriter=mock_rewriter,
    )

    handle_chat_turn("What are its benefits?", pipeline, conv)

    messages = conv.get_messages()
    # Turn 2 user message must be exactly the user's literal prompt
    assert messages[2].role == "user"
    assert messages[2].content == "What are its benefits?"


# ---------------------------------------------------------------------------
# Test 8 — Retrieval receives rewritten query while generator receives original
# ---------------------------------------------------------------------------
def test_retrieval_receives_rewritten_query():
    """Verify retriever receives rewritten standalone query and generator receives user's question."""
    mock_rewriter = MagicMock(spec=QueryRewriter)
    mock_rewriter.rewrite_query.return_value = "What are the benefits of Retrieval-Augmented Generation?"

    mock_retriever = MagicMock(spec=Retriever)
    mock_retriever.retrieve.return_value = [
        RetrievedChunk("c1", "Text on RAG benefits", "rag.pdf", 2, 0, 0.15)
    ]

    mock_generator = MagicMock(spec=GeminiGenerator)
    mock_generator.generate.return_value = "Grounding and reduced hallucination."

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        query_rewriter=mock_rewriter,
    )

    history = [ChatMessage(role="user", content="What is RAG?")]
    response = pipeline.ask(
        "What are its benefits?",
        top_k=3,
        conversation_messages=history,
    )

    # Retriever must have been queried with the rewritten standalone query
    mock_retriever.retrieve.assert_called_once_with(
        query="What are the benefits of Retrieval-Augmented Generation?",
        top_k=3,
    )

    # Generator must be passed the original user question
    mock_generator.generate.assert_called_once_with(
        question="What are its benefits?",
        retrieved_chunks=mock_retriever.retrieve.return_value,
    )

    # Response should record both
    assert response.query == "What are its benefits?"
    assert response.retrieval_query == "What are the benefits of Retrieval-Augmented Generation?"


# ---------------------------------------------------------------------------
# Test 9 — First query bypasses rewriting in pipeline
# ---------------------------------------------------------------------------
def test_first_query_bypasses_rewriting_in_pipeline():
    """Verify that when no conversation history is provided to pipeline.ask, rewriter is not called."""
    mock_rewriter = MagicMock(spec=QueryRewriter)
    mock_retriever = MagicMock(spec=Retriever)
    mock_retriever.retrieve.return_value = []
    mock_generator = MagicMock(spec=GeminiGenerator)
    mock_generator.generate.return_value = "Answer text."

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        query_rewriter=mock_rewriter,
    )

    # First query (no conversation messages)
    response = pipeline.ask("What is RAG?", conversation_messages=[])

    mock_rewriter.rewrite_query.assert_not_called()
    mock_retriever.retrieve.assert_called_once_with(query="What is RAG?", top_k=None)
    assert response.query == "What is RAG?"
    assert response.retrieval_query == "What is RAG?"
