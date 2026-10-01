"""
Unit tests for ContextIQ Chat UI logic, conversational message tracking,
and per-turn source/retrieval isolation.
"""

from unittest.mock import MagicMock
import pytest

from app.rag.conversation import Conversation
from app.rag.pipeline import RAGPipeline, RAGPipelineError, RAGResponse
from app.retrieval.retriever import RetrievedChunk
from app.main import handle_chat_turn
from app.vectorstore.chroma_store import ChromaVectorStore
from app.ingestion.chunker import DocumentChunk

FAKE_DIM = 768


def make_vector(val: float = 0.1, dim: int = FAKE_DIM) -> list:
    return [float(val)] * dim


# ---------------------------------------------------------------------------
# Test 1 — Conversation history rendering data
# ---------------------------------------------------------------------------
def test_conversation_history_rendering_data():
    """Verify sequential chronological ordering and roles of conversation turns."""
    conv = Conversation()
    conv.add_user_message("What is RAG?")
    conv.add_assistant_message("RAG is Retrieval-Augmented Generation.")
    conv.add_user_message("What are its key benefits?")
    conv.add_assistant_message("It grounds LLM responses with external knowledge.")

    messages = conv.get_messages()
    assert len(messages) == 4
    assert messages[0].role == "user"
    assert messages[0].content == "What is RAG?"
    assert messages[1].role == "assistant"
    assert "Retrieval-Augmented Generation" in messages[1].content
    assert messages[2].role == "user"
    assert messages[2].content == "What are its key benefits?"
    assert messages[3].role == "assistant"
    assert "grounds LLM responses" in messages[3].content


# ---------------------------------------------------------------------------
# Test 2 — Successful RAG turn
# ---------------------------------------------------------------------------
def test_successful_rag_turn():
    """Verify handle_chat_turn records both user and assistant messages with metadata."""
    conv = Conversation()
    mock_pipeline = MagicMock(spec=RAGPipeline)

    chunks = [
        RetrievedChunk("c1", "RAG chunk 1", "paper.pdf", 1, 0, 0.12),
        RetrievedChunk("c2", "RAG chunk 2", "paper.pdf", 2, 0, 0.24),
    ]
    sources = [{"source": "paper.pdf", "page": 1}, {"source": "paper.pdf", "page": 2}]
    response = RAGResponse(
        answer="RAG retrieves relevant passages before answering.",
        sources=sources,
        retrieved_chunks=chunks,
        query="Explain RAG",
    )
    mock_pipeline.ask.return_value = response

    res = handle_chat_turn("Explain RAG", mock_pipeline, conv, top_k=2)

    assert res == response
    messages = conv.get_messages()
    assert len(messages) == 2

    # User message
    assert messages[0].role == "user"
    assert messages[0].content == "Explain RAG"

    # Assistant message
    assert messages[1].role == "assistant"
    assert messages[1].content == "RAG retrieves relevant passages before answering."
    assert messages[1].sources == sources
    assert len(messages[1].retrieved_chunks) == 2


# ---------------------------------------------------------------------------
# Test 3 — RAG error does not create fake assistant answer
# ---------------------------------------------------------------------------
def test_rag_error_does_not_corrupt_history():
    """Verify that when pipeline.ask raises an error, no fake assistant answer is added."""
    conv = Conversation()
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.ask.side_effect = RAGPipelineError("ChromaDB query failure")

    with pytest.raises(RAGPipelineError, match="ChromaDB query failure"):
        handle_chat_turn("Failing question", mock_pipeline, conv)

    # Conversation history must remain clean without fake assistant responses
    assert len(conv) == 0
    assert conv.get_messages() == []


# ---------------------------------------------------------------------------
# Test 4 — Sources and retrieved context remain associated per turn
# ---------------------------------------------------------------------------
def test_sources_and_context_isolated_per_turn():
    """Verify multiple turns retain distinct sources and retrieved chunks without overwriting."""
    conv = Conversation()
    mock_pipeline = MagicMock(spec=RAGPipeline)

    # Turn 1
    chunks_1 = [RetrievedChunk("c1", "Attention is all you need.", "attention.pdf", 1, 0, 0.1)]
    sources_1 = [{"source": "attention.pdf", "page": 1}]
    resp_1 = RAGResponse(
        answer="Transformers rely on multi-head attention.",
        sources=sources_1,
        retrieved_chunks=chunks_1,
        query="What is attention?",
    )

    # Turn 2
    chunks_2 = [RetrievedChunk("c2", "Vector indexing with cosine.", "vector_store.pdf", 3, 0, 0.05)]
    sources_2 = [{"source": "vector_store.pdf", "page": 3}]
    resp_2 = RAGResponse(
        answer="Cosine similarity evaluates angle between vectors.",
        sources=sources_2,
        retrieved_chunks=chunks_2,
        query="What is cosine similarity?",
    )

    mock_pipeline.ask.side_effect = [resp_1, resp_2]

    handle_chat_turn("What is attention?", mock_pipeline, conv)
    handle_chat_turn("What is cosine similarity?", mock_pipeline, conv)

    messages = conv.get_messages()
    assert len(messages) == 4

    # Turn 1 assistant message has Turn 1 metadata
    assert messages[1].sources == sources_1
    assert messages[1].retrieved_chunks[0].chunk_id == "c1"

    # Turn 2 assistant message has Turn 2 metadata
    assert messages[3].sources == sources_2
    assert messages[3].retrieved_chunks[0].chunk_id == "c2"


# ---------------------------------------------------------------------------
# Test 5 — Clear conversation removes chat messages without touching vector store
# ---------------------------------------------------------------------------
def test_clear_conversation_preserves_vector_store(tmp_path):
    """Verify clearing conversation removes messages while leaving ChromaDB intact."""
    store = ChromaVectorStore(persist_dir=tmp_path / "test_clear_conv")
    chunk = DocumentChunk("c1", "Vector text", "doc.pdf", 1, 0, "doc_hash")
    store.add_chunks([chunk], [make_vector(0.1)])
    assert store.count() == 1

    conv = Conversation()
    conv.add_user_message("Query")
    conv.add_assistant_message("Answer")
    assert len(conv) == 2

    # Clear conversation only
    conv.clear()
    assert len(conv) == 0
    assert conv.get_messages() == []

    # Vector store remains untouched
    assert store.count() == 1
    assert store.has_document("doc_hash") is True


# ---------------------------------------------------------------------------
# Test 6 — Follow-up query is passed unchanged to pipeline
# ---------------------------------------------------------------------------
def test_follow_up_query_passed_verbatim():
    """Verify follow-up question is passed literally to pipeline.ask without history rewriting."""
    conv = Conversation()
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.ask.return_value = RAGResponse(
        answer="Synthesized answer",
        sources=[],
        retrieved_chunks=[],
        query="Why is it useful?",
    )

    # First turn
    conv.add_user_message("What is RAG?")
    conv.add_assistant_message("RAG is Retrieval-Augmented Generation.")

    # Second turn (follow-up)
    handle_chat_turn("Why is it useful?", mock_pipeline, conv, top_k=5)

    # Must be called with the exact literal string "Why is it useful?"
    mock_pipeline.ask.assert_called_once_with("Why is it useful?", top_k=5)
