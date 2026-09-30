"""
Unit tests for ContextIQ GeminiEmbedder.
Tests use mocked Google GenAI client responses to ensure fast, deterministic offline execution.
"""

from unittest.mock import MagicMock, patch
import pytest

from app.ingestion.chunker import DocumentChunk
from app.ingestion.embedder import (
    GeminiEmbedder,
    GeminiEmbedderError,
    MissingAPIKeyError,
    EmptyInputError,
)


@pytest.fixture
def mock_genai_client():
    """Provides a mock Google GenAI client with fake 768-dim embeddings."""
    with patch("app.ingestion.embedder.genai.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance

        # Default mock response for embed_content
        def fake_embed_content(model, contents, config):
            is_list = isinstance(contents, list)
            count = len(contents) if is_list else 1
            mock_resp = MagicMock()
            fake_embs = []
            for _ in range(count):
                fake_emb = MagicMock()
                # 768-dimensional fake vector
                fake_emb.values = [0.1 * (i % 10) for i in range(768)]
                fake_embs.append(fake_emb)
            mock_resp.embeddings = fake_embs
            return mock_resp

        mock_instance.models.embed_content.side_effect = fake_embed_content
        yield mock_instance


def test_embedder_missing_api_key():
    """Verify that MissingAPIKeyError is raised if api_key is None or empty."""
    with pytest.raises(MissingAPIKeyError, match="Gemini API key is missing or invalid"):
        GeminiEmbedder(api_key="")

    with pytest.raises(MissingAPIKeyError, match="Gemini API key is missing or invalid"):
        GeminiEmbedder(api_key="your_gemini_api_key_here")


def test_embedder_configuration_loading(mock_genai_client):
    """Verify configuration loads default model and dimension."""
    embedder = GeminiEmbedder(api_key="valid-mock-key")
    assert embedder.model_name == "gemini-embedding-001"
    assert embedder.dimension == 768


def test_embed_documents_empty_list(mock_genai_client):
    """Verify that empty document list returns an empty list without calling API."""
    embedder = GeminiEmbedder(api_key="valid-mock-key")
    result = embedder.embed_documents([])
    assert result == []
    mock_genai_client.models.embed_content.assert_not_called()


def test_embed_query_empty_error(mock_genai_client):
    """Verify that empty or whitespace-only query raises EmptyInputError."""
    embedder = GeminiEmbedder(api_key="valid-mock-key")

    with pytest.raises(EmptyInputError, match="Search query must not be empty"):
        embedder.embed_query("")

    with pytest.raises(EmptyInputError, match="Search query must not be empty"):
        embedder.embed_query("   \n\t  ")


def test_embed_documents_success(mock_genai_client):
    """Verify embedding generation for a list of DocumentChunks."""
    chunks = [
        DocumentChunk("doc_p1_c0", "Text chunk 1", "doc.pdf", 1, 0),
        DocumentChunk("doc_p1_c1", "Text chunk 2", "doc.pdf", 1, 1),
        DocumentChunk("doc_p2_c0", "Text chunk 3", "doc.pdf", 2, 0),
    ]
    embedder = GeminiEmbedder(api_key="valid-mock-key")
    embeddings = embedder.embed_documents(chunks)

    assert len(embeddings) == 3
    for emb in embeddings:
        assert isinstance(emb, list)
        assert len(emb) == 768
        assert all(isinstance(v, float) for v in emb)


def test_embed_query_success(mock_genai_client):
    """Verify embedding generation for a single search query."""
    embedder = GeminiEmbedder(api_key="valid-mock-key")
    vec = embedder.embed_query("What is Retrieval-Augmented Generation?")

    assert isinstance(vec, list)
    assert len(vec) == 768
    assert all(isinstance(v, float) for v in vec)
    mock_genai_client.models.embed_content.assert_called_once()


def test_embedder_batching_behavior(mock_genai_client):
    """Verify that chunks exceeding batch_size trigger multiple batch API requests."""
    chunks = [
        DocumentChunk(f"doc_p1_c{i}", f"Text chunk {i}", "doc.pdf", 1, i)
        for i in range(5)
    ]
    embedder = GeminiEmbedder(api_key="valid-mock-key")
    # batch size of 2 -> 3 API calls (2 + 2 + 1)
    embeddings = embedder.embed_documents(chunks, batch_size=2)

    assert len(embeddings) == 5
    assert mock_genai_client.models.embed_content.call_count == 3


def test_embedder_api_error_handling(mock_genai_client):
    """Verify that SDK errors are cleanly wrapped in GeminiEmbedderError."""
    mock_genai_client.models.embed_content.side_effect = Exception("Google GenAI Rate limit exceeded")
    embedder = GeminiEmbedder(api_key="valid-mock-key")

    with pytest.raises(GeminiEmbedderError, match="Gemini Embedding API call failed"):
        embedder.embed_query("Hello test")
