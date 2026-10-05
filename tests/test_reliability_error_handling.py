"""
Unit tests for ContextIQ Error Handling, Reliability, and Secret/Path Sanitization.
Tests user-safe error translation, secret redaction, path sanitization,
specialized PDF exceptions, and ingestion rollback behavior.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.ingestion.pdf_loader import (
    PDFLoader,
    PDFNotFoundError,
    InvalidPDFError,
    EmptyPDFError,
    EncryptedPDFError,
    CorruptPDFError,
    ScannedOrEmptyPDFError,
)
from app.ingestion.embedder import MissingAPIKeyError, GeminiEmbedderError
from app.vectorstore.chroma_store import VectorStoreError
from app.utils.error_handler import (
    sanitize_error_message,
    translate_exception_to_user_message,
    safe_log_exception,
)
from app.main import ingest_pdf_bytes


# ==============================================================================
# 1. Sanitization & Redaction Tests
# ==============================================================================

def test_sanitize_redacts_api_key():
    raw_error = "Error calling Google API with key AIzaSyD1234567890abcdefghijklmnopqrstuv: 403 Forbidden"
    sanitized = sanitize_error_message(raw_error)
    assert "AIzaSyD" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized


def test_sanitize_redacts_windows_paths():
    raw_error = r"Failed to open C:\Users\uthay\Desktop\ContextIQ-RAG\data\uploads\doc.pdf: FileDataError"
    sanitized = sanitize_error_message(raw_error)
    assert r"C:\Users\uthay" not in sanitized
    assert "[local path]" in sanitized


def test_sanitize_redacts_unix_paths():
    raw_error = "File not found at /home/user/app/data/uploads/document.pdf in system"
    sanitized = sanitize_error_message(raw_error)
    assert "/home/user" not in sanitized
    assert "[local path]" in sanitized


def test_sanitize_handles_none_or_empty():
    assert sanitize_error_message(None) == ""
    assert sanitize_error_message("") == ""
    assert sanitize_error_message("   ") == ""


# ==============================================================================
# 2. User-Friendly Error Translation Tests (9 Categories)
# ==============================================================================

def test_translate_missing_api_key():
    err = MissingAPIKeyError("GEMINI_API_KEY environment variable is missing or empty.")
    msg = translate_exception_to_user_message(err)
    assert "Gemini API key is not configured" in msg


def test_translate_authentication_error():
    err = Exception("403 PERMISSION_DENIED: The caller does not have permission")
    msg = translate_exception_to_user_message(err)
    assert "Authentication failed" in msg


def test_translate_rate_limit_error():
    err = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for quota metric")
    msg = translate_exception_to_user_message(err)
    assert "rate limit or quota exceeded" in msg


def test_translate_service_unavailable_error():
    err = Exception("503 UNAVAILABLE: The service is currently unavailable")
    msg = translate_exception_to_user_message(err)
    assert "temporarily unavailable" in msg


def test_translate_scanned_pdf_error():
    err = ScannedOrEmptyPDFError("The PDF contains no extractable text.")
    msg = translate_exception_to_user_message(err)
    assert "Scanned or image-only documents require OCR" in msg


def test_translate_encrypted_pdf_error():
    err = EncryptedPDFError("Document is password-protected or encrypted.")
    msg = translate_exception_to_user_message(err)
    assert "password-protected or encrypted" in msg


def test_translate_empty_pdf_error():
    err = EmptyPDFError("File is empty (0 bytes).")
    msg = translate_exception_to_user_message(err)
    assert "empty (0 bytes or 0 pages)" in msg


def test_translate_corrupt_pdf_error():
    err = CorruptPDFError("Failed to open PDF: FileDataError: cannot find document header")
    msg = translate_exception_to_user_message(err)
    assert "corrupted or is not a valid PDF" in msg


def test_translate_vectorstore_error():
    err = VectorStoreError("ChromaDB sqlite connection lock error")
    msg = translate_exception_to_user_message(err)
    assert "vector database encountered an error" in msg


def test_translate_context_fallback():
    err = RuntimeError("Unknown internal calculation failure")
    msg_embed = translate_exception_to_user_message(err, context="embedding")
    assert "Failed to generate document embeddings" in msg_embed

    msg_gen = translate_exception_to_user_message(err, context="generation")
    assert "Failed to generate an answer" in msg_gen

    msg_generic = translate_exception_to_user_message(err)
    assert "unexpected error occurred" in msg_generic


# ==============================================================================
# 3. PDF Loader Edge Case Tests
# ==============================================================================

def test_empty_zero_byte_pdf(tmp_path: Path):
    empty_pdf = tmp_path / "zero_bytes.pdf"
    empty_pdf.write_bytes(b"")

    loader = PDFLoader(empty_pdf)
    with pytest.raises(EmptyPDFError) as exc_info:
        loader.load()

    assert "empty (0 bytes)" in str(exc_info.value)
    # Ensure raw filesystem directory is NOT in the error message
    assert str(tmp_path) not in str(exc_info.value)


def test_corrupt_pdf_file_raises_corrupt_error(tmp_path: Path):
    corrupt_pdf = tmp_path / "corrupt_data.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4\nGARBAGE_PAYLOAD_NOT_A_REAL_PDF")

    loader = PDFLoader(corrupt_pdf)
    with pytest.raises((CorruptPDFError, InvalidPDFError)) as exc_info:
        loader.load()

    assert issubclass(CorruptPDFError, InvalidPDFError)
    assert str(tmp_path) not in str(exc_info.value)


def test_encrypted_pdf_handling(tmp_path: Path):
    mock_doc = MagicMock()
    mock_doc.is_encrypted = True
    mock_doc.needs_pass = True

    dummy_pdf = tmp_path / "protected.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\nMock encrypted PDF content")

    loader = PDFLoader(dummy_pdf)
    with patch("pymupdf.open", return_value=mock_doc):
        with pytest.raises(EncryptedPDFError) as exc_info:
            loader.load()

    assert "password-protected or encrypted" in str(exc_info.value)
    mock_doc.close.assert_called_once()


def test_safe_log_exception_redaction(caplog):
    test_logger = logging.getLogger("test_logger")
    exc = Exception(r"Failed AIzaSyD9876543210ZYXWVUTSRQPONMLKJHGFEDCBA on C:\Users\secret\path\file.pdf")

    with caplog.at_level(logging.ERROR, logger="test_logger"):
        safe_log_exception(test_logger, "TestContext", exc)

    log_output = caplog.text
    assert "AIzaSyD" not in log_output
    assert "[REDACTED_API_KEY]" in log_output
    assert r"C:\Users\secret" not in log_output
    assert "[local path]" in log_output


# ==============================================================================
# 4. Ingestion Rollback Tests
# ==============================================================================

def test_ingest_pdf_bytes_rollback_on_embedding_failure(tmp_path: Path):
    """
    Verifies that if embedding generation fails after parent chunks are added,
    the ingestion helper rolls back and deletes the document from the vector store,
    leaving no orphaned chunks or staging files.
    """
    mock_store = MagicMock()
    mock_store.has_document.return_value = False

    mock_embedder = MagicMock()
    mock_embedder.embed_documents.side_effect = GeminiEmbedderError("Network timeout during embedding")

    fake_pdf_bytes = b"%PDF-1.4 mock valid pdf bytes"

    # Mock PDFLoader to return 1 page
    mock_loader = MagicMock()
    mock_page = MagicMock()
    mock_page.text = "Valid document text for testing rollback."
    mock_page.page_number = 1
    mock_page.source = "test.pdf"

    mock_load_result = MagicMock()
    mock_load_result.pages = [mock_page]
    mock_load_result.non_empty_page_count = 1
    mock_load_result.total_pages = 1
    mock_load_result.empty_pages = []
    mock_loader.load.return_value = mock_load_result

    with patch("app.main.PDFLoader", return_value=mock_loader):
        with pytest.raises(GeminiEmbedderError):
            ingest_pdf_bytes(
                file_name="test.pdf",
                file_bytes=fake_pdf_bytes,
                vector_store=mock_store,
                embedder=mock_embedder,
                upload_dir=tmp_path,
            )

    # Rollback assertion: delete_by_document_id MUST have been called
    mock_store.delete_by_document_id.assert_called_once()
