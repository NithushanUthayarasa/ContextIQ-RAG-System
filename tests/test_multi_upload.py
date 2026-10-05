"""
Unit and integration tests for multi-PDF ingestion in ContextIQ.
Tests verify SHA-256 identity calculation, duplicate skipping, isolated staging,
same-name different-content coexistence, same-content different-name deduplication,
and partial failure isolation using mock embedders and isolated temporary directories.
"""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import pymupdf

from app.ingestion.embedder import GeminiEmbedder
from app.ingestion.pdf_loader import InvalidPDFError, ScannedOrEmptyPDFError
from app.main import ingest_pdf_bytes
from app.vectorstore.chroma_store import ChromaVectorStore

FAKE_DIM = 768


def make_vector(val: float = 0.1, dim: int = FAKE_DIM) -> list:
    return [float(val)] * dim


def create_pdf_bytes(text: str) -> bytes:
    """Creates in-memory valid PDF bytes containing text."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_blank_pdf_bytes() -> bytes:
    """Creates an empty 1-page PDF without any extractable text."""
    doc = pymupdf.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def mock_embedder():
    """Mock GeminiEmbedder returning predictable vectors matching input chunks count."""
    embedder = MagicMock(spec=GeminiEmbedder)
    embedder.embed_documents.side_effect = lambda chunks: [make_vector(0.1)] * len(chunks)
    return embedder


@pytest.fixture
def isolated_store(tmp_path: Path):
    """Temporary ChromaVectorStore instance."""
    return ChromaVectorStore(
        persist_dir=tmp_path / "test_multi_upload_chroma",
        collection_name="test_multi_upload",
    )


# ---------------------------------------------------------------------------
# Test 1 — Content-based SHA-256 document_id calculation
# ---------------------------------------------------------------------------
def test_sha256_document_id_calculation(tmp_path: Path, mock_embedder, isolated_store):
    """Verify document_id equals exact SHA-256 of file bytes."""
    content = "ContextIQ multi-document identity test content."
    pdf_bytes = create_pdf_bytes(content)
    expected_hash = hashlib.sha256(pdf_bytes).hexdigest()

    result = ingest_pdf_bytes(
        file_name="sample.pdf",
        file_bytes=pdf_bytes,
        vector_store=isolated_store,
        embedder=mock_embedder,
        upload_dir=tmp_path / "uploads",
    )

    assert result["status"] == "indexed"
    assert result["document_id"] == expected_hash
    assert isolated_store.has_document(expected_hash) is True


# ---------------------------------------------------------------------------
# Test 2 — Duplicate detection BEFORE saving and embedding
# ---------------------------------------------------------------------------
def test_duplicate_skipping_before_saving(tmp_path: Path, mock_embedder, isolated_store):
    """
    Verify duplicate content is detected via has_document() BEFORE saving to disk
    or calling embedder.
    """
    pdf_bytes = create_pdf_bytes("Unique content to test duplicate skipping.")
    upload_dir = tmp_path / "uploads"

    # First upload
    res1 = ingest_pdf_bytes(
        file_name="report.pdf",
        file_bytes=pdf_bytes,
        vector_store=isolated_store,
        embedder=mock_embedder,
        upload_dir=upload_dir,
    )
    assert res1["status"] == "indexed"
    assert mock_embedder.embed_documents.call_count == 1

    # Second upload with same bytes
    res2 = ingest_pdf_bytes(
        file_name="report.pdf",
        file_bytes=pdf_bytes,
        vector_store=isolated_store,
        embedder=mock_embedder,
        upload_dir=upload_dir,
    )
    assert res2["status"] == "skipped"
    assert res2["message"] == "Already indexed"
    # Embedder was NOT called again
    assert mock_embedder.embed_documents.call_count == 1


# ---------------------------------------------------------------------------
# Test 3 — Safe staging path based on document_id while preserving source
# ---------------------------------------------------------------------------
def test_safe_staging_path_and_source_preservation(tmp_path: Path, mock_embedder, isolated_store):
    """
    Verify upload is stored on disk as {document_id}.pdf rather than original filename,
    while ChromaDB metadata preserves the human-readable original filename as source.
    """
    pdf_bytes = create_pdf_bytes("Testing safe staging paths.")
    expected_hash = hashlib.sha256(pdf_bytes).hexdigest()
    upload_dir = tmp_path / "uploads"

    result = ingest_pdf_bytes(
        file_name="Quarterly Financial Report 2026.pdf",
        file_bytes=pdf_bytes,
        vector_store=isolated_store,
        embedder=mock_embedder,
        upload_dir=upload_dir,
    )

    assert result["status"] == "indexed"
    staged_file = upload_dir / f"{expected_hash}.pdf"
    assert staged_file.exists()
    assert not (upload_dir / "Quarterly Financial Report 2026.pdf").exists()

    # ChromaDB metadata should preserve original human-readable source
    docs = isolated_store.list_indexed_documents()
    assert len(docs) == 1
    assert docs[0]["source"] == "Quarterly Financial Report 2026.pdf"
    assert docs[0]["document_id"] == expected_hash


# ---------------------------------------------------------------------------
# Test 4 — Same filename with different content
# ---------------------------------------------------------------------------
def test_same_filename_different_content_coexist(tmp_path: Path, mock_embedder, isolated_store):
    """
    Verify two different PDFs both named 'report.pdf' are both indexed and coexist
    in separate staging paths and ChromaDB records without collision.
    """
    bytes_a = create_pdf_bytes("Report content version A.")
    bytes_b = create_pdf_bytes("Report content version B with different text.")
    upload_dir = tmp_path / "uploads"

    hash_a = hashlib.sha256(bytes_a).hexdigest()
    hash_b = hashlib.sha256(bytes_b).hexdigest()
    assert hash_a != hash_b

    res_a = ingest_pdf_bytes(
        file_name="report.pdf",
        file_bytes=bytes_a,
        vector_store=isolated_store,
        embedder=mock_embedder,
        upload_dir=upload_dir,
    )
    res_b = ingest_pdf_bytes(
        file_name="report.pdf",
        file_bytes=bytes_b,
        vector_store=isolated_store,
        embedder=mock_embedder,
        upload_dir=upload_dir,
    )

    assert res_a["status"] == "indexed"
    assert res_b["status"] == "indexed"

    assert (upload_dir / f"{hash_a}.pdf").exists()
    assert (upload_dir / f"{hash_b}.pdf").exists()

    docs = isolated_store.list_indexed_documents()
    assert len(docs) == 2
    assert {d["document_id"] for d in docs} == {hash_a, hash_b}
    for d in docs:
        assert d["source"] == "report.pdf"


# ---------------------------------------------------------------------------
# Test 5 — Same content with different filenames
# ---------------------------------------------------------------------------
def test_same_content_different_filename_skipped(tmp_path: Path, mock_embedder, isolated_store):
    """
    Verify that renaming a file does not re-index content that has already been indexed.
    """
    shared_bytes = create_pdf_bytes("Identical content in both files.")
    upload_dir = tmp_path / "uploads"

    res1 = ingest_pdf_bytes(
        file_name="original.pdf",
        file_bytes=shared_bytes,
        vector_store=isolated_store,
        embedder=mock_embedder,
        upload_dir=upload_dir,
    )
    res2 = ingest_pdf_bytes(
        file_name="renamed_copy.pdf",
        file_bytes=shared_bytes,
        vector_store=isolated_store,
        embedder=mock_embedder,
        upload_dir=upload_dir,
    )

    assert res1["status"] == "indexed"
    assert res2["status"] == "skipped"

    docs = isolated_store.list_indexed_documents()
    assert len(docs) == 1


# ---------------------------------------------------------------------------
# Test 6 — Multi-file processing with partial failure
# ---------------------------------------------------------------------------
def test_multi_file_partial_failure_isolation(tmp_path: Path, mock_embedder, isolated_store):
    """
    Verify that when processing multiple files, one failure (e.g. invalid PDF)
    does not disrupt the remaining valid files.
    """
    valid_bytes_1 = create_pdf_bytes("Valid document 1.")
    corrupt_bytes = b"Not a valid PDF header or content at all."
    valid_bytes_2 = create_pdf_bytes("Valid document 2.")

    files = [
        ("doc1.pdf", valid_bytes_1),
        ("corrupt.pdf", corrupt_bytes),
        ("doc2.pdf", valid_bytes_2),
    ]

    outcomes = []
    upload_dir = tmp_path / "uploads"

    for name, content in files:
        try:
            res = ingest_pdf_bytes(
                file_name=name,
                file_bytes=content,
                vector_store=isolated_store,
                embedder=mock_embedder,
                upload_dir=upload_dir,
            )
            outcomes.append(res["status"])
        except InvalidPDFError:
            outcomes.append("failed")

    assert outcomes == ["indexed", "failed", "indexed"]
    assert isolated_store.has_document(hashlib.sha256(valid_bytes_1).hexdigest()) is True
    assert isolated_store.has_document(hashlib.sha256(valid_bytes_2).hexdigest()) is True
    assert len(isolated_store.list_indexed_documents()) == 2


# ---------------------------------------------------------------------------
# Test 7 — Scanned or empty PDF error handling
# ---------------------------------------------------------------------------
def test_scanned_or_empty_pdf_raises_cleanly(tmp_path: Path, mock_embedder, isolated_store):
    """Verify that image-only or blank PDF raises ScannedOrEmptyPDFError."""
    blank_bytes = create_blank_pdf_bytes()
    upload_dir = tmp_path / "uploads"

    with pytest.raises(ScannedOrEmptyPDFError, match="contains no extractable text"):
        ingest_pdf_bytes(
            file_name="empty.pdf",
            file_bytes=blank_bytes,
            vector_store=isolated_store,
            embedder=mock_embedder,
            upload_dir=upload_dir,
        )
