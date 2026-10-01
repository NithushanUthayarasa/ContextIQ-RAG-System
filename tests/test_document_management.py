"""
Unit tests for ContextIQ Document Management functionality.
Tests verify listing indexed documents, handling identical filenames across
distinct document IDs, isolated document deletion, accurate page/chunk statistics,
and legacy record compatibility using isolated temporary ChromaVectorStore instances.
"""

from pathlib import Path
import pytest

from app.ingestion.chunker import DocumentChunk
from app.vectorstore.chroma_store import ChromaVectorStore

FAKE_DIM = 768


def make_vector(val: float = 0.1, dim: int = FAKE_DIM) -> list:
    """Helper to produce predictable test embedding vectors."""
    return [float(val)] * dim


def make_chunk(
    chunk_id: str,
    text: str,
    source: str = "report.pdf",
    page: int = 1,
    idx: int = 0,
    document_id: str = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page_number=page,
        chunk_index=idx,
        document_id=document_id,
    )


@pytest.fixture
def store(tmp_path: Path):
    """Provides an isolated ChromaVectorStore for testing."""
    return ChromaVectorStore(
        persist_dir=tmp_path / "test_doc_mgmt_chroma",
        collection_name="test_doc_mgmt",
    )


# ---------------------------------------------------------------------------
# Test 1 — Empty document list
# ---------------------------------------------------------------------------
def test_empty_document_list(store: ChromaVectorStore):
    """Verify list_indexed_documents returns an empty list when no documents are indexed."""
    assert store.count() == 0
    docs = store.list_indexed_documents()
    assert docs == []


# ---------------------------------------------------------------------------
# Test 2 — Multiple documents displayed
# ---------------------------------------------------------------------------
def test_multiple_documents_displayed(store: ChromaVectorStore):
    """Verify multiple distinct documents appear in the returned metadata listing."""
    doc_1 = "hash_doc_alpha"
    doc_2 = "hash_doc_beta"

    chunks = [
        make_chunk("c1", "Alpha page 1 text", source="alpha.pdf", page=1, idx=0, document_id=doc_1),
        make_chunk("c2", "Alpha page 2 text", source="alpha.pdf", page=2, idx=0, document_id=doc_1),
        make_chunk("c3", "Beta page 1 text", source="beta.pdf", page=1, idx=0, document_id=doc_2),
    ]
    store.add_chunks(chunks, [make_vector(0.1)] * 3)

    docs = store.list_indexed_documents()
    assert len(docs) == 2

    doc_ids = {d["document_id"] for d in docs}
    assert doc_ids == {doc_1, doc_2}

    alpha_info = next(d for d in docs if d["document_id"] == doc_1)
    beta_info = next(d for d in docs if d["document_id"] == doc_2)

    assert alpha_info["source"] == "alpha.pdf"
    assert alpha_info["page_count"] == 2
    assert alpha_info["chunk_count"] == 2

    assert beta_info["source"] == "beta.pdf"
    assert beta_info["page_count"] == 1
    assert beta_info["chunk_count"] == 1


# ---------------------------------------------------------------------------
# Test 3 — Same filename documents coexist independently
# ---------------------------------------------------------------------------
def test_same_filename_documents_displayed_separately(store: ChromaVectorStore):
    """
    Verify two documents with the identical filename 'report.pdf' but different
    document_ids appear as two separate records in the management listing.
    """
    doc_a = "1111111111111111111111111111111111111111111111111111111111111111"
    doc_b = "2222222222222222222222222222222222222222222222222222222222222222"

    chunks = [
        make_chunk("a1", "Version 1 content", source="report.pdf", page=1, idx=0, document_id=doc_a),
        make_chunk("b1", "Version 2 content", source="report.pdf", page=1, idx=0, document_id=doc_b),
        make_chunk("b2", "Version 2 page 2 content", source="report.pdf", page=2, idx=0, document_id=doc_b),
    ]
    store.add_chunks(chunks, [make_vector(0.1)] * 3)

    docs = store.list_indexed_documents()
    assert len(docs) == 2

    # Verify both records share the filename but have different document_ids and chunk counts
    doc_dict = {d["document_id"]: d for d in docs}
    assert doc_a in doc_dict
    assert doc_b in doc_dict
    assert doc_dict[doc_a]["source"] == "report.pdf"
    assert doc_dict[doc_b]["source"] == "report.pdf"
    assert doc_dict[doc_a]["chunk_count"] == 1
    assert doc_dict[doc_b]["chunk_count"] == 2


# ---------------------------------------------------------------------------
# Test 4 — Delete one document isolated by document_id
# ---------------------------------------------------------------------------
def test_delete_one_document_isolation(store: ChromaVectorStore):
    """
    Verify deleting HASH_A leaves HASH_B completely present even when they share
    the exact same filename.
    """
    doc_a = "hash_to_delete"
    doc_b = "hash_to_keep"

    chunks = [
        make_chunk("a1", "Doc A chunk 1", source="report.pdf", page=1, idx=0, document_id=doc_a),
        make_chunk("a2", "Doc A chunk 2", source="report.pdf", page=2, idx=0, document_id=doc_a),
        make_chunk("b1", "Doc B chunk 1", source="report.pdf", page=1, idx=0, document_id=doc_b),
    ]
    store.add_chunks(chunks, [make_vector(0.1)] * 3)
    assert store.count() == 3

    # Delete doc_a
    deleted_count = store.delete_by_document_id(doc_a)
    assert deleted_count == 2

    # Verify doc_a is absent, doc_b is present
    assert store.has_document(doc_a) is False
    assert store.has_document(doc_b) is True
    assert store.count() == 1

    remaining_docs = store.list_indexed_documents()
    assert len(remaining_docs) == 1
    assert remaining_docs[0]["document_id"] == doc_b
    assert remaining_docs[0]["source"] == "report.pdf"
    assert remaining_docs[0]["chunk_count"] == 1


# ---------------------------------------------------------------------------
# Test 5 — Accurate document statistics calculation
# ---------------------------------------------------------------------------
def test_document_statistics_accuracy(store: ChromaVectorStore):
    """
    Verify page_count reflects unique pages and chunk_count reflects total chunks.
    Document has 4 chunks across 2 pages (page 1 and page 2).
    """
    doc_id = "stats_verification_hash"
    chunks = [
        make_chunk("c1", "Page 1 chunk 0", source="paper.pdf", page=1, idx=0, document_id=doc_id),
        make_chunk("c2", "Page 1 chunk 1", source="paper.pdf", page=1, idx=1, document_id=doc_id),
        make_chunk("c3", "Page 2 chunk 0", source="paper.pdf", page=2, idx=0, document_id=doc_id),
        make_chunk("c4", "Page 2 chunk 1", source="paper.pdf", page=2, idx=1, document_id=doc_id),
    ]
    store.add_chunks(chunks, [make_vector(0.1)] * 4)

    docs = store.list_indexed_documents()
    assert len(docs) == 1
    info = docs[0]
    assert info["document_id"] == doc_id
    assert info["source"] == "paper.pdf"
    assert info["page_count"] == 2
    assert info["chunk_count"] == 4


# ---------------------------------------------------------------------------
# Test 6 — Legacy records handling in management listing
# ---------------------------------------------------------------------------
def test_legacy_records_do_not_crash_listing(store: ChromaVectorStore):
    """
    Verify legacy records lacking document_id are listed safely without raising exceptions.
    """
    legacy_chunk = make_chunk(
        "legacy_c1",
        "Legacy text without document_id",
        source="legacy_survey.pdf",
        page=1,
        idx=0,
        document_id=None,
    )
    store.add_chunks([legacy_chunk], [make_vector(0.1)])

    docs = store.list_indexed_documents()
    assert len(docs) == 1
    assert docs[0]["source"] == "legacy_survey.pdf"
    assert docs[0]["page_count"] == 1
    assert docs[0]["chunk_count"] == 1
    assert docs[0]["document_id"].startswith("legacy_")
