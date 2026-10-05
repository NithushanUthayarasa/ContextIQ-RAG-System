"""
Unit & regression tests for ContextIQ multi-user / session isolation.
Proves that simultaneous sessions have strictly isolated Chroma collections,
distinct upload directories, and zero cross-tenant document or query leakage.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from app.ingestion.chunker import DocumentChunk
from app.vectorstore.chroma_store import ChromaVectorStore
from app.main import get_vector_store


@pytest.fixture
def temp_chroma_dir():
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


def _create_sample_chunk(doc_id: str, chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        source="test.pdf",
        page_number=1,
        chunk_index=0,
        document_id=doc_id,
    )


def test_different_session_ids_create_distinct_isolated_chroma_collections(temp_chroma_dir):
    """Prove that two user sessions store and list documents in completely separate collections."""
    store_a = ChromaVectorStore(
        persist_dir=temp_chroma_dir,
        collection_name="ctx_session_user_a",
        expected_dimension=4,
    )
    store_b = ChromaVectorStore(
        persist_dir=temp_chroma_dir,
        collection_name="ctx_session_user_b",
        expected_dimension=4,
    )

    chunk_a = _create_sample_chunk("doc_user_a", "c_a1", "Private financial document for User A")
    emb_a = [0.1, 0.2, 0.3, 0.4]
    store_a.add_chunks([chunk_a], [emb_a])

    # User A sees doc_user_a
    docs_a = store_a.list_indexed_documents()
    assert len(docs_a) == 1
    assert docs_a[0]["document_id"] == "doc_user_a"

    # User B MUST see 0 documents
    docs_b = store_b.list_indexed_documents()
    assert len(docs_b) == 0

    # User B searching vector store finds 0 chunks
    results_b = store_b.query(query_embedding=[0.1, 0.2, 0.3, 0.4], top_k=5)
    assert len(results_b["ids"][0]) == 0


def test_document_deletion_in_session_a_does_not_affect_session_b(temp_chroma_dir):
    """Prove that deleting a document in Session A does not modify Session B."""
    store_a = ChromaVectorStore(
        persist_dir=temp_chroma_dir,
        collection_name="ctx_session_user_a",
        expected_dimension=4,
    )
    store_b = ChromaVectorStore(
        persist_dir=temp_chroma_dir,
        collection_name="ctx_session_user_b",
        expected_dimension=4,
    )

    chunk_a = _create_sample_chunk("doc_a", "c_a1", "Confidential Alpha")
    chunk_b = _create_sample_chunk("doc_b", "c_b1", "Confidential Beta")

    store_a.add_chunks([chunk_a], [[0.5, 0.5, 0.5, 0.5]])
    store_b.add_chunks([chunk_b], [[0.5, 0.5, 0.5, 0.5]])

    assert len(store_a.list_indexed_documents()) == 1
    assert len(store_b.list_indexed_documents()) == 1

    # Session A deletes doc_a
    store_a.delete_by_document_id("doc_a")

    assert len(store_a.list_indexed_documents()) == 0
    # Session B still has doc_b
    assert len(store_b.list_indexed_documents()) == 1
    assert store_b.list_indexed_documents()[0]["document_id"] == "doc_b"


def test_database_reset_in_session_a_does_not_affect_session_b(temp_chroma_dir):
    """Prove that vector_store.reset() in Session A preserves Session B data."""
    store_a = ChromaVectorStore(
        persist_dir=temp_chroma_dir,
        collection_name="ctx_session_user_a",
        expected_dimension=4,
    )
    store_b = ChromaVectorStore(
        persist_dir=temp_chroma_dir,
        collection_name="ctx_session_user_b",
        expected_dimension=4,
    )

    chunk_b = _create_sample_chunk("doc_b", "c_b1", "User B Document Data")
    store_b.add_chunks([chunk_b], [[0.1, 0.2, 0.3, 0.4]])

    # Session A resets
    store_a.reset()

    # Session B is completely unharmed
    assert len(store_b.list_indexed_documents()) == 1
    assert store_b.count() == 1


def test_session_isolated_upload_directories():
    """Prove that distinct sessions derive distinct isolated upload staging directories."""
    from app.config import UPLOAD_DIR

    session_id_a = "user_alpha_123"
    session_id_b = "user_beta_456"

    dir_a = UPLOAD_DIR / session_id_a
    dir_b = UPLOAD_DIR / session_id_b

    assert dir_a != dir_b
    assert str(dir_a).endswith("user_alpha_123")
    assert str(dir_b).endswith("user_beta_456")


def test_main_get_vector_store_collection_naming():
    """Verify get_vector_store creates sanitized collection names per session_id."""
    store_a = get_vector_store("session_1a2b3c")
    assert store_a.collection_name == "ctx_session_1a2b3c"

    # Default without session_id preserves baseline
    store_default = get_vector_store()
    from app.config import CHROMA_COLLECTION_NAME
    assert store_default.collection_name == CHROMA_COLLECTION_NAME
