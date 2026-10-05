"""
Unit tests for ContextIQ ChromaVectorStore.
Tests use temporary local directories and fake embedding vectors to ensure fast, isolated execution.
"""

from pathlib import Path
import pytest

from typing import Optional
from app.ingestion.chunker import DocumentChunk
from app.vectorstore.chroma_store import (
    ChromaVectorStore,
    EmptyVectorStoreInputError,
    MismatchedInputError,
    InvalidEmbeddingError,
)

FAKE_DIM = 768


def make_chunk(
    cid: str,
    text: str,
    source: str = "test.pdf",
    page: int = 1,
    idx: int = 0,
    document_id: Optional[str] = None,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid,
        text=text,
        source=source,
        page_number=page,
        chunk_index=idx,
        document_id=document_id,
    )


def make_vector(val: float = 0.1, dim: int = FAKE_DIM) -> list:
    return [float(val)] * dim


def test_store_initialization(tmp_path: Path):
    """Test initializing ChromaVectorStore with a custom temporary directory."""
    store = ChromaVectorStore(persist_dir=tmp_path, collection_name="test_init")
    assert store.count() == 0
    assert store.collection_name == "test_init"
    assert store.persist_dir == tmp_path


def test_collection_creation_cosine_metric(tmp_path: Path):
    """Verify that the collection is configured explicitly for cosine space."""
    store = ChromaVectorStore(persist_dir=tmp_path, collection_name="test_cosine")
    # Verify collection metadata includes hnsw:space == cosine
    coll = store.collection
    assert coll.metadata.get("hnsw:space") == "cosine"


def test_empty_chunks_validation(tmp_path: Path):
    """Verify error raised when empty chunks list is provided."""
    store = ChromaVectorStore(persist_dir=tmp_path)
    with pytest.raises(EmptyVectorStoreInputError, match="must not be empty"):
        store.add_chunks([], [make_vector()])


def test_empty_embeddings_validation(tmp_path: Path):
    """Verify error raised when empty embeddings list is provided."""
    store = ChromaVectorStore(persist_dir=tmp_path)
    chunk = make_chunk("c1", "text")
    with pytest.raises(EmptyVectorStoreInputError, match="must not be empty"):
        store.add_chunks([chunk], [])


def test_mismatched_counts_validation(tmp_path: Path):
    """Verify error raised when chunks and embeddings counts differ."""
    store = ChromaVectorStore(persist_dir=tmp_path)
    chunks = [make_chunk("c1", "text 1"), make_chunk("c2", "text 2")]
    embeddings = [make_vector()]  # Only 1 embedding for 2 chunks

    with pytest.raises(MismatchedInputError, match="does not match number of embeddings"):
        store.add_chunks(chunks, embeddings)


def test_successful_insertion_and_count(tmp_path: Path):
    """Verify successful insertion and count update."""
    store = ChromaVectorStore(persist_dir=tmp_path)
    chunks = [
        make_chunk("doc_p1_c0", "First chunk text", page=1, idx=0),
        make_chunk("doc_p1_c1", "Second chunk text", page=1, idx=1),
    ]
    embeddings = [make_vector(0.1), make_vector(0.2)]

    added = store.add_chunks(chunks, embeddings)
    assert added == 2
    assert store.count() == 2


def test_metadata_preservation(tmp_path: Path):
    """Verify that chunk metadata (source, page_number, chunk_index) is accurately stored."""
    store = ChromaVectorStore(persist_dir=tmp_path)
    chunk = make_chunk("paper_p3_c1", "Paper content on page 3", source="paper.pdf", page=3, idx=1)

    store.add_chunks([chunk], [make_vector()])

    res = store.get(ids=["paper_p3_c1"])
    assert len(res["ids"]) == 1
    assert res["documents"][0] == "Paper content on page 3"
    meta = res["metadatas"][0]
    assert meta["source"] == "paper.pdf"
    assert meta["page_number"] == 3
    assert meta["chunk_index"] == 1


def test_deterministic_ids_and_upsert_behavior(tmp_path: Path):
    """Verify that re-inserting with the same chunk_id updates existing record without duplicates."""
    store = ChromaVectorStore(persist_dir=tmp_path)
    chunk_v1 = make_chunk("c1", "Original text")
    store.add_chunks([chunk_v1], [make_vector(0.1)])
    assert store.count() == 1

    # Upsert with same ID but updated text
    chunk_v2 = make_chunk("c1", "Updated text content")
    store.add_chunks([chunk_v2], [make_vector(0.5)])

    # Count must remain 1, text updated
    assert store.count() == 1
    res = store.get(ids=["c1"])
    assert res["documents"][0] == "Updated text content"


def test_persistence_after_reopening(tmp_path: Path):
    """Verify that chunks persist on disk and are available after instantiating a new store."""
    persist_dir = tmp_path / "persistent_chroma"
    store1 = ChromaVectorStore(persist_dir=persist_dir, collection_name="persist_test")
    chunks = [
        make_chunk("persist_p1_c0", "Chunk 1 to persist"),
        make_chunk("persist_p1_c1", "Chunk 2 to persist"),
    ]
    store1.add_chunks(chunks, [make_vector(0.1), make_vector(0.2)])
    assert store1.count() == 2

    # Instantiate a completely separate store pointing to the same directory
    store2 = ChromaVectorStore(persist_dir=persist_dir, collection_name="persist_test")
    assert store2.count() == 2
    res = store2.get(ids=["persist_p1_c0", "persist_p1_c1"])
    assert set(res["ids"]) == {"persist_p1_c0", "persist_p1_c1"}


def test_reset_and_delete_by_source(tmp_path: Path):
    """Verify reset() and delete_by_source() functionality."""
    store = ChromaVectorStore(persist_dir=tmp_path)
    chunks = [
        make_chunk("c_docA_1", "text A1", source="docA.pdf"),
        make_chunk("c_docA_2", "text A2", source="docA.pdf"),
        make_chunk("c_docB_1", "text B1", source="docB.pdf"),
    ]
    store.add_chunks(chunks, [make_vector(), make_vector(), make_vector()])
    assert store.count() == 3

    # Delete docA only
    store.delete_by_source("docA.pdf")
    assert store.count() == 1
    remaining = store.get(ids=["c_docB_1"])
    assert len(remaining["ids"]) == 1

    # Reset clears everything
    store.reset()
    assert store.count() == 0


def test_embedding_dimension_validation(tmp_path: Path):
    """Verify error when embedding dimensionality does not match expected dimension."""
    store = ChromaVectorStore(persist_dir=tmp_path, expected_dimension=FAKE_DIM)
    chunk = make_chunk("c1", "text")
    wrong_dim_vector = [0.1] * 512  # 512 instead of 768

    with pytest.raises(InvalidEmbeddingError, match="does not match expected dimension"):
        store.add_chunks([chunk], [wrong_dim_vector])


def test_non_numeric_embedding_validation(tmp_path: Path):
    """Verify error when embedding vector contains invalid non-numeric values."""
    store = ChromaVectorStore(persist_dir=tmp_path, expected_dimension=3)
    chunk = make_chunk("c1", "text")

    with pytest.raises(InvalidEmbeddingError, match="contains non-numeric values"):
        store.add_chunks([chunk], [["not", "a", "number"]])


def test_document_id_metadata_storage(tmp_path: Path):
    """Test 1: Verify document_id is accurately saved into ChromaDB metadata."""
    store = ChromaVectorStore(persist_dir=tmp_path)
    doc_id_1 = "hash_doc_1"
    doc_id_2 = "hash_doc_2"

    chunks = [
        make_chunk("c1", "text 1", source="doc1.pdf", page=1, idx=0, document_id=doc_id_1),
        make_chunk("c2", "text 2", source="doc2.pdf", page=2, idx=0, document_id=doc_id_2),
    ]
    store.add_chunks(chunks, [make_vector(0.1), make_vector(0.2)])

    res1 = store.get(ids=["c1"])
    assert res1["metadatas"][0]["document_id"] == doc_id_1
    assert res1["metadatas"][0]["source"] == "doc1.pdf"

    res2 = store.get(ids=["c2"])
    assert res2["metadatas"][0]["document_id"] == doc_id_2
    assert res2["metadatas"][0]["source"] == "doc2.pdf"


def test_has_document(tmp_path: Path):
    """Test 2: Verify has_document returns True for existing docs and False for unknown/invalid docs."""
    store = ChromaVectorStore(persist_dir=tmp_path)
    doc_id = "target_doc_sha256"
    chunks = [make_chunk("c1", "hello world", document_id=doc_id)]
    store.add_chunks(chunks, [make_vector(0.1)])

    assert store.has_document(doc_id) is True
    assert store.has_document("non_existent_hash") is False
    assert store.has_document("") is False
    assert store.has_document("   ") is False


def test_delete_by_document_id(tmp_path: Path):
    """Test 3: Verify delete_by_document_id deletes all chunks for target document while leaving others untouched."""
    store = ChromaVectorStore(persist_dir=tmp_path)
    doc_a = "doc_a_hash"
    doc_b = "doc_b_hash"

    chunks = [
        make_chunk("a1", "A chunk 1", document_id=doc_a),
        make_chunk("a2", "A chunk 2", document_id=doc_a),
        make_chunk("a3", "A chunk 3", document_id=doc_a),
        make_chunk("b1", "B chunk 1", document_id=doc_b),
        make_chunk("b2", "B chunk 2", document_id=doc_b),
    ]
    store.add_chunks(chunks, [make_vector(0.1)] * 5)
    assert store.count() == 5

    # Delete doc A
    deleted_count = store.delete_by_document_id(doc_a)
    assert deleted_count == 3
    assert store.count() == 2
    assert store.has_document(doc_a) is False
    assert store.has_document(doc_b) is True

    # Remaining chunks must belong only to doc B
    remaining = store.get()
    assert set(remaining["ids"]) == {"b1", "b2"}
    for meta in remaining["metadatas"]:
        assert meta["document_id"] == doc_b

    # Deleting non-existent document returns 0
    assert store.delete_by_document_id("unknown_doc") == 0


def test_same_filename_different_document_ids(tmp_path: Path):
    """Test 4: Verify two documents with the same filename but different document_ids coexist independently."""
    store = ChromaVectorStore(persist_dir=tmp_path)
    doc_v1 = "hash_report_version1"
    doc_v2 = "hash_report_version2"

    chunks = [
        make_chunk("v1_c0", "Version 1 chunk", source="report.pdf", page=1, idx=0, document_id=doc_v1),
        make_chunk("v2_c0", "Version 2 chunk", source="report.pdf", page=1, idx=0, document_id=doc_v2),
    ]
    store.add_chunks(chunks, [make_vector(0.1), make_vector(0.2)])
    assert store.count() == 2

    docs = store.list_indexed_documents()
    assert len(docs) == 2
    ids = {d["document_id"] for d in docs}
    assert ids == {doc_v1, doc_v2}
    for d in docs:
        assert d["source"] == "report.pdf"


def test_list_indexed_documents_aggregation(tmp_path: Path):
    """Test 5: Verify list_indexed_documents correctly computes page_count, chunk_count, and preserves source & document_id."""
    store = ChromaVectorStore(persist_dir=tmp_path)
    doc_id = "doc_multi_page_hash"

    # 4 chunks across 2 distinct pages (pages 1 and 2)
    chunks = [
        make_chunk("c1", "p1 c0", source="paper.pdf", page=1, idx=0, document_id=doc_id),
        make_chunk("c2", "p1 c1", source="paper.pdf", page=1, idx=1, document_id=doc_id),
        make_chunk("c3", "p2 c0", source="paper.pdf", page=2, idx=0, document_id=doc_id),
        make_chunk("c4", "p2 c1", source="paper.pdf", page=2, idx=1, document_id=doc_id),
    ]
    store.add_chunks(chunks, [make_vector(0.1)] * 4)

    docs = store.list_indexed_documents()
    assert len(docs) == 1
    doc_info = docs[0]
    assert doc_info["document_id"] == doc_id
    assert doc_info["source"] == "paper.pdf"
    assert doc_info["page_count"] == 2
    assert doc_info["chunk_count"] == 4


def test_same_content_identity_upsert_no_duplicates(tmp_path: Path):
    """Test 6: Verify re-inserting same document_id with same chunk IDs updates without creating duplicate records."""
    store = ChromaVectorStore(persist_dir=tmp_path)
    doc_id = "deterministic_doc_hash"
    chunk_1 = make_chunk(f"{doc_id}_p1_c0", "Initial text", source="doc.pdf", page=1, idx=0, document_id=doc_id)
    store.add_chunks([chunk_1], [make_vector(0.1)])
    assert store.count() == 1

    # Re-insert with same chunk ID
    chunk_1_updated = make_chunk(f"{doc_id}_p1_c0", "Updated text", source="doc.pdf", page=1, idx=0, document_id=doc_id)
    store.add_chunks([chunk_1_updated], [make_vector(0.2)])

    assert store.count() == 1
    docs = store.list_indexed_documents()
    assert len(docs) == 1
    assert docs[0]["chunk_count"] == 1
    res = store.get(ids=[f"{doc_id}_p1_c0"])
    assert res["documents"][0] == "Updated text"


def test_list_indexed_documents_empty_collection(tmp_path: Path):
    """Test 7: Verify list_indexed_documents returns an empty list cleanly when store is empty."""
    store = ChromaVectorStore(persist_dir=tmp_path)
    assert store.list_indexed_documents() == []

