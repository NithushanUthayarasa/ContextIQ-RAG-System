"""
Unit tests for ContextIQ ChromaVectorStore.
Tests use temporary local directories and fake embedding vectors to ensure fast, isolated execution.
"""

from pathlib import Path
import pytest

from app.ingestion.chunker import DocumentChunk
from app.vectorstore.chroma_store import (
    ChromaVectorStore,
    EmptyVectorStoreInputError,
    MismatchedInputError,
    InvalidEmbeddingError,
)

FAKE_DIM = 768


def make_chunk(cid: str, text: str, source: str = "test.pdf", page: int = 1, idx: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid,
        text=text,
        source=source,
        page_number=page,
        chunk_index=idx,
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
