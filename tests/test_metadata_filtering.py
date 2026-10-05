"""
Unit and integration tests for Metadata-Aware Document Filtering (Phase 13 Step 3).
Tests document_ids filtering, interplay with top_k and similarity threshold,
legacy records handling, same-filename disambiguation, and pipeline integration.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app.generation.generator import GeminiGenerator
from app.ingestion.chunker import DocumentChunk
from app.ingestion.embedder import GeminiEmbedder
from app.rag.pipeline import RAGPipeline, RAGResponse
from app.retrieval.models import RetrievalResult, RetrievedChunk
from app.retrieval.retriever import (
    InvalidDocumentFilterError,
    Retriever,
)
from app.vectorstore.chroma_store import ChromaVectorStore

FAKE_DIM = 768


def make_vector(val: float = 0.1, dim: int = FAKE_DIM) -> list:
    return [float(val)] * dim


@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=GeminiEmbedder)
    embedder.embed_query.return_value = make_vector(0.1)
    return embedder


@pytest.fixture
def multi_doc_store(tmp_path: Path):
    """
    ChromaVectorStore populated with 3 documents:
    - doc_a: "attention.pdf" (2 chunks)
    - doc_b: "rag.pdf" (2 chunks)
    - doc_c: "mamba.pdf" (1 chunk)
    """
    store = ChromaVectorStore(
        persist_dir=tmp_path / "multi_doc_db",
        collection_name="multi_doc_coll",
    )
    chunks = [
        DocumentChunk("a_c0", "Attention mechanism chunk 0", "attention.pdf", 1, 0, "hash_doc_a"),
        DocumentChunk("a_c1", "Attention mechanism chunk 1", "attention.pdf", 2, 0, "hash_doc_a"),
        DocumentChunk("b_c0", "RAG architecture chunk 0", "rag.pdf", 1, 0, "hash_doc_b"),
        DocumentChunk("b_c1", "RAG architecture chunk 1", "rag.pdf", 2, 0, "hash_doc_b"),
        DocumentChunk("c_c0", "State space models chunk 0", "mamba.pdf", 1, 0, "hash_doc_c"),
    ]
    # Distances will vary based on vectors
    embeddings = [
        make_vector(0.1),
        make_vector(0.15),
        make_vector(0.2),
        make_vector(0.25),
        make_vector(0.3),
    ]
    store.add_chunks(chunks, embeddings)
    return store


def test_no_document_filter_searches_all_documents(mock_embedder, multi_doc_store):
    """Test 1: document_ids=None retrieves chunks across all indexed documents."""
    retriever = Retriever(embedder=mock_embedder, vector_store=multi_doc_store, default_min_similarity=0.0)
    results = retriever.retrieve("query", top_k=5, document_ids=None)

    assert len(results) == 5
    doc_ids = {r.document_id for r in results}
    assert doc_ids == {"hash_doc_a", "hash_doc_b", "hash_doc_c"}


def test_single_document_filter(mock_embedder, multi_doc_store):
    """Test 2: document_ids=['hash_doc_a'] retrieves only chunks from doc_a."""
    retriever = Retriever(embedder=mock_embedder, vector_store=multi_doc_store, default_min_similarity=0.0)
    results = retriever.retrieve("query", top_k=5, document_ids=["hash_doc_a"])

    assert len(results) == 2
    for r in results:
        assert r.document_id == "hash_doc_a"
        assert r.source == "attention.pdf"


def test_multiple_document_filter(mock_embedder, multi_doc_store):
    """Test 3: document_ids=['hash_doc_a', 'hash_doc_c'] retrieves chunks from doc_a and doc_c, excluding doc_b."""
    retriever = Retriever(embedder=mock_embedder, vector_store=multi_doc_store, default_min_similarity=0.0)
    results = retriever.retrieve("query", top_k=5, document_ids=["hash_doc_a", "hash_doc_c"])

    assert len(results) == 3
    doc_ids = {r.document_id for r in results}
    assert doc_ids == {"hash_doc_a", "hash_doc_c"}
    assert "hash_doc_b" not in doc_ids


def test_nonexistent_document_id_returns_empty(mock_embedder, multi_doc_store):
    """Test 4: Filtering by a document_id that doesn't exist returns empty list []."""
    retriever = Retriever(embedder=mock_embedder, vector_store=multi_doc_store, default_min_similarity=0.0)
    results = retriever.retrieve("query", top_k=5, document_ids=["hash_nonexistent_999"])

    assert results == []


def test_empty_document_filter_behavior(mock_embedder, multi_doc_store):
    """Test 5: document_ids=[] is treated as no restriction (searches all documents)."""
    retriever = Retriever(embedder=mock_embedder, vector_store=multi_doc_store, default_min_similarity=0.0)
    results = retriever.retrieve("query", top_k=5, document_ids=[])

    assert len(results) == 5
    doc_ids = {r.document_id for r in results}
    assert doc_ids == {"hash_doc_a", "hash_doc_b", "hash_doc_c"}


def test_duplicate_document_ids_deduplicated(mock_embedder, multi_doc_store):
    """Test 6: Duplicate document IDs in the input list do not cause duplicated results."""
    retriever = Retriever(embedder=mock_embedder, vector_store=multi_doc_store, default_min_similarity=0.0)
    results = retriever.retrieve("query", top_k=5, document_ids=["hash_doc_a", "hash_doc_a", "hash_doc_a"])

    assert len(results) == 2
    chunk_ids = [r.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert all(r.document_id == "hash_doc_a" for r in results)


def test_document_filter_with_similarity_threshold(mock_embedder):
    """Test 7: Document filter and similarity threshold work together."""
    fake_store = MagicMock(spec=ChromaVectorStore)
    fake_store.count.return_value = 2
    # doc_a chunks with similarities 0.95 and 0.40 (distances 0.05 and 0.60)
    fake_store.query.return_value = {
        "ids": [["a_c0", "a_c1"]],
        "documents": [["text 0", "text 1"]],
        "metadatas": [[
            {"source": "a.pdf", "page_number": 1, "chunk_index": 0, "document_id": "hash_a"},
            {"source": "a.pdf", "page_number": 2, "chunk_index": 0, "document_id": "hash_a"},
        ]],
        "distances": [[0.05, 0.60]],
    }
    retriever = Retriever(embedder=mock_embedder, vector_store=fake_store)
    results = retriever.retrieve("query", top_k=2, similarity_threshold=0.70, document_ids=["hash_a"])

    # Chroma was queried with where={"document_id": "hash_a"}
    fake_store.query.assert_called_once_with(
        query_embedding=mock_embedder.embed_query.return_value,
        top_k=2,
        where={"document_id": "hash_a"},
    )
    # Only the chunk with similarity 0.95 >= 0.70 is retained
    assert len(results) == 1
    assert results[0].chunk_id == "a_c0"
    assert round(results[0].cosine_similarity, 2) == 0.95


def test_document_filter_preserves_top_k(mock_embedder, multi_doc_store):
    """Test 8: Document filter preserves top_k limit."""
    retriever = Retriever(embedder=mock_embedder, vector_store=multi_doc_store, default_min_similarity=0.0)
    # doc_a has 2 chunks; top_k=1 limits return to 1
    results = retriever.retrieve("query", top_k=1, document_ids=["hash_doc_a"])
    assert len(results) == 1
    assert results[0].document_id == "hash_doc_a"


def test_document_filter_preserves_ordering(mock_embedder):
    """Test 9: Document filter preserves nearest-first distance ordering."""
    fake_store = MagicMock(spec=ChromaVectorStore)
    fake_store.count.return_value = 3
    fake_store.query.return_value = {
        "ids": [["c1", "c2", "c3"]],
        "documents": [["doc 1", "doc 2", "doc 3"]],
        "metadatas": [[
            {"source": "a.pdf", "page_number": 1, "chunk_index": 0, "document_id": "hash_a"},
            {"source": "a.pdf", "page_number": 1, "chunk_index": 1, "document_id": "hash_a"},
            {"source": "a.pdf", "page_number": 2, "chunk_index": 0, "document_id": "hash_a"},
        ]],
        "distances": [[0.08, 0.15, 0.22]],
    }
    retriever = Retriever(embedder=mock_embedder, vector_store=fake_store)
    results = retriever.retrieve("query", top_k=3, similarity_threshold=0.0, document_ids=["hash_a"])

    assert len(results) == 3
    assert results[0].distance < results[1].distance < results[2].distance
    assert [r.chunk_id for r in results] == ["c1", "c2", "c3"]


def test_metadata_survives_document_filtering(mock_embedder, multi_doc_store):
    """Test 10: Retained chunks preserve document_id, source, page_number, chunk_index, and distance."""
    retriever = Retriever(embedder=mock_embedder, vector_store=multi_doc_store, default_min_similarity=0.0)
    results = retriever.retrieve("query", top_k=1, document_ids=["hash_doc_b"])

    assert len(results) == 1
    chunk = results[0]
    assert chunk.document_id == "hash_doc_b"
    assert chunk.source == "rag.pdf"
    assert chunk.page_number in {1, 2}
    assert isinstance(chunk.chunk_index, int)
    assert isinstance(chunk.distance, float)
    assert isinstance(chunk.cosine_similarity, float)


def test_legacy_records_excluded_by_document_filter(tmp_path: Path, mock_embedder):
    """
    Test 11:
    - Legacy records without document_id are searched when document_ids=None.
    - Legacy records are NOT matched when document_ids=['some_sha256'].
    """
    store = ChromaVectorStore(
        persist_dir=tmp_path / "legacy_test_db",
        collection_name="legacy_coll",
    )
    # Add a legacy chunk (no document_id) and a modern chunk (with document_id)
    leg_chunk = DocumentChunk("leg_1", "Legacy content without doc_id", "old.pdf", 1, 0, None)
    mod_chunk = DocumentChunk("mod_1", "Modern content with doc_id", "new.pdf", 1, 0, "sha_modern_123")

    store.add_chunks([leg_chunk, mod_chunk], [make_vector(0.1), make_vector(0.1)])

    retriever = Retriever(embedder=mock_embedder, vector_store=store, default_min_similarity=0.0)

    # 1. No filter -> both legacy and modern returned
    all_res = retriever.retrieve("q", top_k=2, document_ids=None)
    assert len(all_res) == 2
    sources = {r.source for r in all_res}
    assert sources == {"old.pdf", "new.pdf"}

    # 2. Filter by modern doc_id -> only modern returned, legacy excluded
    mod_res = retriever.retrieve("q", top_k=2, document_ids=["sha_modern_123"])
    assert len(mod_res) == 1
    assert mod_res[0].chunk_id == "mod_1"
    assert mod_res[0].document_id == "sha_modern_123"


def test_same_filename_different_documents_filtered_independently(tmp_path: Path, mock_embedder):
    """Test 12: Two documents with identical filenames but different document_ids can be filtered independently."""
    store = ChromaVectorStore(
        persist_dir=tmp_path / "same_filename_db",
        collection_name="same_filename_coll",
    )
    doc_v1 = DocumentChunk("v1_c0", "Version 1 content", "report.pdf", 1, 0, "hash_v1_aaa")
    doc_v2 = DocumentChunk("v2_c0", "Version 2 content", "report.pdf", 1, 0, "hash_v2_bbb")

    store.add_chunks([doc_v1, doc_v2], [make_vector(0.1), make_vector(0.15)])

    retriever = Retriever(embedder=mock_embedder, vector_store=store, default_min_similarity=0.0)

    # Filter only v1
    res_v1 = retriever.retrieve("q", top_k=2, document_ids=["hash_v1_aaa"])
    assert len(res_v1) == 1
    assert res_v1[0].document_id == "hash_v1_aaa"
    assert res_v1[0].text == "Version 1 content"

    # Filter only v2
    res_v2 = retriever.retrieve("q", top_k=2, document_ids=["hash_v2_bbb"])
    assert len(res_v2) == 1
    assert res_v2[0].document_id == "hash_v2_bbb"
    assert res_v2[0].text == "Version 2 content"


def test_pipeline_forwards_document_ids(mock_embedder):
    """Test 13: RAGPipeline.ask forwards document_ids to retriever and records in RAGResponse."""
    mock_retriever = MagicMock(spec=Retriever)
    mock_retriever.retrieve.return_value = [
        RetrievedChunk("c1", "Retrieved answer text", "doc.pdf", 1, 0, 0.1, "hash_abc")
    ]
    mock_generator = MagicMock(spec=GeminiGenerator)
    mock_generator.generate.return_value = "Answer about doc."

    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
    response = pipeline.ask("What is in doc?", document_ids=["hash_abc"])

    assert isinstance(response, RAGResponse)
    assert response.document_ids == ["hash_abc"]
    mock_retriever.retrieve.assert_called_once_with(
        query="What is in doc?",
        top_k=None,
        document_ids=["hash_abc"],
    )


def test_filtered_results_trigger_insufficient_context_behavior(mock_embedder, multi_doc_store):
    """Test 14: When document filter yields 0 chunks, generator produces insufficient-context message."""
    retriever = Retriever(embedder=mock_embedder, vector_store=multi_doc_store)
    generator = MagicMock(spec=GeminiGenerator)
    generator.EMPTY_CONTEXT_MESSAGE = "I couldn't find relevant information in the indexed documents."
    generator.generate.return_value = "I couldn't find relevant information in the indexed documents."

    pipeline = RAGPipeline(retriever=retriever, generator=generator)
    response = pipeline.ask("Question", document_ids=["hash_nonexistent_xyz"])

    assert response.retrieved_chunks == []
    assert response.sources == []
    assert response.answer == "I couldn't find relevant information in the indexed documents."
    generator.generate.assert_called_once_with(question="Question", retrieved_chunks=[])


def test_invalid_document_filter_inputs(mock_embedder, multi_doc_store):
    """Test 15: Strings, non-iterable types, and empty elements raise InvalidDocumentFilterError."""
    retriever = Retriever(embedder=mock_embedder, vector_store=multi_doc_store)

    # String instead of list of strings (prevents character-by-character iteration)
    with pytest.raises(InvalidDocumentFilterError, match="must be a sequence of strings"):
        retriever.retrieve("q", document_ids="hash_doc_a")

    # Non-iterable integer
    with pytest.raises(InvalidDocumentFilterError, match="must be an iterable sequence"):
        retriever.retrieve("q", document_ids=123)

    # List with empty string
    with pytest.raises(InvalidDocumentFilterError, match="must be a non-empty string"):
        retriever.retrieve("q", document_ids=[""])

    # List with whitespace-only string
    with pytest.raises(InvalidDocumentFilterError, match="must be a non-empty string"):
        retriever.retrieve("q", document_ids=["   "])

    # List with non-string element
    with pytest.raises(InvalidDocumentFilterError, match="must be a non-empty string"):
        retriever.retrieve("q", document_ids=["hash_doc_a", 42])
