"""
Unit tests for ContextIQ Retriever and RetrievedChunk.
Tests use mocked query embeddings and isolated temporary ChromaVectorStore instances.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app.ingestion.chunker import DocumentChunk
from app.ingestion.embedder import GeminiEmbedder
from app.retrieval.models import RetrievalResult
from app.retrieval.retriever import (
    RetrievedChunk,
    Retriever,
    RetrieverError,
    EmptyQueryError,
    InvalidTopKError,
)
from app.vectorstore.chroma_store import ChromaVectorStore

FAKE_DIM = 768


def make_vector(val: float = 0.1, dim: int = FAKE_DIM) -> list:
    return [float(val)] * dim


@pytest.fixture
def mock_embedder():
    """Mock GeminiEmbedder returning deterministic fake vectors."""
    embedder = MagicMock(spec=GeminiEmbedder)
    embedder.embed_query.return_value = make_vector(0.1)
    return embedder


@pytest.fixture
def populated_vector_store(tmp_path: Path):
    """Temporary ChromaVectorStore with 3 pre-inserted test chunks."""
    store = ChromaVectorStore(persist_dir=tmp_path / "test_retrieval_db", collection_name="test_retrieval")
    chunks = [
        DocumentChunk("rag_p1_c0", "What is RAG? Overview of retrieval-augmented generation.", "paper.pdf", 1, 0),
        DocumentChunk("rag_p2_c0", "Chunking strategies and vector similarity metrics.", "paper.pdf", 2, 0),
        DocumentChunk("rag_p3_c0", "ChromaDB persistent vector indexing with cosine distance.", "paper.pdf", 3, 0),
    ]
    # Vectors with varying values
    embeddings = [make_vector(0.1), make_vector(0.3), make_vector(0.5)]
    store.add_chunks(chunks, embeddings)
    return store


def test_retriever_initialization(mock_embedder, populated_vector_store):
    """Verify Retriever initializes properly with injected components."""
    retriever = Retriever(embedder=mock_embedder, vector_store=populated_vector_store, default_top_k=2)
    assert retriever.default_top_k == 2

    with pytest.raises(RetrieverError, match="embedder instance must be provided"):
        Retriever(embedder=None, vector_store=populated_vector_store)

    with pytest.raises(RetrieverError, match="vector_store instance must be provided"):
        Retriever(embedder=mock_embedder, vector_store=None)


def test_successful_retrieval(mock_embedder, populated_vector_store):
    """Verify end-to-end query retrieval with RetrievedChunk conversion."""
    retriever = Retriever(embedder=mock_embedder, vector_store=populated_vector_store, default_top_k=2)
    results = retriever.retrieve("What is RAG?", top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_query_embedding_called(mock_embedder, populated_vector_store):
    """Verify that embed_query is called with the exact trimmed query string."""
    retriever = Retriever(embedder=mock_embedder, vector_store=populated_vector_store)
    retriever.retrieve("   Explain vector search   ")

    mock_embedder.embed_query.assert_called_once_with("Explain vector search")


def test_correct_top_k_passed(mock_embedder, populated_vector_store):
    """Verify that custom top_k limits the number of returned chunks."""
    retriever = Retriever(embedder=mock_embedder, vector_store=populated_vector_store, default_top_k=5)
    results = retriever.retrieve("Any query", top_k=1)
    assert len(results) == 1

    results_all = retriever.retrieve("Any query", top_k=3)
    assert len(results_all) == 3


def test_metadata_and_text_preserved(mock_embedder, populated_vector_store):
    """Verify text, source, page_number, chunk_index, and distance are preserved."""
    retriever = Retriever(embedder=mock_embedder, vector_store=populated_vector_store)
    results = retriever.retrieve("RAG query", top_k=1)

    top = results[0]
    assert top.chunk_id in {"rag_p1_c0", "rag_p2_c0", "rag_p3_c0"}
    assert top.source == "paper.pdf"
    assert top.page_number in {1, 2, 3}
    assert isinstance(top.chunk_index, int)
    assert isinstance(top.distance, float)
    assert len(top.text) > 0


def test_empty_query_validation(mock_embedder, populated_vector_store):
    """Verify that empty or whitespace query raises EmptyQueryError."""
    retriever = Retriever(embedder=mock_embedder, vector_store=populated_vector_store)

    with pytest.raises(EmptyQueryError, match="must not be empty"):
        retriever.retrieve("")

    with pytest.raises(EmptyQueryError, match="must not be empty"):
        retriever.retrieve("    \t\n   ")


def test_invalid_top_k_validation(mock_embedder, populated_vector_store):
    """Verify that non-positive or non-integer top_k raises InvalidTopKError."""
    retriever = Retriever(embedder=mock_embedder, vector_store=populated_vector_store)

    with pytest.raises(InvalidTopKError, match="must be a positive integer"):
        retriever.retrieve("Valid query", top_k=0)

    with pytest.raises(InvalidTopKError, match="must be a positive integer"):
        retriever.retrieve("Valid query", top_k=-2)

    with pytest.raises(InvalidTopKError, match="must be a positive integer"):
        retriever.retrieve("Valid query", top_k="invalid")


def test_empty_collection_handling(mock_embedder, tmp_path: Path):
    """Verify that querying an empty collection returns [] cleanly without crashing."""
    empty_store = ChromaVectorStore(persist_dir=tmp_path / "empty_db", collection_name="empty_coll")
    retriever = Retriever(embedder=mock_embedder, vector_store=empty_store)

    results = retriever.retrieve("Hello in empty database")
    assert results == []
    # embed_query should not even be called if collection is empty
    mock_embedder.embed_query.assert_not_called()


def test_retrieval_ordering(tmp_path: Path):
    """Verify that retrieved chunks are ordered by cosine distance (nearest first)."""
    store = ChromaVectorStore(persist_dir=tmp_path / "ordered_db", collection_name="ordered_coll")
    # Insert 3 chunks with specific known vectors
    c1 = DocumentChunk("c1", "Exact match chunk", "test.pdf", 1, 0)
    c2 = DocumentChunk("c2", "Moderate match chunk", "test.pdf", 2, 0)
    c3 = DocumentChunk("c3", "Poor match chunk", "test.pdf", 3, 0)

    # Unit vector query: [1.0, 0.0, 0.0, ...]
    v_query = [0.0] * FAKE_DIM
    v_query[0] = 1.0

    # c1 is identical to query (cosine distance ~ 0)
    v1 = [0.0] * FAKE_DIM
    v1[0] = 1.0

    # c2 has some similarity
    v2 = [0.0] * FAKE_DIM
    v2[0] = 0.5
    v2[1] = 0.5

    # c3 is orthogonal
    v3 = [0.0] * FAKE_DIM
    v3[1] = 1.0

    store.add_chunks([c1, c2, c3], [v1, v2, v3])

    embedder = MagicMock(spec=GeminiEmbedder)
    embedder.embed_query.return_value = v_query

    retriever = Retriever(embedder=embedder, vector_store=store)
    results = retriever.retrieve("Unit vector query", top_k=3, similarity_threshold=0.0)

    assert len(results) == 3
    # Nearest first: distance(c1) <= distance(c2) <= distance(c3)
    assert results[0].chunk_id == "c1"
    assert results[0].distance <= results[1].distance <= results[2].distance


def test_results_contain_numeric_distance(mock_embedder, populated_vector_store):
    """Test 1: Verify every returned result contains a valid float distance."""
    retriever = Retriever(embedder=mock_embedder, vector_store=populated_vector_store)
    results = retriever.retrieve("query", top_k=3)
    assert len(results) == 3
    for res in results:
        assert isinstance(res.distance, float)
        assert res.distance >= 0.0


def test_results_ordered_correctly_from_mocked_distances(mock_embedder):
    """Test 2: Given Chroma results with distances 0.10, 0.30, 0.60, verify ordering is preserved."""
    fake_store = MagicMock(spec=ChromaVectorStore)
    fake_store.count.return_value = 3
    fake_store.query.return_value = {
        "ids": [["c1", "c2", "c3"]],
        "documents": [["doc 1", "doc 2", "doc 3"]],
        "metadatas": [[
            {"source": "doc.pdf", "page_number": 1, "chunk_index": 0},
            {"source": "doc.pdf", "page_number": 1, "chunk_index": 1},
            {"source": "doc.pdf", "page_number": 2, "chunk_index": 0},
        ]],
        "distances": [[0.10, 0.30, 0.60]],
    }
    retriever = Retriever(embedder=mock_embedder, vector_store=fake_store)
    results = retriever.retrieve("test ordering", top_k=3, similarity_threshold=0.0)

    assert len(results) == 3
    assert [r.distance for r in results] == [0.10, 0.30, 0.60]
    assert results[0].distance < results[1].distance < results[2].distance


def test_top_k_limits_1_3_5(tmp_path: Path, mock_embedder):
    """Test 3: Verify top_k=1, top_k=3, and top_k=5 limit returned result count accurately."""
    store = ChromaVectorStore(persist_dir=tmp_path / "topk_db", collection_name="topk_coll")
    chunks = [
        DocumentChunk(f"c_{i}", f"Chunk text {i}", "sample.pdf", 1, i)
        for i in range(5)
    ]
    embeddings = [make_vector(0.1 * (i + 1)) for i in range(5)]
    store.add_chunks(chunks, embeddings)

    retriever = Retriever(embedder=mock_embedder, vector_store=store)

    assert len(retriever.retrieve("q", top_k=1)) == 1
    assert len(retriever.retrieve("q", top_k=3)) == 3
    assert len(retriever.retrieve("q", top_k=5)) == 5


def test_metadata_preservation_with_document_id(tmp_path: Path, mock_embedder):
    """Test 4: Verify result retains document_id, source, page_number, chunk_index, and distance."""
    store = ChromaVectorStore(persist_dir=tmp_path / "meta_db", collection_name="meta_coll")
    chunk = DocumentChunk(
        chunk_id="att_p4_c2",
        text="Self-attention replaces recurrent layers.",
        source="attention.pdf",
        page_number=4,
        chunk_index=2,
        document_id="sha256_hash_abcdef123456",
    )
    store.add_chunks([chunk], [make_vector(0.1)])

    retriever = Retriever(embedder=mock_embedder, vector_store=store)
    results = retriever.retrieve("attention mechanism", top_k=1)

    assert len(results) == 1
    res = results[0]
    assert res.document_id == "sha256_hash_abcdef123456"
    assert res.source == "attention.pdf"
    assert res.page_number == 4
    assert res.chunk_index == 2
    assert isinstance(res.distance, float)
    assert res.text == "Self-attention replaces recurrent layers."


def test_same_filename_different_documents_distinguishable(mock_embedder):
    """Test 5: Verify chunks with identical filenames but different document_ids remain distinguishable."""
    fake_store = MagicMock(spec=ChromaVectorStore)
    fake_store.count.return_value = 2
    fake_store.query.return_value = {
        "ids": [["docA_c0", "docB_c0"]],
        "documents": [["Content from first version", "Content from second version"]],
        "metadatas": [[
            {"source": "report.pdf", "page_number": 1, "chunk_index": 0, "document_id": "aaa111"},
            {"source": "report.pdf", "page_number": 1, "chunk_index": 0, "document_id": "bbb222"},
        ]],
        "distances": [[0.12, 0.28]],
    }
    retriever = Retriever(embedder=mock_embedder, vector_store=fake_store)
    results = retriever.retrieve("report query", top_k=2)

    assert len(results) == 2
    assert results[0].source == "report.pdf"
    assert results[1].source == "report.pdf"
    assert results[0].document_id == "aaa111"
    assert results[1].document_id == "bbb222"
    assert results[0].document_id != results[1].document_id


def test_empty_collection_returns_empty_list(mock_embedder, tmp_path: Path):
    """Test 6: Verify retrieval returns empty list for an empty collection without calling embedder."""
    empty_store = ChromaVectorStore(persist_dir=tmp_path / "empty_store_db", collection_name="empty_test")
    retriever = Retriever(embedder=mock_embedder, vector_store=empty_store)

    results = retriever.retrieve("Any question")
    assert results == []
    mock_embedder.embed_query.assert_not_called()


def test_invalid_top_k_values(mock_embedder, populated_vector_store):
    """Test 7: Verify top_k=0 and top_k=-1 raise InvalidTopKError."""
    retriever = Retriever(embedder=mock_embedder, vector_store=populated_vector_store)

    with pytest.raises(InvalidTopKError, match="must be a positive integer"):
        retriever.retrieve("Valid question", top_k=0)

    with pytest.raises(InvalidTopKError, match="must be a positive integer"):
        retriever.retrieve("Valid question", top_k=-1)


def test_distance_and_similarity_semantics():
    """Test 8: Verify distance and cosine similarity semantics under cosine distance space."""
    res_zero = RetrievalResult(
        chunk_id="c0",
        text="text",
        source="doc.pdf",
        page_number=1,
        chunk_index=0,
        distance=0.0,
    )
    assert res_zero.distance == 0.0
    assert res_zero.cosine_similarity == 1.0
    assert res_zero.similarity == 1.0

    res_one = RetrievalResult(
        chunk_id="c1",
        text="text",
        source="doc.pdf",
        page_number=1,
        chunk_index=0,
        distance=1.0,
    )
    assert res_one.distance == 1.0
    assert res_one.cosine_similarity == 0.0
    assert res_one.similarity == 0.0

    res_arbitrary = RetrievalResult(
        chunk_id="c2",
        text="text",
        source="doc.pdf",
        page_number=1,
        chunk_index=0,
        distance=0.1842,
    )
    assert res_arbitrary.distance == 0.1842
    assert res_arbitrary.cosine_similarity == pytest.approx(1.0 - 0.1842)
    assert res_arbitrary.similarity == pytest.approx(1.0 - 0.1842)
    # Check RetrievedChunk is compatible
    assert isinstance(res_arbitrary, RetrievedChunk)

