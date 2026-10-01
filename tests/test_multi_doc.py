"""
ContextIQ - Phase 11 Multi-Document Integration Tests.
Validates multi-document identity, chunking, storage, deduplication, retrieval,
and legacy record handling in isolated temporary environments without real API calls.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app.ingestion.chunker import DocumentChunk, TextChunker
from app.ingestion.embedder import GeminiEmbedder
from app.ingestion.pdf_loader import DocumentPage
from app.vectorstore.chroma_store import ChromaVectorStore
from app.retrieval.retriever import RetrievedChunk, Retriever
from app.rag.pipeline import RAGPipeline

FAKE_DIM = 768


def make_vector(val: float = 0.1, dim: int = FAKE_DIM) -> list:
    """Helper to produce predictable test embedding vectors."""
    return [float(val)] * dim


# ---------------------------------------------------------------------------
# Test 1 — Different documents have unique chunk IDs
# ---------------------------------------------------------------------------
def test_different_documents_have_unique_chunk_ids():
    """
    Verify two distinct documents sharing the same filename, page number, and chunk index
    generate different chunk IDs based on their unique document_id.
    """
    doc_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    doc_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    page_a = DocumentPage(text="Content for Document A", page_number=1, source="report.pdf")
    page_b = DocumentPage(text="Content for Document B", page_number=1, source="report.pdf")

    chunker = TextChunker()
    chunks_a = chunker.split_page(page_a, document_id=doc_a)
    chunks_b = chunker.split_page(page_b, document_id=doc_b)

    assert len(chunks_a) == 1
    assert len(chunks_b) == 1

    chunk_a = chunks_a[0]
    chunk_b = chunks_b[0]

    assert chunk_a.chunk_id != chunk_b.chunk_id
    assert chunk_a.chunk_id == f"{doc_a}_p1_c0"
    assert chunk_b.chunk_id == f"{doc_b}_p1_c0"
    assert chunk_a.document_id == doc_a
    assert chunk_b.document_id == doc_b


# ---------------------------------------------------------------------------
# Test 2 — Same document produces deterministic IDs
# ---------------------------------------------------------------------------
def test_same_document_produces_deterministic_ids():
    """
    Verify chunking the same document content twice with identical configuration
    produces identical, deterministic chunk IDs.
    """
    doc_id = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    text = "Deterministic chunking content across repeated invocations."
    page1 = DocumentPage(text=text, page_number=2, source="survey.pdf")
    page2 = DocumentPage(text=text, page_number=2, source="survey.pdf")

    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    run1 = chunker.split_page(page1, document_id=doc_id)
    run2 = chunker.split_page(page2, document_id=doc_id)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.document_id == c2.document_id
        assert c1.text == c2.text


# ---------------------------------------------------------------------------
# Test 3 — Same content with different filenames
# ---------------------------------------------------------------------------
def test_same_content_different_filenames(tmp_path: Path):
    """
    Verify that content with identical document_id but different filenames
    does not create duplicate document records in ChromaDB.
    """
    store = ChromaVectorStore(persist_dir=tmp_path / "test_same_content")
    doc_id = "shared_sha256_hash_content_12345"

    # Same content uploaded initially as report.pdf
    chunk_v1 = DocumentChunk(
        chunk_id=f"{doc_id}_p1_c0",
        text="Content identical across file renames.",
        source="report.pdf",
        page_number=1,
        chunk_index=0,
        document_id=doc_id,
    )
    store.add_chunks([chunk_v1], [make_vector(0.1)])
    assert store.count() == 1

    # Same content uploaded again under a renamed filename
    chunk_v2 = DocumentChunk(
        chunk_id=f"{doc_id}_p1_c0",
        text="Content identical across file renames.",
        source="renamed_report.pdf",
        page_number=1,
        chunk_index=0,
        document_id=doc_id,
    )
    store.add_chunks([chunk_v2], [make_vector(0.1)])

    # Upsert updates the existing chunk ID rather than creating a new identity
    assert store.count() == 1
    docs = store.list_indexed_documents()
    assert len(docs) == 1
    assert docs[0]["document_id"] == doc_id
    assert docs[0]["chunk_count"] == 1


# ---------------------------------------------------------------------------
# Test 4 — Same filename with different content
# ---------------------------------------------------------------------------
def test_same_filename_different_content(tmp_path: Path):
    """
    Verify that two documents with identical filenames ('report.pdf') but different
    content hashes coexist as two distinct documents in ChromaDB.
    """
    store = ChromaVectorStore(persist_dir=tmp_path / "test_duplicate_filenames")
    doc_a = "1111111111111111111111111111111111111111111111111111111111111111"
    doc_b = "2222222222222222222222222222222222222222222222222222222222222222"

    chunk_a = DocumentChunk(
        chunk_id=f"{doc_a}_p1_c0",
        text="Version A financial figures.",
        source="report.pdf",
        page_number=1,
        chunk_index=0,
        document_id=doc_a,
    )
    chunk_b = DocumentChunk(
        chunk_id=f"{doc_b}_p1_c0",
        text="Version B updated figures.",
        source="report.pdf",
        page_number=1,
        chunk_index=0,
        document_id=doc_b,
    )

    store.add_chunks([chunk_a, chunk_b], [make_vector(0.1), make_vector(0.2)])
    assert store.count() == 2

    assert store.has_document(doc_a) is True
    assert store.has_document(doc_b) is True

    indexed_docs = store.list_indexed_documents()
    assert len(indexed_docs) == 2

    doc_ids = {d["document_id"] for d in indexed_docs}
    assert doc_ids == {doc_a, doc_b}
    for d in indexed_docs:
        assert d["source"] == "report.pdf"


# ---------------------------------------------------------------------------
# Test 5 — Delete one document only
# ---------------------------------------------------------------------------
def test_delete_one_document_only(tmp_path: Path):
    """
    Verify deleting document A removes all its chunks while leaving document B intact.
    """
    store = ChromaVectorStore(persist_dir=tmp_path / "test_delete_isolation")
    doc_a = "doc_a_unique_hash"
    doc_b = "doc_b_unique_hash"

    chunks = [
        DocumentChunk(f"{doc_a}_p1_c0", "Doc A chunk 1", "docA.pdf", 1, 0, doc_a),
        DocumentChunk(f"{doc_a}_p1_c1", "Doc A chunk 2", "docA.pdf", 1, 1, doc_a),
        DocumentChunk(f"{doc_a}_p2_c0", "Doc A chunk 3", "docA.pdf", 2, 0, doc_a),
        DocumentChunk(f"{doc_b}_p1_c0", "Doc B chunk 1", "docB.pdf", 1, 0, doc_b),
        DocumentChunk(f"{doc_b}_p1_c1", "Doc B chunk 2", "docB.pdf", 1, 1, doc_b),
    ]
    store.add_chunks(chunks, [make_vector(0.1)] * 5)
    assert store.count() == 5

    # Delete Document A
    deleted = store.delete_by_document_id(doc_a)
    assert deleted == 3

    assert store.count() == 2
    assert store.has_document(doc_a) is False
    assert store.has_document(doc_b) is True

    remaining = store.list_indexed_documents()
    assert len(remaining) == 1
    assert remaining[0]["document_id"] == doc_b
    assert remaining[0]["chunk_count"] == 2


# ---------------------------------------------------------------------------
# Test 6 — Document statistics (page_count vs chunk_count)
# ---------------------------------------------------------------------------
def test_document_statistics_calculation(tmp_path: Path):
    """
    Verify that page_count reflects distinct pages (not chunks) and chunk_count is accurate.
    Document A:
      page 1 -> 2 chunks
      page 2 -> 3 chunks
      page 3 -> 1 chunk
    Expected: page_count == 3, chunk_count == 6
    """
    store = ChromaVectorStore(persist_dir=tmp_path / "test_stats")
    doc_id = "stats_test_hash"

    chunks = [
        DocumentChunk(f"{doc_id}_p1_c0", "Page 1 c0", "doc.pdf", 1, 0, doc_id),
        DocumentChunk(f"{doc_id}_p1_c1", "Page 1 c1", "doc.pdf", 1, 1, doc_id),
        DocumentChunk(f"{doc_id}_p2_c0", "Page 2 c0", "doc.pdf", 2, 0, doc_id),
        DocumentChunk(f"{doc_id}_p2_c1", "Page 2 c1", "doc.pdf", 2, 1, doc_id),
        DocumentChunk(f"{doc_id}_p2_c2", "Page 2 c2", "doc.pdf", 2, 2, doc_id),
        DocumentChunk(f"{doc_id}_p3_c0", "Page 3 c0", "doc.pdf", 3, 0, doc_id),
    ]
    store.add_chunks(chunks, [make_vector(0.1)] * 6)

    docs = store.list_indexed_documents()
    assert len(docs) == 1
    doc_info = docs[0]
    assert doc_info["document_id"] == doc_id
    assert doc_info["page_count"] == 3
    assert doc_info["chunk_count"] == 6


# ---------------------------------------------------------------------------
# Test 7 — Cross-document retrieval
# ---------------------------------------------------------------------------
def test_cross_document_retrieval(tmp_path: Path):
    """
    Verify semantic retrieval across multiple documents stored in ChromaDB.
    """
    store = ChromaVectorStore(persist_dir=tmp_path / "test_cross_retrieval")
    doc_a = "hash_attention_paper"
    doc_b = "hash_rag_paper"

    # Distinct orthogonal vectors for clear separation
    vec_a = [0.0] * FAKE_DIM
    vec_a[0] = 1.0  # Points along axis 0

    vec_b = [0.0] * FAKE_DIM
    vec_b[1] = 1.0  # Points along axis 1

    chunks = [
        DocumentChunk(
            f"{doc_a}_p1_c0",
            "Transformers use self-attention mechanisms.",
            "transformers.pdf",
            1,
            0,
            doc_a,
        ),
        DocumentChunk(
            f"{doc_b}_p1_c0",
            "Retrieval augmented generation retrieves relevant external documents.",
            "rag.pdf",
            1,
            0,
            doc_b,
        ),
    ]
    store.add_chunks(chunks, [vec_a, vec_b])
    assert store.count() == 2

    # Query 1 aligns with Document A
    embedder = MagicMock(spec=GeminiEmbedder)
    embedder.embed_query.return_value = vec_a

    retriever = Retriever(embedder=embedder, vector_store=store)
    results_a = retriever.retrieve("Tell me about self-attention", top_k=2)

    assert len(results_a) == 2
    # Nearest result should be Doc A
    assert results_a[0].source == "transformers.pdf"
    assert "Transformers" in results_a[0].text
    assert results_a[0].distance < results_a[1].distance

    # Query 2 aligns with Document B
    embedder.embed_query.return_value = vec_b
    results_b = retriever.retrieve("Explain retrieval augmented generation", top_k=2)

    assert len(results_b) == 2
    # Nearest result should be Doc B
    assert results_b[0].source == "rag.pdf"
    assert "Retrieval augmented" in results_b[0].text
    assert results_b[0].distance < results_b[1].distance


# ---------------------------------------------------------------------------
# Test 8 — Cross-document source extraction
# ---------------------------------------------------------------------------
def test_cross_document_source_extraction():
    """
    Verify RAGPipeline.extract_sources() deduplicates across multiple documents
    and preserves original retrieval ranking order.
    """
    retrieved = [
        RetrievedChunk("c1", "chunk 1", "docA.pdf", 1, 0, 0.1),
        RetrievedChunk("c2", "chunk 2", "docB.pdf", 3, 0, 0.2),
        RetrievedChunk("c3", "chunk 3", "docA.pdf", 1, 1, 0.3),  # Duplicate (docA, page 1)
    ]

    sources = RAGPipeline.extract_sources(retrieved)

    assert len(sources) == 2
    assert sources[0] == {"source": "docA.pdf", "page": 1}
    assert sources[1] == {"source": "docB.pdf", "page": 3}


# ---------------------------------------------------------------------------
# Test 9 — Empty document collection
# ---------------------------------------------------------------------------
def test_empty_document_collection(tmp_path: Path):
    """
    Verify that list_indexed_documents() returns [] and retrieval returns []
    cleanly on an empty collection.
    """
    store = ChromaVectorStore(persist_dir=tmp_path / "test_empty_store")
    assert store.count() == 0
    assert store.list_indexed_documents() == []

    mock_embedder = MagicMock(spec=GeminiEmbedder)
    retriever = Retriever(embedder=mock_embedder, vector_store=store)
    results = retriever.retrieve("Any query in empty store")
    assert results == []
    mock_embedder.embed_query.assert_not_called()


# ---------------------------------------------------------------------------
# Test 10 — Upsert idempotency
# ---------------------------------------------------------------------------
def test_upsert_idempotency(tmp_path: Path):
    """
    Verify that re-indexing the same chunks with deterministic chunk IDs
    does not create duplicate chunks (N remains N, not 2N).
    """
    store = ChromaVectorStore(persist_dir=tmp_path / "test_idempotency")
    doc_id = "idempotent_doc_hash"

    chunks = [
        DocumentChunk(f"{doc_id}_p1_c0", "Text chunk 1", "doc.pdf", 1, 0, doc_id),
        DocumentChunk(f"{doc_id}_p1_c1", "Text chunk 2", "doc.pdf", 1, 1, doc_id),
        DocumentChunk(f"{doc_id}_p2_c0", "Text chunk 3", "doc.pdf", 2, 0, doc_id),
    ]
    embs = [make_vector(0.1), make_vector(0.2), make_vector(0.3)]

    # First insert
    store.add_chunks(chunks, embs)
    assert store.count() == 3

    # Second insert with identical chunk IDs
    store.add_chunks(chunks, embs)
    assert store.count() == 3

    docs = store.list_indexed_documents()
    assert len(docs) == 1
    assert docs[0]["chunk_count"] == 3


# ---------------------------------------------------------------------------
# Test 11 — Legacy V1 records
# ---------------------------------------------------------------------------
def test_legacy_v1_records_handling(tmp_path: Path):
    """
    Verify that ChromaDB records without document_id (legacy V1 data)
    are handled gracefully by list_indexed_documents() without crashing.
    """
    store = ChromaVectorStore(persist_dir=tmp_path / "test_legacy")

    # Insert chunk without document_id (V1 format)
    legacy_chunk = DocumentChunk(
        chunk_id="legacy_doc_p1_c0",
        text="Legacy V1 chunk text without document_id.",
        source="legacy_paper.pdf",
        page_number=1,
        chunk_index=0,
        document_id=None,
    )
    store.add_chunks([legacy_chunk], [make_vector(0.1)])
    assert store.count() == 1

    docs = store.list_indexed_documents()
    assert len(docs) == 1
    assert docs[0]["source"] == "legacy_paper.pdf"
    assert docs[0]["chunk_count"] == 1
    assert docs[0]["page_count"] == 1
    assert docs[0]["document_id"] == "legacy_legacy_paper.pdf"
