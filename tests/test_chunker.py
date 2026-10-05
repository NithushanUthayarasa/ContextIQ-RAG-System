"""
Unit tests for ContextIQ TextChunker and DocumentChunk.
"""

import pytest
from app.ingestion.pdf_loader import DocumentPage
from app.ingestion.chunker import DocumentChunk, TextChunker


def test_chunker_basic_creation():
    """Test basic chunking on sample text."""
    text = "A" * 1500
    page = DocumentPage(text=text, page_number=1, source="test_doc.pdf")
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)

    chunks = chunker.split_page(page)

    assert len(chunks) == 2
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_chunk_size_limit():
    """Verify that chunks do not exceed the configured chunk_size."""
    text = "Word " * 500  # 2500 characters
    page = DocumentPage(text=text, page_number=1, source="doc.pdf")
    chunk_size = 800
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=100)

    chunks = chunker.split_page(page)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= chunk_size


def test_overlap_correctness():
    """
    Verify that consecutive chunks share exactly chunk_overlap characters at boundaries.
    """
    # Create distinct characters where character position is predictable
    alphabet_text = "".join(f"{i:04d}" for i in range(500))  # 2000 chars: '000000010002...'
    page = DocumentPage(text=alphabet_text, page_number=1, source="overlap_test.pdf")

    chunk_size = 1000
    chunk_overlap = 200
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = chunker.split_page(page)

    assert len(chunks) == 3

    # Check overlap between Chunk 0 and Chunk 1:
    # The tail of Chunk 0 of length chunk_overlap should match the head of Chunk 1
    chunk_0_tail = chunks[0].text[-chunk_overlap:]
    chunk_1_head = chunks[1].text[:chunk_overlap]
    assert chunk_0_tail == chunk_1_head
    assert len(chunk_0_tail) == chunk_overlap

    # Check overlap between Chunk 1 and Chunk 2:
    chunk_1_tail = chunks[1].text[-chunk_overlap:]
    chunk_2_head = chunks[2].text[:chunk_overlap]
    assert chunk_1_tail == chunk_2_head
    assert len(chunk_1_tail) == chunk_overlap


def test_multiple_pages():
    """Verify chunking across multiple DocumentPages with independent indexing."""
    pages = [
        DocumentPage(text="Page 1 content " * 100, page_number=1, source="multipage.pdf"),
        DocumentPage(text="Page 2 content " * 100, page_number=2, source="multipage.pdf"),
        DocumentPage(text="Page 3 content " * 100, page_number=3, source="multipage.pdf"),
    ]
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    all_chunks = chunker.split_pages(pages)

    assert len(all_chunks) > 3
    # Check that each page has chunks with page_number matching
    p1_chunks = [c for c in all_chunks if c.page_number == 1]
    p2_chunks = [c for c in all_chunks if c.page_number == 2]
    p3_chunks = [c for c in all_chunks if c.page_number == 3]

    assert len(p1_chunks) > 0
    assert len(p2_chunks) > 0
    assert len(p3_chunks) > 0

    # Indexing starts at 0 for each page
    assert p1_chunks[0].chunk_index == 0
    assert p2_chunks[0].chunk_index == 0
    assert p3_chunks[0].chunk_index == 0


def test_metadata_preservation():
    """Verify source filename, page number, and chunk index are preserved."""
    page = DocumentPage(text="ContextIQ test " * 80, page_number=4, source="research_survey.pdf")
    chunker = TextChunker(chunk_size=400, chunk_overlap=80)
    chunks = chunker.split_page(page)

    for idx, chunk in enumerate(chunks):
        assert chunk.source == "research_survey.pdf"
        assert chunk.page_number == 4
        assert chunk.chunk_index == idx
        assert chunk.chunk_id == f"research_survey_p4_c{idx}"


def test_chunk_id_generation():
    """Verify deterministic chunk ID formatting and sanitization."""
    chunker = TextChunker()

    cid1 = chunker.generate_chunk_id("rag_paper.pdf", 3, 2)
    assert cid1 == "rag_paper_p3_c2"

    cid2 = chunker.generate_chunk_id("research paper v1.0.pdf", 1, 0)
    # Replaces spaces and periods in stem with underscores
    assert cid2 == "research_paper_v1_0_p1_c0"


def test_empty_page_handling():
    """Verify that empty pages or whitespace-only pages are cleanly skipped."""
    chunker = TextChunker()
    empty_page = DocumentPage(text="", page_number=2, source="empty.pdf")
    whitespace_page = DocumentPage(text="   \n\t  \n  ", page_number=3, source="empty.pdf")

    assert chunker.split_page(empty_page) == []
    assert chunker.split_page(whitespace_page) == []

    all_chunks = chunker.split_pages([empty_page, whitespace_page])
    assert all_chunks == []


def test_short_text_handling():
    """Verify that text shorter than chunk_size produces exactly one chunk."""
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    short_page = DocumentPage(text="Short text", page_number=1, source="short.pdf")

    chunks = chunker.split_page(short_page)

    assert len(chunks) == 1
    assert chunks[0].text == "Short text"
    assert chunks[0].chunk_id == "short_p1_c0"
    assert chunks[0].chunk_index == 0


def test_custom_chunk_size_and_overlap():
    """Verify custom chunk size and overlap parameters."""
    text = "Custom chunk configuration testing. " * 30
    page = DocumentPage(text=text, page_number=1, source="custom.pdf")

    custom_chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    chunks = custom_chunker.split_page(page)

    assert custom_chunker.chunk_size == 200
    assert custom_chunker.chunk_overlap == 50
    assert custom_chunker.step_size == 150
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 200


def test_invalid_parameters():
    """Verify validation when invalid chunk size or overlap is provided."""
    # Chunk size <= 0
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        TextChunker(chunk_size=0, chunk_overlap=0)

    # Chunk overlap < 0
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        TextChunker(chunk_size=500, chunk_overlap=-10)

    # Overlap >= chunk size
    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        TextChunker(chunk_size=500, chunk_overlap=500)

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        TextChunker(chunk_size=500, chunk_overlap=600)


def test_document_id_preserved():
    """Verify document_id is preserved on DocumentChunk when provided."""
    doc_id = "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069"
    page = DocumentPage(text="ContextIQ V2 multi-doc chunking test.", page_number=1, source="doc.pdf")
    chunker = TextChunker()
    chunks = chunker.split_page(page, document_id=doc_id)

    assert len(chunks) == 1
    assert chunks[0].document_id == doc_id
    assert chunks[0].source == "doc.pdf"

    # Also test split_pages preservation
    pages = [
        DocumentPage(text="Page 1 text", page_number=1, source="doc.pdf"),
        DocumentPage(text="Page 2 text", page_number=2, source="doc.pdf"),
    ]
    all_chunks = chunker.split_pages(pages, document_id=doc_id)
    assert len(all_chunks) == 2
    assert all_chunks[0].document_id == doc_id
    assert all_chunks[1].document_id == doc_id


def test_v2_chunk_id_format():
    """Verify {full_document_id}_p{page_number}_c{chunk_index} format with full SHA-256."""
    doc_id = "a8f3b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069"
    chunker = TextChunker()

    cid = chunker.generate_chunk_id("report.pdf", page_number=3, chunk_index=2, document_id=doc_id)
    assert cid == f"{doc_id}_p3_c2"
    # Ensure full 64-char hash is preserved (not truncated)
    assert cid.startswith(doc_id)


def test_different_documents_no_collision():
    """Verify identical page/chunk on different document_id produce different chunk IDs."""
    doc_id_a = "1111111111111111111111111111111111111111111111111111111111111111"
    doc_id_b = "2222222222222222222222222222222222222222222222222222222222222222"
    chunker = TextChunker()

    cid_a = chunker.generate_chunk_id("same_name.pdf", page_number=1, chunk_index=0, document_id=doc_id_a)
    cid_b = chunker.generate_chunk_id("same_name.pdf", page_number=1, chunk_index=0, document_id=doc_id_b)

    assert cid_a != cid_b
    assert cid_a == f"{doc_id_a}_p1_c0"
    assert cid_b == f"{doc_id_b}_p1_c0"


def test_same_document_deterministic():
    """Verify identical input & document_id produces identical chunk IDs."""
    doc_id = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    chunker = TextChunker()

    cid_1 = chunker.generate_chunk_id("doc.pdf", page_number=5, chunk_index=4, document_id=doc_id)
    cid_2 = chunker.generate_chunk_id("doc.pdf", page_number=5, chunk_index=4, document_id=doc_id)

    assert cid_1 == cid_2


def test_document_id_validation():
    """Verify invalid document_id (empty or whitespace-only) raises ValueError."""
    chunker = TextChunker()

    with pytest.raises(ValueError, match="document_id must be a non-empty string"):
        chunker.generate_chunk_id("doc.pdf", 1, 0, document_id="")

    with pytest.raises(ValueError, match="document_id must be a non-empty string"):
        chunker.generate_chunk_id("doc.pdf", 1, 0, document_id="   ")

    with pytest.raises(ValueError, match="document_id must be a non-empty string"):
        DocumentChunk(
            chunk_id="chunk_1",
            text="sample text",
            source="doc.pdf",
            page_number=1,
            chunk_index=0,
            document_id="",
        )

    with pytest.raises(ValueError, match="document_id must be a non-empty string"):
        DocumentChunk(
            chunk_id="chunk_1",
            text="sample text",
            source="doc.pdf",
            page_number=1,
            chunk_index=0,
            document_id="   \t  ",
        )

