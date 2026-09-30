"""
Unit tests for ContextIQ PDFLoader and DocumentPage.
"""

from pathlib import Path
import pytest
import pymupdf

from app.ingestion.pdf_loader import (
    DocumentPage,
    PDFLoader,
    PDFLoadResult,
    PDFNotFoundError,
    InvalidPDFError,
    ScannedOrEmptyPDFError,
)


@pytest.fixture
def sample_multipage_pdf(tmp_path: Path) -> Path:
    """Creates a temporary 3-page PDF with known text content."""
    pdf_path = tmp_path / "sample_doc.pdf"
    doc = pymupdf.open()

    # Page 1
    page1 = doc.new_page()
    page1.insert_text(
        (50, 72),
        "ContextIQ is an enterprise document intelligence system.\n"
        "It uses Retrieval-Augmented Generation.",
    )

    # Page 2
    page2 = doc.new_page()
    page2.insert_text(
        (50, 72),
        "PyMuPDF provides high performance text extraction.\n"
        "Page numbering starts at index 1 for human readability.",
    )

    # Page 3
    page3 = doc.new_page()
    page3.insert_text(
        (50, 72),
        "Vector search uses ChromaDB for persistent storage.\n"
        "Gemini generates grounded answers.",
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def partially_empty_pdf(tmp_path: Path) -> Path:
    """Creates a temporary 3-page PDF where page 2 has no text."""
    pdf_path = tmp_path / "partially_empty.pdf"
    doc = pymupdf.open()

    p1 = doc.new_page()
    p1.insert_text((50, 72), "Text on first page.")

    # Page 2 is completely blank
    doc.new_page()

    p3 = doc.new_page()
    p3.insert_text((50, 72), "Text on third page.")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def completely_empty_pdf(tmp_path: Path) -> Path:
    """Creates a temporary PDF with 2 blank pages (simulating scanned/image-only without OCR)."""
    pdf_path = tmp_path / "scanned_or_blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_pdf_loader_success(sample_multipage_pdf: Path):
    """Test loading a valid multi-page PDF."""
    loader = PDFLoader(sample_multipage_pdf)
    result = loader.load()

    assert isinstance(result, PDFLoadResult)
    assert result.source == "sample_doc.pdf"
    assert result.total_pages == 3
    assert len(result.pages) == 3
    assert result.empty_pages == []


def test_page_numbering_one_indexed(sample_multipage_pdf: Path):
    """Test that page numbering is 1-indexed for user readability."""
    loader = PDFLoader(sample_multipage_pdf)
    pages = loader.load_pages()

    assert [p.page_number for p in pages] == [1, 2, 3]


def test_metadata_preservation(sample_multipage_pdf: Path):
    """Test that source filename and page numbers are preserved on every DocumentPage."""
    loader = PDFLoader(sample_multipage_pdf)
    pages = loader.load_pages()

    for idx, page in enumerate(pages, start=1):
        assert isinstance(page, DocumentPage)
        assert page.source == "sample_doc.pdf"
        assert page.page_number == idx
        assert len(page.text) > 0


def test_text_extraction_content(sample_multipage_pdf: Path):
    """Test that extracted page text matches inserted content."""
    loader = PDFLoader(sample_multipage_pdf)
    pages = loader.load_pages()

    assert "ContextIQ is an enterprise document intelligence system" in pages[0].text
    assert "PyMuPDF provides high performance text extraction" in pages[1].text
    assert "Vector search uses ChromaDB" in pages[2].text


def test_whitespace_cleaning():
    """Test whitespace cleaning logic."""
    raw = "  Line 1   with    spaces.  \n\n\n\n\nLine   2 with \xa0 nonbreaking space.   \n\n"
    cleaned = PDFLoader.clean_text(raw)

    assert "Line 1 with spaces." in cleaned
    assert "Line 2 with nonbreaking space." in cleaned
    # Ensure excessive blank lines collapsed to no more than 2 newlines
    assert "\n\n\n" not in cleaned


def test_missing_file_error(tmp_path: Path):
    """Test that non-existent file raises PDFNotFoundError."""
    missing_path = tmp_path / "does_not_exist.pdf"
    loader = PDFLoader(missing_path)

    with pytest.raises(PDFNotFoundError) as exc_info:
        loader.load()

    assert "PDF file not found" in str(exc_info.value)


def test_invalid_non_pdf_file(tmp_path: Path):
    """Test that a non-PDF file or corrupted file raises InvalidPDFError."""
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_text("This is plain text, not a valid PDF header.")

    loader = PDFLoader(corrupt_path)

    with pytest.raises(InvalidPDFError) as exc_info:
        loader.load()

    assert "Failed to open PDF" in str(exc_info.value)


def test_partially_empty_pages_detected(partially_empty_pdf: Path):
    """Test detection of blank pages without failing if some text is present."""
    loader = PDFLoader(partially_empty_pdf)
    result = loader.load()

    assert result.total_pages == 3
    assert result.empty_pages == [2]
    assert len(result.pages) == 2
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 3


def test_completely_empty_or_scanned_pdf_error(completely_empty_pdf: Path):
    """Test that an image-only/blank PDF raises ScannedOrEmptyPDFError."""
    loader = PDFLoader(completely_empty_pdf)

    with pytest.raises(ScannedOrEmptyPDFError) as exc_info:
        loader.load()

    assert "contains no extractable text" in str(exc_info.value)
