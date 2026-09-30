"""
ContextIQ - PDF Loader & Text Extraction Module
Extracts text and metadata page-by-page from PDFs using PyMuPDF.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union
import re
import pymupdf


class PDFLoaderError(Exception):
    """Base exception for PDF loading errors."""
    pass


class PDFNotFoundError(PDFLoaderError, FileNotFoundError):
    """Raised when the specified PDF file does not exist."""
    pass


class InvalidPDFError(PDFLoaderError):
    """Raised when the file is not a valid or readable PDF."""
    pass


class ScannedOrEmptyPDFError(PDFLoaderError):
    """Raised when a PDF contains no extractable text (e.g. scanned/image-only)."""
    pass


@dataclass
class DocumentPage:
    """Represents a single extracted page from a document."""
    text: str
    page_number: int  # 1-indexed
    source: str


@dataclass
class PDFLoadResult:
    """Represents the complete result of loading and extracting a PDF."""
    pages: List[DocumentPage]
    source: str
    total_pages: int
    empty_pages: List[int] = field(default_factory=list)

    @property
    def has_empty_pages(self) -> bool:
        return len(self.empty_pages) > 0

    @property
    def non_empty_page_count(self) -> int:
        return len(self.pages)


class PDFLoader:
    """Loads a PDF file and extracts text page-by-page with PyMuPDF."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.filename = self.file_path.name

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans extracted text by normalizing whitespace while preserving paragraphs.
        """
        if not text:
            return ""

        # Normalize newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Replace non-breaking spaces
        text = text.replace("\xa0", " ")
        # Reduce horizontal whitespace on each line
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        # Rejoin and collapse excessive blank lines (max 2 consecutive newlines)
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def load(self) -> PDFLoadResult:
        """
        Extracts text from the PDF file page-by-page.

        Returns:
            PDFLoadResult containing extracted DocumentPage objects and metadata.

        Raises:
            PDFNotFoundError: If the file does not exist.
            InvalidPDFError: If the file cannot be opened as a valid PDF.
            ScannedOrEmptyPDFError: If the entire document contains no extractable text.
        """
        if not self.file_path.exists():
            raise PDFNotFoundError(f"PDF file not found: {self.file_path}")

        if not self.file_path.is_file():
            raise InvalidPDFError(f"Path is not a regular file: {self.file_path}")

        try:
            doc = pymupdf.open(str(self.file_path))
        except Exception as e:
            raise InvalidPDFError(f"Failed to open PDF '{self.filename}': {str(e)}") from e

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise InvalidPDFError(f"PDF '{self.filename}' has 0 pages.")

        extracted_pages: List[DocumentPage] = []
        empty_pages: List[int] = []

        try:
            for idx, page in enumerate(doc):
                page_number = idx + 1  # 1-indexed for user readability
                raw_text = page.get_text("text") or ""
                cleaned_text = self.clean_text(raw_text)

                if cleaned_text:
                    extracted_pages.append(
                        DocumentPage(
                            text=cleaned_text,
                            page_number=page_number,
                            source=self.filename,
                        )
                    )
                else:
                    empty_pages.append(page_number)
        finally:
            doc.close()

        # If no page had any extractable text, raise ScannedOrEmptyPDFError
        if not extracted_pages:
            raise ScannedOrEmptyPDFError(
                f"The PDF '{self.filename}' contains no extractable text. "
                "It may be scanned or image-only, which is not currently supported (OCR disabled)."
            )

        return PDFLoadResult(
            pages=extracted_pages,
            source=self.filename,
            total_pages=total_pages,
            empty_pages=empty_pages,
        )

    def load_pages(self) -> List[DocumentPage]:
        """Convenience method returning the list of DocumentPage objects directly."""
        return self.load().pages
