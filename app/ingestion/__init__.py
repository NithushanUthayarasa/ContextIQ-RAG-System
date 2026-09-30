"""Ingestion package: PDF loading, chunking, and embedding."""

from app.ingestion.pdf_loader import (
    DocumentPage,
    PDFLoader,
    PDFLoadResult,
    PDFLoaderError,
    PDFNotFoundError,
    InvalidPDFError,
    ScannedOrEmptyPDFError,
)
from app.ingestion.chunker import (
    DocumentChunk,
    TextChunker,
)

__all__ = [
    "DocumentPage",
    "PDFLoader",
    "PDFLoadResult",
    "PDFLoaderError",
    "PDFNotFoundError",
    "InvalidPDFError",
    "ScannedOrEmptyPDFError",
    "DocumentChunk",
    "TextChunker",
]
