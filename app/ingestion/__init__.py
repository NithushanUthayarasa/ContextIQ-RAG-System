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
from app.ingestion.embedder import (
    GeminiEmbedder,
    GeminiEmbedderError,
    MissingAPIKeyError,
    EmptyInputError,
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
    "GeminiEmbedder",
    "GeminiEmbedderError",
    "MissingAPIKeyError",
    "EmptyInputError",
]
