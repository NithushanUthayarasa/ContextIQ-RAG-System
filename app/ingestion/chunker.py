"""
ContextIQ - Text Chunking Module
Splits document pages into overlapping character-based chunks while preserving source metadata and page numbering.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional

from app.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from app.ingestion.pdf_loader import DocumentPage


@dataclass
class DocumentChunk:
    """Represents a single chunk of text with traceable document metadata."""
    chunk_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int

    def __post_init__(self):
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty.")
        if not self.text:
            raise ValueError("text must not be empty.")
        if self.page_number < 1:
            raise ValueError("page_number must be >= 1.")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be >= 0.")


class TextChunker:
    """
    Splits text from DocumentPages into overlapping chunks.
    Uses sliding character-based windows for transparent, deterministic chunking.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step_size = chunk_size - chunk_overlap

    @staticmethod
    def generate_chunk_id(source: str, page_number: int, chunk_index: int) -> str:
        """
        Generates a deterministic chunk identifier: {sanitized_stem}_p{page}_c{index}
        Example: research_paper.pdf, page 1, chunk 0 -> research_paper_p1_c0
        """
        stem = Path(source).stem if source else "doc"
        # Sanitize filename stem to contain only alphanumeric and underscores/hyphens
        clean_stem = re.sub(r"[^\w\-]", "_", stem)
        return f"{clean_stem}_p{page_number}_c{chunk_index}"

    def split_page(self, page: DocumentPage) -> List[DocumentChunk]:
        """
        Splits a single DocumentPage into overlapping DocumentChunk objects.
        Empty or whitespace-only pages yield an empty list.
        """
        if not page.text or not page.text.strip():
            return []

        text = page.text.strip()
        text_len = len(text)
        chunks: List[DocumentChunk] = []
        chunk_index = 0

        # If page text is smaller than or equal to chunk_size, return a single chunk
        if text_len <= self.chunk_size:
            chunk_id = self.generate_chunk_id(page.source, page.page_number, chunk_index)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source=page.source,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                )
            )
            return chunks

        start = 0
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = self.generate_chunk_id(page.source, page.page_number, chunk_index)
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source=page.source,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            if end == text_len:
                break

            start += self.step_size

        return chunks

    def split_pages(self, pages: List[DocumentPage]) -> List[DocumentChunk]:
        """
        Processes a list of DocumentPages and returns all resulting DocumentChunks.
        Pages with no text are skipped.
        """
        all_chunks: List[DocumentChunk] = []
        for page in pages:
            page_chunks = self.split_page(page)
            all_chunks.extend(page_chunks)
        return all_chunks
