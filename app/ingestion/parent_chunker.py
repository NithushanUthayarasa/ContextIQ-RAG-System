"""
ContextIQ - Parent-Child Chunking Module
Splits document pages into hierarchical parent sections and smaller child chunks.
Child chunks provide high-precision retrieval targets while retaining references
to their parent sections for richer generation context.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional, Tuple

from app.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PARENT_CHUNK_OVERLAP,
    DEFAULT_PARENT_CHUNK_SIZE,
)
from app.ingestion.chunker import DocumentChunk, TextChunker
from app.ingestion.pdf_loader import DocumentPage


@dataclass
class ParentChunk:
    """Represents a larger parent section that encompasses one or more child chunks."""
    parent_id: str
    text: str
    source: str
    page_number: int
    parent_index: int
    document_id: Optional[str] = None

    def __post_init__(self):
        if not self.parent_id or not isinstance(self.parent_id, str) or not self.parent_id.strip():
            raise ValueError("parent_id must be a non-empty string.")
        if not self.text or not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string.")
        if self.page_number < 1:
            raise ValueError("page_number must be >= 1.")
        if self.parent_index < 0:
            raise ValueError("parent_index must be >= 0.")
        if self.document_id is not None:
            if not isinstance(self.document_id, str) or not self.document_id.strip():
                raise ValueError("document_id must be a non-empty string when provided.")


class ParentChildChunker:
    """
    Constructs deterministic parent sections and child chunks from DocumentPages.
    Parent chunks span larger text windows (e.g., 1200-2000 chars), while child chunks
    span smaller windows (e.g., 400-700 chars) for precise vector/keyword retrieval.
    """

    def __init__(
        self,
        parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
        parent_chunk_overlap: int = DEFAULT_PARENT_CHUNK_OVERLAP,
        child_chunk_size: int = DEFAULT_CHUNK_SIZE,
        child_chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        if parent_chunk_size <= 0:
            raise ValueError(f"parent_chunk_size must be positive, got {parent_chunk_size}")
        if parent_chunk_overlap < 0:
            raise ValueError(f"parent_chunk_overlap cannot be negative, got {parent_chunk_overlap}")
        if parent_chunk_overlap >= parent_chunk_size:
            raise ValueError(
                f"parent_chunk_overlap ({parent_chunk_overlap}) must be strictly less than parent_chunk_size ({parent_chunk_size})"
            )

        if child_chunk_size <= 0:
            raise ValueError(f"child_chunk_size must be positive, got {child_chunk_size}")
        if child_chunk_overlap < 0:
            raise ValueError(f"child_chunk_overlap cannot be negative, got {child_chunk_overlap}")
        if child_chunk_overlap >= child_chunk_size:
            raise ValueError(
                f"child_chunk_overlap ({child_chunk_overlap}) must be strictly less than child_chunk_size ({child_chunk_size})"
            )

        if child_chunk_size > parent_chunk_size:
            raise ValueError(
                f"child_chunk_size ({child_chunk_size}) cannot be larger than parent_chunk_size ({parent_chunk_size})"
            )

        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.parent_step_size = parent_chunk_size - parent_chunk_overlap

        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.child_step_size = child_chunk_size - child_chunk_overlap

    @staticmethod
    def generate_parent_id(
        source: str,
        page_number: int,
        parent_index: int,
        document_id: Optional[str] = None,
    ) -> str:
        """
        Generates a deterministic parent section identifier.
        When document_id is provided: {document_id}_p{page}_parent{parent_index}
        Otherwise: {sanitized_stem}_p{page}_parent{parent_index}
        """
        if document_id is not None:
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError("document_id must be a non-empty string when provided.")
            clean_doc_id = document_id.strip()
            return f"{clean_doc_id}_p{page_number}_parent{parent_index}"

        stem = Path(source).stem if source else "doc"
        clean_stem = re.sub(r"[^\w\-]", "_", stem)
        return f"{clean_stem}_p{page_number}_parent{parent_index}"

    def split_page(
        self,
        page: DocumentPage,
        document_id: Optional[str] = None,
    ) -> Tuple[List[ParentChunk], List[DocumentChunk]]:
        """
        Splits a single DocumentPage into parent sections and associated child chunks.
        Empty or whitespace-only pages yield empty lists.
        """
        if not page.text or not page.text.strip():
            return [], []

        text = page.text.strip()
        text_len = len(text)
        parents: List[ParentChunk] = []
        children: List[DocumentChunk] = []

        start = 0
        parent_index = 0
        chunk_index = 0

        # If page text is smaller than or equal to parent_chunk_size, create a single parent
        if text_len <= self.parent_chunk_size:
            parent_id = self.generate_parent_id(
                source=page.source,
                page_number=page.page_number,
                parent_index=0,
                document_id=document_id,
            )
            parents.append(
                ParentChunk(
                    parent_id=parent_id,
                    text=text,
                    source=page.source,
                    page_number=page.page_number,
                    parent_index=0,
                    document_id=document_id,
                )
            )

            # Generate child chunks inside this single parent
            if text_len <= self.child_chunk_size:
                cid = TextChunker.generate_chunk_id(
                    source=page.source,
                    page_number=page.page_number,
                    chunk_index=0,
                    document_id=document_id,
                )
                children.append(
                    DocumentChunk(
                        chunk_id=cid,
                        text=text,
                        source=page.source,
                        page_number=page.page_number,
                        chunk_index=0,
                        document_id=document_id,
                        parent_id=parent_id,
                        parent_index=0,
                    )
                )
            else:
                c_start = 0
                while c_start < text_len:
                    c_end = min(c_start + self.child_chunk_size, text_len)
                    c_text = text[c_start:c_end].strip()
                    if c_text:
                        cid = TextChunker.generate_chunk_id(
                            source=page.source,
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                            document_id=document_id,
                        )
                        children.append(
                            DocumentChunk(
                                chunk_id=cid,
                                text=c_text,
                                source=page.source,
                                page_number=page.page_number,
                                chunk_index=chunk_index,
                                document_id=document_id,
                                parent_id=parent_id,
                                parent_index=0,
                            )
                        )
                        chunk_index += 1
                    if c_end == text_len:
                        break
                    c_start += self.child_step_size

            return parents, children

        # Sliding window for multiple parent chunks
        while start < text_len:
            end = min(start + self.parent_chunk_size, text_len)
            parent_text = text[start:end].strip()

            if parent_text:
                parent_id = self.generate_parent_id(
                    source=page.source,
                    page_number=page.page_number,
                    parent_index=parent_index,
                    document_id=document_id,
                )
                parents.append(
                    ParentChunk(
                        parent_id=parent_id,
                        text=parent_text,
                        source=page.source,
                        page_number=page.page_number,
                        parent_index=parent_index,
                        document_id=document_id,
                    )
                )

                # Generate child chunks within this parent window
                p_len = len(parent_text)
                if p_len <= self.child_chunk_size:
                    cid = TextChunker.generate_chunk_id(
                        source=page.source,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        document_id=document_id,
                    )
                    children.append(
                        DocumentChunk(
                            chunk_id=cid,
                            text=parent_text,
                            source=page.source,
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                            document_id=document_id,
                            parent_id=parent_id,
                            parent_index=parent_index,
                        )
                    )
                    chunk_index += 1
                else:
                    c_start = 0
                    while c_start < p_len:
                        c_end = min(c_start + self.child_chunk_size, p_len)
                        c_text = parent_text[c_start:c_end].strip()
                        if c_text:
                            cid = TextChunker.generate_chunk_id(
                                source=page.source,
                                page_number=page.page_number,
                                chunk_index=chunk_index,
                                document_id=document_id,
                            )
                            children.append(
                                DocumentChunk(
                                    chunk_id=cid,
                                    text=c_text,
                                    source=page.source,
                                    page_number=page.page_number,
                                    chunk_index=chunk_index,
                                    document_id=document_id,
                                    parent_id=parent_id,
                                    parent_index=parent_index,
                                )
                            )
                            chunk_index += 1
                        if c_end == p_len:
                            break
                        c_start += self.child_step_size

                parent_index += 1

            if end == text_len:
                break
            start += self.parent_step_size

        return parents, children

    def split_pages(
        self,
        pages: List[DocumentPage],
        document_id: Optional[str] = None,
    ) -> Tuple[List[ParentChunk], List[DocumentChunk]]:
        """
        Processes a list of DocumentPages and returns all resulting ParentChunks and DocumentChunks.
        """
        all_parents: List[ParentChunk] = []
        all_children: List[DocumentChunk] = []

        for page in pages:
            parents, children = self.split_page(page, document_id=document_id)
            all_parents.extend(parents)
            all_children.extend(children)

        return all_parents, all_children
