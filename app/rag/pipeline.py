"""
ContextIQ - Central RAG Pipeline Orchestrator
Coordinates retrieval of relevant document chunks and generation of grounded answers.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.generation.generator import GeminiGenerator, EmptyQuestionError
from app.retrieval.retriever import RetrievedChunk, Retriever


class RAGPipelineError(Exception):
    """Base exception for RAG pipeline failures."""
    pass


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline containing answer, sources, and context."""
    answer: str
    sources: List[Dict[str, Any]]
    retrieved_chunks: List[RetrievedChunk]
    query: str


class RAGPipeline:
    """
    Coordinates semantic retrieval and LLM generation through dependency injection.
    Decoupled from specific vector stores or generative models.
    """

    def __init__(self, retriever: Retriever, generator: GeminiGenerator):
        if retriever is None:
            raise RAGPipelineError("A valid Retriever instance must be provided.")
        if generator is None:
            raise RAGPipelineError("A valid GeminiGenerator instance must be provided.")

        self.retriever = retriever
        self.generator = generator

    @staticmethod
    def extract_sources(chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """
        Extracts unique (source, page) combinations from retrieved chunks while preserving retrieval order.
        """
        seen = set()
        sources: List[Dict[str, Any]] = []

        for chunk in chunks:
            key = (chunk.source, chunk.page_number)
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "source": chunk.source,
                        "page": chunk.page_number,
                    }
                )

        return sources

    def ask(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Executes the end-to-end RAG pipeline for a user question.

        Args:
            question: The user query string.
            top_k: Number of relevant chunks to retrieve (optional override).

        Returns:
            RAGResponse containing generated answer, cited sources, retrieved chunks, and query.

        Raises:
            EmptyQuestionError: If question is empty or whitespace-only.
            RetrieverError / GeminiGenerationError: Upstream service errors.
        """
        if not question or not question.strip():
            raise EmptyQuestionError("Question must not be empty.")

        cleaned_question = question.strip()

        # Step 1: Semantic retrieval
        retrieved_chunks = self.retriever.retrieve(
            query=cleaned_question,
            top_k=top_k,
        )

        # Step 2: Source deduplication
        sources = self.extract_sources(retrieved_chunks)

        # Step 3: Grounded generation
        answer = self.generator.generate(
            question=cleaned_question,
            retrieved_chunks=retrieved_chunks,
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=retrieved_chunks,
            query=cleaned_question,
        )
