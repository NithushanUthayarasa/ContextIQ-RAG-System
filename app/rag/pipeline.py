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
    retrieval_query: Optional[str] = None
    similarity_threshold: Optional[float] = None
    document_ids: Optional[List[str]] = None
    retrieval_mode: Optional[str] = None

    def __post_init__(self):
        if self.retrieval_query is None:
            self.retrieval_query = self.query


class RAGPipeline:
    """
    Coordinates semantic/hybrid retrieval and LLM generation through dependency injection.
    Decoupled from specific vector stores, generative models, and query rewriters.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: GeminiGenerator,
        query_rewriter: Optional[Any] = None,
    ):
        if retriever is None:
            raise RAGPipelineError("A valid Retriever instance must be provided.")
        if generator is None:
            raise RAGPipelineError("A valid GeminiGenerator instance must be provided.")

        self.retriever = retriever
        self.generator = generator
        self.query_rewriter = query_rewriter

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

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        conversation_messages: Optional[List[Any]] = None,
        similarity_threshold: Optional[float] = None,
        document_ids: Optional[List[str]] = None,
        retrieval_mode: Optional[str] = None,
    ) -> RAGResponse:
        """
        Executes the end-to-end RAG pipeline for a user question.

        Args:
            question: The user query string.
            top_k: Number of relevant chunks to retrieve (optional override).
            conversation_messages: Optional sequence of prior ChatMessage objects for query rewriting.
            similarity_threshold: Optional minimum cosine similarity threshold override.
            document_ids: Optional list of document_id strings to restrict retrieval scope.
            retrieval_mode: Optional retrieval mode ("semantic", "bm25", "hybrid").

        Returns:
            RAGResponse containing generated answer, cited sources, retrieved chunks, and query.

        Raises:
            EmptyQuestionError: If question is empty or whitespace-only.
            RetrieverError / GeminiGenerationError: Upstream service errors.
        """
        if not question or not question.strip():
            raise EmptyQuestionError("Question must not be empty.")

        cleaned_question = question.strip()

        # Step 0: Context-aware query rewriting
        if self.query_rewriter is not None and conversation_messages:
            try:
                retrieval_query = self.query_rewriter.rewrite_query(
                    cleaned_question,
                    conversation_messages=conversation_messages,
                )
            except Exception:
                retrieval_query = cleaned_question
        else:
            retrieval_query = cleaned_question

        # Fallback safeguard: if rewrite produces empty or invalid text, use original question
        if not retrieval_query or not retrieval_query.strip():
            retrieval_query = cleaned_question

        # Step 1: Retrieval using standalone query with similarity threshold, document filtering, and retrieval mode
        retrieve_kwargs: Dict[str, Any] = {"query": retrieval_query, "top_k": top_k}
        if similarity_threshold is not None:
            retrieve_kwargs["similarity_threshold"] = similarity_threshold
        if document_ids is not None:
            retrieve_kwargs["document_ids"] = document_ids
        if retrieval_mode is not None:
            retrieve_kwargs["retrieval_mode"] = retrieval_mode

        retrieved_chunks = self.retriever.retrieve(**retrieve_kwargs)

        # Step 2: Source deduplication
        sources = self.extract_sources(retrieved_chunks)

        # Step 3: Grounded generation using user's original question
        answer = self.generator.generate(
            question=cleaned_question,
            retrieved_chunks=retrieved_chunks,
        )

        # Determine effective threshold and mode applied
        effective_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else getattr(self.retriever, "default_min_similarity", None)
        )
        effective_mode = (
            retrieval_mode
            if retrieval_mode is not None
            else getattr(self.retriever, "default_retrieval_mode", "semantic")
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=retrieved_chunks,
            query=cleaned_question,
            retrieval_query=retrieval_query,
            similarity_threshold=effective_threshold,
            document_ids=document_ids,
            retrieval_mode=effective_mode,
        )
