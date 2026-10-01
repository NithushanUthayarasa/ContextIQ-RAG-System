"""
ContextIQ - Central RAG Pipeline Orchestrator
Coordinates retrieval of relevant document chunks and generation of grounded answers.

Pipeline modes
--------------
Reranking disabled (default):
    Query → Retriever(top_k) → Generator

Reranking enabled:
    Query → Retriever(candidate_k) → Reranker(top_k) → Generator

The original user question is always passed to the generator.
The rewritten retrieval query is used for both retrieval and reranking.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import QUERY_EXPANSION_MAX_QUERIES
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
    # Reranking transparency fields
    reranking_enabled: bool = False
    candidates_retrieved: Optional[int] = None
    # Query expansion transparency fields
    query_expansion_enabled: bool = False
    expanded_queries: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.retrieval_query is None:
            self.retrieval_query = self.query


class RAGPipeline:
    """
    Coordinates semantic/hybrid retrieval, optional reranking, and LLM generation
    through dependency injection.

    Decoupled from specific vector stores, generative models, query rewriters,
    and rerankers.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: GeminiGenerator,
        query_rewriter: Optional[Any] = None,
        reranker: Optional[Any] = None,
        reranker_candidate_multiplier: int = 3,
        query_expander: Optional[Any] = None,
    ):
        if retriever is None:
            raise RAGPipelineError("A valid Retriever instance must be provided.")
        if generator is None:
            raise RAGPipelineError("A valid GeminiGenerator instance must be provided.")

        self.retriever = retriever
        self.generator = generator
        self.query_rewriter = query_rewriter
        self.reranker = reranker
        self.reranker_candidate_multiplier = max(1, int(reranker_candidate_multiplier))
        self.query_expander = query_expander

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
            top_k: Number of final chunks to pass to the generator.
            conversation_messages: Optional prior ChatMessage objects for query rewriting.
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

        # Step 1: Query Expansion (optional)
        query_expansion_enabled = self.query_expander is not None
        expanded_queries = [retrieval_query]

        if query_expansion_enabled:
            try:
                expanded = self.query_expander.expand(
                    retrieval_query,
                    max_queries=QUERY_EXPANSION_MAX_QUERIES,
                )
                if expanded:
                    # Ensure original query is first
                    if retrieval_query not in expanded:
                        expanded = [retrieval_query] + expanded
                    else:
                        expanded = [retrieval_query] + [q for q in expanded if q != retrieval_query]
                    expanded_queries = expanded
            except Exception:
                expanded_queries = [retrieval_query]

        # Step 2: Retrieval
        reranking_active = self.reranker is not None

        if reranking_active:
            final_k = top_k if top_k is not None else getattr(self.retriever, "default_top_k", 5)
            candidate_k = final_k * self.reranker_candidate_multiplier
        else:
            final_k = top_k
            candidate_k = top_k

        # Retrieval for each expanded query
        all_candidates: List[RetrievedChunk] = []
        for q in expanded_queries:
            retrieve_kwargs: Dict[str, Any] = {"query": q, "top_k": candidate_k}
            if similarity_threshold is not None:
                retrieve_kwargs["similarity_threshold"] = similarity_threshold
            if document_ids is not None:
                retrieve_kwargs["document_ids"] = document_ids
            if retrieval_mode is not None:
                retrieve_kwargs["retrieval_mode"] = retrieval_mode

            chunks = self.retriever.retrieve(**retrieve_kwargs)
            all_candidates.extend(chunks)

        # Deduplicate candidates, keeping strongest evidence per chunk_id
        seen_ids: Dict[str, RetrievedChunk] = {}
        for chunk in all_candidates:
            cid = getattr(chunk, "chunk_id", None)
            if cid in seen_ids:
                prev = seen_ids[cid]
                better = False
                if hasattr(chunk, "distance") and hasattr(prev, "distance"):
                    better = chunk.distance < prev.distance
                elif hasattr(chunk, "cosine_similarity") and hasattr(prev, "cosine_similarity"):
                    better = chunk.cosine_similarity > prev.cosine_similarity
                elif hasattr(chunk, "bm25_score") and hasattr(prev, "bm25_score"):
                    better = chunk.bm25_score > prev.bm25_score
                if better:
                    seen_ids[cid] = chunk
            else:
                seen_ids[cid] = chunk
        deduped: List[RetrievedChunk] = list(seen_ids.values())

        candidates_retrieved = len(deduped)

        # Apply reranking if enabled
        if reranking_active and deduped:
            final_chunks = self.reranker.rerank(
                query=retrieval_query,
                candidates=deduped,
                top_k=final_k,
            )
        else:
            final_chunks = deduped

        # Source deduplication and generation as before
        sources = self.extract_sources(final_chunks)
        answer = self.generator.generate(
            question=cleaned_question,
            retrieved_chunks=final_chunks,
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
            retrieved_chunks=final_chunks,
            query=cleaned_question,
            retrieval_query=retrieval_query,
            similarity_threshold=effective_threshold,
            document_ids=document_ids,
            retrieval_mode=effective_mode,
            reranking_enabled=reranking_active,
            candidates_retrieved=candidates_retrieved,
            query_expansion_enabled=query_expansion_enabled,
            expanded_queries=expanded_queries,
        )
