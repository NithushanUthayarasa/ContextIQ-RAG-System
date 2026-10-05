"""
Tests for app/retrieval/reranker.py — TFIDFReranker and BaseReranker interface.

All tests are deterministic and mock-free (reranker itself has no external deps).
RAG pipeline tests use MagicMock so no Gemini API is called.
"""

from __future__ import annotations

import math
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import (
    BaseReranker,
    InvalidTopKError,
    RerankerError,
    TFIDFReranker,
    _tfidf_score,
    _tokenize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(
    chunk_id: str = "c1",
    text: str = "sample text",
    source: str = "doc.pdf",
    page_number: int = 1,
    chunk_index: int = 0,
    distance: float = 0.2,
    bm25_score: float | None = None,
    rrf_score: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page_number=page_number,
        chunk_index=chunk_index,
        distance=distance,
        bm25_score=bm25_score,
        rrf_score=rrf_score,
        retrieval_method="semantic",
    )


reranker = TFIDFReranker()


# ---------------------------------------------------------------------------
# 1. BaseReranker is abstract
# ---------------------------------------------------------------------------

def test_base_reranker_is_abstract():
    with pytest.raises(TypeError):
        BaseReranker()  # type: ignore


# ---------------------------------------------------------------------------
# 2. Empty candidate list
# ---------------------------------------------------------------------------

def test_rerank_empty_candidates():
    result = reranker.rerank("query", [], top_k=5)
    assert result == []


# ---------------------------------------------------------------------------
# 3. Single candidate
# ---------------------------------------------------------------------------

def test_rerank_single_candidate():
    chunk = make_chunk(chunk_id="c1", text="retrieval augmented generation")
    result = reranker.rerank("retrieval augmented generation", [chunk], top_k=1)
    assert len(result) == 1
    assert result[0].chunk_id == "c1"
    assert result[0].rerank_score is not None
    assert result[0].rerank_score >= 0.0


# ---------------------------------------------------------------------------
# 4. Multiple candidates — output limited to top_k
# ---------------------------------------------------------------------------

def test_rerank_limits_to_top_k():
    chunks = [make_chunk(chunk_id=f"c{i}", text=f"text chunk {i}") for i in range(10)]
    result = reranker.rerank("text", chunks, top_k=3)
    assert len(result) == 3


def test_rerank_top_k_larger_than_candidates():
    chunks = [make_chunk(chunk_id=f"c{i}", text=f"text {i}") for i in range(2)]
    result = reranker.rerank("text", chunks, top_k=10)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# 5. Candidates reordered by relevance
# ---------------------------------------------------------------------------

def test_rerank_orders_by_relevance():
    """Chunk with more matching terms should rank higher."""
    irrelevant = make_chunk(chunk_id="low", text="completely unrelated topic about cooking")
    relevant = make_chunk(chunk_id="high", text="BM25 is a ranking function used in information retrieval BM25")
    result = reranker.rerank("BM25 information retrieval", [irrelevant, relevant], top_k=2)
    assert result[0].chunk_id == "high"


# ---------------------------------------------------------------------------
# 6. Original metadata preserved
# ---------------------------------------------------------------------------

def test_rerank_preserves_original_metadata():
    chunk = make_chunk(
        chunk_id="meta_c",
        text="some text",
        source="paper.pdf",
        page_number=7,
        chunk_index=3,
        distance=0.15,
    )
    result = reranker.rerank("some", [chunk], top_k=1)
    r = result[0]
    assert r.chunk_id == "meta_c"
    assert r.source == "paper.pdf"
    assert r.page_number == 7
    assert r.chunk_index == 3
    assert r.distance == 0.15
    assert r.retrieval_method == "semantic"


# ---------------------------------------------------------------------------
# 7. Original retrieval scores preserved (distance, bm25_score, rrf_score)
# ---------------------------------------------------------------------------

def test_rerank_preserves_retrieval_scores():
    chunk = make_chunk(chunk_id="c1", text="hybrid chunk", distance=0.22, bm25_score=3.5, rrf_score=0.012)
    result = reranker.rerank("hybrid", [chunk], top_k=1)
    r = result[0]
    assert r.distance == 0.22
    assert r.bm25_score == 3.5
    assert r.rrf_score == 0.012


# ---------------------------------------------------------------------------
# 8. rerank_score and original_rank populated correctly
# ---------------------------------------------------------------------------

def test_rerank_sets_rerank_score_and_original_rank():
    chunks = [
        make_chunk(chunk_id="c1", text="retrieval augmented generation"),
        make_chunk(chunk_id="c2", text="database schema authentication"),
    ]
    result = reranker.rerank("retrieval augmented", chunks, top_k=2)
    for r in result:
        assert r.rerank_score is not None
        assert r.original_rank is not None
        assert r.original_rank >= 1


def test_rerank_original_rank_reflects_input_order():
    chunk_a = make_chunk(chunk_id="a", text="apple banana cherry")
    chunk_b = make_chunk(chunk_id="b", text="delta echo foxtrot")
    result = reranker.rerank("delta", [chunk_a, chunk_b], top_k=2)
    # chunk_b appears first in output (higher score for "delta")
    top = result[0]
    assert top.chunk_id == "b"
    assert top.original_rank == 2  # It was 2nd in input


# ---------------------------------------------------------------------------
# 9. relevant result at rank 1
# ---------------------------------------------------------------------------

def test_relevant_at_rank_1():
    best = make_chunk(chunk_id="best", text="authentication JWT role based access control")
    worst = make_chunk(chunk_id="worst", text="cherry pie recipe ingredients")
    result = reranker.rerank("JWT authentication", [best, worst], top_k=2)
    assert result[0].chunk_id == "best"
    assert result[0].original_rank == 1


# ---------------------------------------------------------------------------
# 10. Relevant result at rank 3 in output
# ---------------------------------------------------------------------------

def test_relevant_at_rank_3():
    chunks = [
        make_chunk(chunk_id="c1", text="completely unrelated alpha"),
        make_chunk(chunk_id="c2", text="completely unrelated beta"),
        make_chunk(chunk_id="c3", text="PostgreSQL schema design authentication"),
    ]
    result = reranker.rerank("PostgreSQL authentication", chunks, top_k=3)
    ids = [r.chunk_id for r in result]
    assert "c3" in ids
    assert ids.index("c3") == 0  # c3 should float to the top


# ---------------------------------------------------------------------------
# 11. Deterministic output
# ---------------------------------------------------------------------------

def test_rerank_is_deterministic():
    chunks = [make_chunk(chunk_id=f"c{i}", text=f"token{i} alpha beta gamma") for i in range(5)]
    r1 = reranker.rerank("alpha beta", chunks, top_k=5)
    r2 = reranker.rerank("alpha beta", chunks, top_k=5)
    assert [c.chunk_id for c in r1] == [c.chunk_id for c in r2]
    assert [c.rerank_score for c in r1] == [c.rerank_score for c in r2]


# ---------------------------------------------------------------------------
# 12. Invalid top_k rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_k", [0, -1, -100, 0.5, "five", None, True])
def test_rerank_invalid_top_k_raises(bad_k):
    chunks = [make_chunk()]
    with pytest.raises(InvalidTopKError):
        reranker.rerank("query", chunks, top_k=bad_k)  # type: ignore


# ---------------------------------------------------------------------------
# 13. Duplicate chunk IDs handled safely (deduplicated)
# ---------------------------------------------------------------------------

def test_rerank_deduplicates_chunk_ids():
    chunk = make_chunk(chunk_id="dup", text="duplicate chunk text about retrieval")
    duplicated = [chunk, chunk, chunk]
    result = reranker.rerank("retrieval", duplicated, top_k=5)
    ids = [r.chunk_id for r in result]
    assert ids.count("dup") == 1


# ---------------------------------------------------------------------------
# 14. TFIDFReranker does NOT mutate input objects
# ---------------------------------------------------------------------------

def test_rerank_does_not_mutate_input():
    chunk = make_chunk(chunk_id="orig", text="immutable chunk")
    original_rerank_score = chunk.rerank_score
    original_original_rank = chunk.original_rank
    reranker.rerank("immutable", [chunk], top_k=1)
    # Original object unchanged
    assert chunk.rerank_score == original_rerank_score
    assert chunk.original_rank == original_original_rank


# ---------------------------------------------------------------------------
# 15. Reranking disabled: pipeline does NOT call reranker
# ---------------------------------------------------------------------------

def test_pipeline_reranking_disabled_no_reranker_call():
    """When reranker=None, pipeline must not attempt any reranking."""
    from app.rag.pipeline import RAGPipeline

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        make_chunk(chunk_id="c1", text="relevant text about retrieval")
    ]
    mock_retriever.default_top_k = 5
    mock_retriever.default_min_similarity = 0.0
    mock_retriever.default_retrieval_mode = "semantic"

    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Generated answer"

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        reranker=None,
    )
    response = pipeline.ask("test question", top_k=5)
    assert response.reranking_enabled is False
    # Retriever called with candidate_k == final_k (no multiplier)
    call_kwargs = mock_retriever.retrieve.call_args
    assert call_kwargs.kwargs.get("top_k", call_kwargs.args[1] if len(call_kwargs.args) > 1 else None) == 5


# ---------------------------------------------------------------------------
# 16. Reranking enabled: pipeline calls reranker with candidate_k chunks
# ---------------------------------------------------------------------------

def test_pipeline_reranking_enabled_calls_reranker():
    from app.rag.pipeline import RAGPipeline

    chunk_a = make_chunk(chunk_id="a", text="retrieval information")
    chunk_b = make_chunk(chunk_id="b", text="authentication JWT")
    chunk_c = make_chunk(chunk_id="c", text="database schema")

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [chunk_a, chunk_b, chunk_c]
    mock_retriever.default_top_k = 1
    mock_retriever.default_min_similarity = 0.0
    mock_retriever.default_retrieval_mode = "semantic"

    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Answer"

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [chunk_a]

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        reranker=mock_reranker,
        reranker_candidate_multiplier=3,
    )

    response = pipeline.ask("question", top_k=1)

    assert response.reranking_enabled is True
    assert response.candidates_retrieved == 3
    # Retriever should have been asked for candidate_k = 1 * 3 = 3
    retrieve_kwargs = mock_retriever.retrieve.call_args.kwargs
    assert retrieve_kwargs["top_k"] == 3
    # Reranker must have been called with final_k = 1
    mock_reranker.rerank.assert_called_once()
    rerank_call = mock_reranker.rerank.call_args
    assert rerank_call.kwargs["top_k"] == 1


# ---------------------------------------------------------------------------
# 17. Reranking enabled: generator receives reranked chunks, not raw candidates
# ---------------------------------------------------------------------------

def test_pipeline_generator_receives_reranked_chunks():
    from app.rag.pipeline import RAGPipeline

    raw = [make_chunk(chunk_id=f"c{i}", text=f"text {i}") for i in range(6)]
    final = [make_chunk(chunk_id="reranked_top", text="best result")]

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = raw
    mock_retriever.default_top_k = 2
    mock_retriever.default_min_similarity = 0.0
    mock_retriever.default_retrieval_mode = "semantic"

    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Answer"

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = final

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        reranker=mock_reranker,
        reranker_candidate_multiplier=3,
    )
    response = pipeline.ask("question", top_k=2)
    # Generator should receive the reranked chunks, not the raw candidates
    gen_call = mock_generator.generate.call_args
    assert gen_call.kwargs["retrieved_chunks"] == final
    assert response.retrieved_chunks == final


# ---------------------------------------------------------------------------
# 18. No Gemini call is made by TFIDFReranker
# ---------------------------------------------------------------------------

def test_tfidf_reranker_makes_no_gemini_call():
    """Ensure TFIDFReranker never touches any Gemini API module."""
    with patch("app.generation.generator.GeminiGenerator") as mock_gemini:
        chunks = [make_chunk(chunk_id="c1", text="some text about retrieval")]
        _ = reranker.rerank("retrieval", chunks, top_k=1)
        mock_gemini.assert_not_called()


# ---------------------------------------------------------------------------
# 19. Document filtering still respected (pipeline passes document_ids through)
# ---------------------------------------------------------------------------

def test_pipeline_document_ids_passed_to_retriever():
    from app.rag.pipeline import RAGPipeline

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [make_chunk()]
    mock_retriever.default_top_k = 5
    mock_retriever.default_min_similarity = 0.0
    mock_retriever.default_retrieval_mode = "semantic"

    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Answer"

    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
    pipeline.ask("question", top_k=5, document_ids=["doc_abc"])

    retrieve_kwargs = mock_retriever.retrieve.call_args.kwargs
    assert retrieve_kwargs["document_ids"] == ["doc_abc"]


# ---------------------------------------------------------------------------
# 20. Hybrid retrieval mode forwarded correctly
# ---------------------------------------------------------------------------

def test_pipeline_retrieval_mode_forwarded():
    from app.rag.pipeline import RAGPipeline

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [make_chunk()]
    mock_retriever.default_top_k = 5
    mock_retriever.default_min_similarity = 0.0
    mock_retriever.default_retrieval_mode = "semantic"

    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Answer"

    pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)
    pipeline.ask("question", top_k=5, retrieval_mode="hybrid")

    retrieve_kwargs = mock_retriever.retrieve.call_args.kwargs
    assert retrieve_kwargs["retrieval_mode"] == "hybrid"


# ---------------------------------------------------------------------------
# 21. _tfidf_score edge cases
# ---------------------------------------------------------------------------

def test_tfidf_score_empty_query():
    assert _tfidf_score([], ["alpha", "beta"]) == 0.0


def test_tfidf_score_empty_doc():
    assert _tfidf_score(["alpha", "beta"], []) == 0.0


def test_tfidf_score_no_overlap():
    score = _tfidf_score(["zebra"], ["alpha", "beta", "gamma"])
    assert score == 0.0


def test_tfidf_score_full_overlap():
    score = _tfidf_score(["alpha", "beta"], ["alpha", "beta", "alpha"])
    assert score > 0.0


def test_tfidf_score_positive():
    score = _tfidf_score(["retrieval"], ["retrieval", "augmented", "generation"])
    assert score > 0.0


# ---------------------------------------------------------------------------
# 22. _tokenize correctness
# ---------------------------------------------------------------------------

def test_tokenize_lowercases():
    assert _tokenize("Hello World") == ["hello", "world"]


def test_tokenize_strips_punctuation():
    tokens = _tokenize("JWT-based auth.")
    assert "jwt" in tokens
    assert "based" in tokens
    assert "auth" in tokens


def test_tokenize_empty():
    assert _tokenize("") == []
