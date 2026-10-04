"""
ContextIQ - Advanced RAG Retrieval Evaluator Test Suite
Deterministic unit tests covering IR metrics, context reduction, evidence retention,
citation precision/recall, refusal evaluation, dataset loading, edge cases, and report export.
All tests use deterministic mocks; zero real Gemini API calls.
"""

import json
from pathlib import Path
import tempfile
from typing import List, Optional
from unittest.mock import MagicMock
import pytest

from app.evaluation.retrieval_evaluator import (
    AdvancedRAGEvaluator,
    ConfigurationSummary,
    DeterministicQueryExpander,
    EvaluationQuery,
    calculate_citation_metrics,
    calculate_context_reduction_pct,
    calculate_evidence_retention_rate,
    calculate_hit_at_k,
    calculate_mrr_at_k,
    calculate_recall_at_k,
    calculate_refusal_accuracy,
    is_chunk_relevant,
    normalize_text,
)
from app.rag.pipeline import RAGPipeline, RAGResponse
from app.retrieval.models import RetrievedChunk


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def make_chunk(
    chunk_id: str,
    text: str,
    source: str = "doc.pdf",
    page: int = 1,
    index: int = 0,
    distance: float = 0.20,
    original_text: Optional[str] = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page_number=page,
        chunk_index=index,
        distance=distance,
        document_id="doc-sha256",
        original_text=original_text or text,
    )


def make_query(
    qid: str = "q1",
    question: str = "Test question?",
    source: str = "doc.pdf",
    pages: Optional[List[int]] = None,
    evidence: Optional[List[str]] = None,
    unanswerable: bool = False,
) -> EvaluationQuery:
    return EvaluationQuery(
        id=qid,
        query_type="conceptual" if not unanswerable else "unanswerable",
        question=question,
        expected_source=source,
        expected_pages=pages or [1],
        expected_evidence=evidence or ["key evidence"],
        unanswerable=unanswerable,
    )


# ---------------------------------------------------------------------------
# 1. Deterministic Relevancy Matching Tests
# ---------------------------------------------------------------------------

class TestRelevancyMatching:
    def test_chunk_relevant_when_source_page_and_evidence_match(self):
        query = make_query(source="doc.pdf", pages=[1], evidence=["JWT signature"])
        chunk = make_chunk("c1", "This section explains the JWT signature verification.", source="doc.pdf", page=1)
        assert is_chunk_relevant(chunk, query) is True

    def test_chunk_not_relevant_when_page_mismatches(self):
        query = make_query(source="doc.pdf", pages=[1], evidence=["JWT signature"])
        chunk = make_chunk("c1", "JWT signature explanation.", source="doc.pdf", page=5)
        assert is_chunk_relevant(chunk, query) is False

    def test_chunk_not_relevant_when_source_mismatches(self):
        query = make_query(source="doc.pdf", pages=[1], evidence=["JWT signature"])
        chunk = make_chunk("c1", "JWT signature explanation.", source="other.pdf", page=1)
        assert is_chunk_relevant(chunk, query) is False

    def test_chunk_not_relevant_when_evidence_missing(self):
        query = make_query(source="doc.pdf", pages=[1], evidence=["JWT signature"])
        chunk = make_chunk("c1", "General introduction to web frameworks.", source="doc.pdf", page=1)
        assert is_chunk_relevant(chunk, query) is False

    def test_chunk_never_relevant_for_unanswerable_query(self):
        query = make_query(unanswerable=True)
        chunk = make_chunk("c1", "Some arbitrary text content.", source="doc.pdf", page=1)
        assert is_chunk_relevant(chunk, query) is False


# ---------------------------------------------------------------------------
# 2. Layer A: Retrieval Quality Metrics Tests
# ---------------------------------------------------------------------------

class TestRetrievalMetrics:
    def test_hit_at_k(self):
        query = make_query(pages=[2], evidence=["evidence phrase"])
        c_rel = make_chunk("c1", "Contains evidence phrase.", page=2)
        c_irrel = make_chunk("c2", "Irrelevant content.", page=1)

        # Relevant at rank 1 -> Hit@1, Hit@3, Hit@5 = 1.0
        assert calculate_hit_at_k([c_rel, c_irrel], query, 1) == 1.0
        assert calculate_hit_at_k([c_rel, c_irrel], query, 3) == 1.0

        # Relevant at rank 3
        chunks = [c_irrel, c_irrel, c_rel]
        assert calculate_hit_at_k(chunks, query, 2) == 0.0
        assert calculate_hit_at_k(chunks, query, 3) == 1.0

        # Empty list
        assert calculate_hit_at_k([], query, 5) == 0.0

    def test_recall_at_k(self):
        query = make_query(pages=[1, 2, 3], evidence=["evidence phrase"])
        c1 = make_chunk("c1", "Contains evidence phrase on page 1.", page=1)
        c2 = make_chunk("c2", "Contains evidence phrase on page 2.", page=2)
        c_other = make_chunk("c3", "Other page.", page=4)

        # Retrieves 2 out of 3 expected pages in top-3
        recall = calculate_recall_at_k([c1, c2, c_other], query, 3)
        assert round(recall, 4) == round(2 / 3, 4)

        # Retrieves 1 out of 3 in top-1
        assert round(calculate_recall_at_k([c1, c2, c_other], query, 1), 4) == round(1 / 3, 4)

    def test_mrr_at_k(self):
        query = make_query(pages=[1], evidence=["relevant info"])
        c_rel = make_chunk("c1", "Has relevant info.", page=1)
        c_irrel = make_chunk("c2", "Other text.", page=2)

        # First relevant at rank 1
        assert calculate_mrr_at_k([c_rel, c_irrel], query, 5) == 1.0

        # First relevant at rank 2
        assert calculate_mrr_at_k([c_irrel, c_rel], query, 5) == 0.5

        # First relevant at rank 4 (within top-5)
        assert calculate_mrr_at_k([c_irrel, c_irrel, c_irrel, c_rel], query, 5) == 0.25

        # No relevant chunk in top-3
        assert calculate_mrr_at_k([c_irrel, c_irrel, c_irrel, c_rel], query, 3) == 0.0


# ---------------------------------------------------------------------------
# 3. Layer B: Context Efficiency & Evidence Preservation Tests
# ---------------------------------------------------------------------------

class TestContextMetrics:
    def test_context_reduction_pct(self):
        # 1000 original down to 250 compressed -> 75% reduction
        assert calculate_context_reduction_pct(1000, 250) == 75.0
        # No reduction
        assert calculate_context_reduction_pct(1000, 1000) == 0.0
        # Zero original chars guard
        assert calculate_context_reduction_pct(0, 0) == 0.0

    def test_evidence_retention_rate_full(self):
        uncomp = "ContextIQ uses ChromaDB. It stores document embeddings."
        comp = "ContextIQ uses ChromaDB. It stores document embeddings."
        evidence = ["ChromaDB", "document embeddings"]
        assert calculate_evidence_retention_rate(uncomp, comp, evidence) == 1.0

    def test_evidence_retention_rate_partial(self):
        uncomp = "ContextIQ uses ChromaDB. It stores document embeddings and provides cosine search."
        comp = "ContextIQ uses ChromaDB. It performs fast retrieval."
        evidence = ["ChromaDB", "document embeddings"]
        # 'ChromaDB' kept, 'document embeddings' dropped
        assert calculate_evidence_retention_rate(uncomp, comp, evidence) == 0.5

    def test_evidence_retention_rate_empty_or_missing(self):
        uncomp = "Unrelated background information."
        comp = "Unrelated background."
        evidence = ["Missing evidence phrase"]
        # Evidence was never in uncompressed text, so compression is not penalized
        assert calculate_evidence_retention_rate(uncomp, comp, evidence) == 1.0


# ---------------------------------------------------------------------------
# 4. Layer C: Citation Accuracy & Refusal Tests
# ---------------------------------------------------------------------------

class TestCitationAndRefusalMetrics:
    def test_citation_precision_and_recall_perfect(self):
        query = make_query(source="lecture.pdf", pages=[4, 5])
        cited = [{"source": "lecture.pdf", "page": 4}, {"source": "lecture.pdf", "page": 5}]
        prec, rec = calculate_citation_metrics(cited, query)
        assert prec == 1.0
        assert rec == 1.0

    def test_citation_precision_and_recall_partial(self):
        query = make_query(source="lecture.pdf", pages=[4, 5])
        # Cites page 4 (correct) and page 10 (incorrect)
        cited = [{"source": "lecture.pdf", "page": 4}, {"source": "lecture.pdf", "page": 10}]
        prec, rec = calculate_citation_metrics(cited, query)
        assert prec == 0.5
        assert rec == 0.5

    def test_citation_empty(self):
        query = make_query(source="lecture.pdf", pages=[4])
        prec, rec = calculate_citation_metrics([], query)
        assert prec == 0.0
        assert rec == 0.0

    def test_refusal_accuracy_detected(self):
        answer = "I do not have enough information from the provided documents to answer this question."
        assert calculate_refusal_accuracy(answer) == 1.0

    def test_refusal_accuracy_hallucination(self):
        answer = "The redis clustering configuration uses 3 master nodes and 3 slave nodes."
        assert calculate_refusal_accuracy(answer) == 0.0


# ---------------------------------------------------------------------------
# 5. Dataset Loading, Edge Cases & Export Tests
# ---------------------------------------------------------------------------

class TestDatasetAndEvaluatorOrchestration:
    def test_load_dataset_valid(self):
        sample_data = [
            {
                "id": "q1",
                "query_type": "conceptual",
                "question": "What is JWT?",
                "expected_source": "slides.pdf",
                "expected_pages": [43],
                "expected_evidence": ["header", "payload", "signature"],
                "unanswerable": False,
            }
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(sample_data, f)
            temp_path = f.name

        evaluator = AdvancedRAGEvaluator()
        queries = evaluator.load_dataset(temp_path)
        assert len(queries) == 1
        assert queries[0].id == "q1"
        assert queries[0].expected_pages == [43]
        Path(temp_path).unlink()

    def test_load_dataset_file_not_found(self):
        evaluator = AdvancedRAGEvaluator()
        with pytest.raises(FileNotFoundError):
            evaluator.load_dataset("non_existent_file.json")

    def test_evaluate_configuration_summary_aggregation(self):
        evaluator = AdvancedRAGEvaluator()
        query = make_query(pages=[1], evidence=["evidence phrase"])
        chunk = make_chunk("c1", "Contains evidence phrase.", page=1)

        mock_pipeline = MagicMock(spec=RAGPipeline)
        mock_pipeline.ask.return_value = RAGResponse(
            answer="Grounded answer.",
            sources=[{"source": "doc.pdf", "page": 1}],
            retrieved_chunks=[chunk],
            query=query.question,
            total_chars_original=200,
            total_chars_compressed=100,
        )

        summary = evaluator.evaluate_configuration(
            pipeline=mock_pipeline,
            config_id="test_cfg",
            description="Test Configuration",
            dataset=[query],
        )

        assert summary.configuration_id == "test_cfg"
        assert summary.queries_evaluated == 1
        assert summary.hit_at_3 == 1.0
        assert summary.mrr_at_5 == 1.0
        assert summary.avg_context_reduction_pct == 50.0
        assert summary.avg_evidence_retention_rate == 1.0
        assert summary.avg_citation_precision == 1.0

    def test_deterministic_query_expander(self):
        expander = DeterministicQueryExpander()
        variations = expander.expand("What is database authentication and password hashing?", max_queries=3)
        assert len(variations) <= 3
        assert variations[0] == "What is database authentication and password hashing?"

    def test_export_results_json_and_markdown_report(self):
        summary = ConfigurationSummary(
            configuration_id="test_cfg",
            description="Test Configuration",
            queries_evaluated=1,
            hit_at_3=1.0,
            hit_at_5=1.0,
            recall_at_3=1.0,
            recall_at_5=1.0,
            mrr_at_3=1.0,
            mrr_at_5=1.0,
            avg_original_chars=500.0,
            avg_compressed_chars=200.0,
            avg_context_reduction_pct=60.0,
            avg_evidence_retention_rate=1.0,
            avg_citation_precision=1.0,
            avg_citation_recall=1.0,
            refusal_accuracy=1.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "results.json"
            md_path = Path(tmp_dir) / "REPORT.md"

            AdvancedRAGEvaluator.export_results_json({"test_cfg": summary}, json_path)
            assert json_path.exists()
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "configurations" in data
            assert "test_cfg" in data["configurations"]

            report = AdvancedRAGEvaluator.generate_markdown_report({"test_cfg": summary}, md_path)
            assert md_path.exists()
            assert "# ContextIQ Advanced RAG Evaluation Report" in report
            assert "test_cfg" in report
