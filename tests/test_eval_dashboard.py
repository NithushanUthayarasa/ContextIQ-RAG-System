"""
Unit tests for ContextIQ Evaluation Dashboard module (app/ui/eval_dashboard.py).
Tests safe loading, summary metric calculations, chart DataFrame builders,
error handling for missing/corrupt data, and Streamlit rendering.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd

from app.ui.eval_dashboard import (
    DEFAULT_REPORT_PATH,
    DEFAULT_RESULTS_PATH,
    build_comparison_dataframe,
    build_context_chars_chart_df,
    build_efficiency_chart_df,
    build_query_breakdown_df,
    build_retrieval_chart_df,
    get_summary_metrics,
    load_evaluation_report,
    load_evaluation_results,
    render_evaluation_dashboard,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_results_data():
    """Returns sample benchmark results dictionary mirroring real evaluation results."""
    return {
        "timestamp": "2026-10-04T19:46:55.947246+00:00",
        "configurations_count": 5,
        "configurations": {
            "C1_Hybrid_Baseline": {
                "configuration_id": "C1_Hybrid_Baseline",
                "description": "Hybrid retrieval only (Semantic + BM25 + RRF)",
                "queries_evaluated": 15,
                "metrics": {
                    "hit_at_3": 0.9167,
                    "hit_at_5": 1.0,
                    "recall_at_3": 0.7917,
                    "recall_at_5": 0.9583,
                    "mrr_at_3": 0.9167,
                    "mrr_at_5": 0.9375,
                    "avg_original_chars": 3650.1,
                    "avg_compressed_chars": 3650.1,
                    "avg_context_reduction_pct": 0.0,
                    "avg_evidence_retention_rate": 1.0,
                    "avg_citation_precision": 0.2875,
                    "avg_citation_recall": 0.9583,
                    "refusal_accuracy": 1.0,
                },
                "per_query_results": [
                    {
                        "query_id": "q01",
                        "query_type": "conceptual",
                        "question": "What is authentication?",
                        "hit_at_5": 1.0,
                        "mrr_at_5": 1.0,
                        "context_reduction_pct": 0.0,
                        "evidence_retention_rate": 1.0,
                        "retrieved_sources": [{"source": "lecture.pdf", "page": 40}],
                    }
                ],
            },
            "C5_Full_Advanced_RAG": {
                "configuration_id": "C5_Full_Advanced_RAG",
                "description": "Full Advanced RAG (Hybrid + Rerank + Parent/Child + Compression)",
                "queries_evaluated": 15,
                "metrics": {
                    "hit_at_3": 0.75,
                    "hit_at_5": 0.9167,
                    "recall_at_3": 0.625,
                    "recall_at_5": 0.7917,
                    "mrr_at_3": 0.7083,
                    "mrr_at_5": 0.7083,
                    "avg_original_chars": 3298.0,
                    "avg_compressed_chars": 973.0,
                    "avg_context_reduction_pct": 70.32,
                    "avg_evidence_retention_rate": 0.6111,
                    "avg_citation_precision": 0.2125,
                    "avg_citation_recall": 0.7917,
                    "refusal_accuracy": 1.0,
                },
                "per_query_results": [
                    {
                        "query_id": "q01",
                        "query_type": "conceptual",
                        "question": "What is authentication?",
                        "hit_at_5": 1.0,
                        "mrr_at_5": 1.0,
                        "context_reduction_pct": 68.5,
                        "evidence_retention_rate": 0.65,
                        "retrieved_sources": [{"source": "lecture.pdf", "page": 40}],
                    }
                ],
            },
        },
    }


# ---------------------------------------------------------------------------
# Test 1 — Loading Real Evaluation Results
# ---------------------------------------------------------------------------
def test_load_real_evaluation_results():
    """Verify loading the real existing evaluation/results.json file."""
    assert DEFAULT_RESULTS_PATH.is_file(), "Default results.json should exist"
    data = load_evaluation_results()
    assert data is not None
    assert "configurations" in data
    configs = data["configurations"]
    assert len(configs) == 5
    assert "C1_Hybrid_Baseline" in configs
    assert "C5_Full_Advanced_RAG" in configs


# ---------------------------------------------------------------------------
# Test 2 — Loading Real Evaluation Report
# ---------------------------------------------------------------------------
def test_load_real_evaluation_report():
    """Verify loading the real existing evaluation/REPORT.md file."""
    assert DEFAULT_REPORT_PATH.is_file(), "Default REPORT.md should exist"
    report = load_evaluation_report()
    assert report is not None
    assert "# ContextIQ Advanced RAG Evaluation Report" in report
    assert "C1_Hybrid_Baseline" in report
    assert "C5_Full_Advanced_RAG" in report


# ---------------------------------------------------------------------------
# Test 3 — Graceful Handling of Missing or Malformed Files
# ---------------------------------------------------------------------------
def test_load_results_missing_file(tmp_path: Path):
    """Verify load_evaluation_results returns None for non-existent file."""
    non_existent = tmp_path / "does_not_exist.json"
    assert load_evaluation_results(non_existent) is None


def test_load_results_malformed_json(tmp_path: Path):
    """Verify load_evaluation_results returns None for malformed JSON."""
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{this is not valid json", encoding="utf-8")
    assert load_evaluation_results(corrupt_file) is None


def test_load_results_missing_configurations_key(tmp_path: Path):
    """Verify load_evaluation_results returns None if 'configurations' key is missing."""
    invalid_file = tmp_path / "no_configs.json"
    invalid_file.write_text(json.dumps({"timestamp": "2026-10-04"}), encoding="utf-8")
    assert load_evaluation_results(invalid_file) is None


def test_load_report_missing_file(tmp_path: Path):
    """Verify load_evaluation_report returns None for non-existent report."""
    non_existent = tmp_path / "non_existent.md"
    assert load_evaluation_report(non_existent) is None


# ---------------------------------------------------------------------------
# Test 4 — Summary Metrics Extraction
# ---------------------------------------------------------------------------
def test_get_summary_metrics(sample_results_data):
    """Verify KPI extraction from evaluation data."""
    summary = get_summary_metrics(sample_results_data)
    assert summary["total_configs"] == 2
    assert summary["queries_count"] == 15
    assert summary["best_hit5"] == 1.0
    assert summary["best_mrr5"] == 0.9375
    assert round(summary["c5_reduction_pct"], 1) == 70.3
    assert round(summary["c5_retention_rate"], 3) == 0.611
    assert summary["refusal_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Test 5 — Comparison DataFrame Builder
# ---------------------------------------------------------------------------
def test_build_comparison_dataframe(sample_results_data):
    """Verify ablation comparison DataFrame generation and formatting."""
    df = build_comparison_dataframe(sample_results_data)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "Configuration" in df.columns
    assert "Hit@5" in df.columns
    assert "MRR@5" in df.columns
    assert "Reduction %" in df.columns
    assert "Evidence Retention" in df.columns

    c1_row = df[df["Config Key"] == "C1_Hybrid_Baseline"].iloc[0]
    assert c1_row["Hit@5"] == 1.0
    assert c1_row["Reduction %"] == "0.0%"
    assert c1_row["Evidence Retention"] == "100.0%"

    c5_row = df[df["Config Key"] == "C5_Full_Advanced_RAG"].iloc[0]
    assert c5_row["Reduction %"] == "70.3%"


# ---------------------------------------------------------------------------
# Test 6 — Chart DataFrames Builders
# ---------------------------------------------------------------------------
def test_chart_dataframe_builders(sample_results_data):
    """Verify structure and indices of chart DataFrames."""
    df_retrieval = build_retrieval_chart_df(sample_results_data)
    assert isinstance(df_retrieval, pd.DataFrame)
    assert "Hit@5" in df_retrieval.columns
    assert "MRR@5" in df_retrieval.columns
    assert "C1" in df_retrieval.index

    df_eff = build_efficiency_chart_df(sample_results_data)
    assert isinstance(df_eff, pd.DataFrame)
    assert "Context Reduction (%)" in df_eff.columns
    assert "Evidence Retention (%)" in df_eff.columns
    assert "C5" in df_eff.index

    df_chars = build_context_chars_chart_df(sample_results_data)
    assert isinstance(df_chars, pd.DataFrame)
    assert "Original Chars" in df_chars.columns
    assert "Compressed Chars" in df_chars.columns


# ---------------------------------------------------------------------------
# Test 7 — Query Breakdown DataFrame Builder
# ---------------------------------------------------------------------------
def test_build_query_breakdown_df():
    """Verify per-query breakdown table generation."""
    queries = [
        {
            "query_id": "q01",
            "query_type": "conceptual",
            "question": "What is authentication?",
            "hit_at_5": 1.0,
            "mrr_at_5": 1.0,
            "context_reduction_pct": 70.0,
            "evidence_retention_rate": 0.65,
            "retrieved_sources": [{"source": "lecture.pdf", "page": 40}],
        }
    ]
    df = build_query_breakdown_df(queries)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["Query ID"] == "q01"
    assert df.iloc[0]["Top Pages"] == "p.40"


# ---------------------------------------------------------------------------
# Test 8 — Streamlit Rendering Safe Execution
# ---------------------------------------------------------------------------
@patch("app.ui.eval_dashboard.st")
def test_render_evaluation_dashboard_with_valid_data(mock_st, sample_results_data, tmp_path: Path):
    """Verify render_evaluation_dashboard executes without errors on valid data."""
    test_json = tmp_path / "results.json"
    test_json.write_text(json.dumps(sample_results_data), encoding="utf-8")

    # Set up mock tabs & columns to support context manager syntax
    mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
    mock_st.tabs.side_effect = lambda tabs: [MagicMock() for _ in tabs]
    mock_st.expander.return_value.__enter__.return_value = MagicMock()

    render_evaluation_dashboard(results_path=test_json)

    # Should have rendered metrics, tables, and charts
    mock_st.metric.assert_called()
    mock_st.dataframe.assert_called()
    mock_st.bar_chart.assert_called()


@patch("app.ui.eval_dashboard.st")
def test_render_evaluation_dashboard_with_missing_file(mock_st, tmp_path: Path):
    """Verify render_evaluation_dashboard displays warning when results file is missing."""
    missing_json = tmp_path / "non_existent.json"
    render_evaluation_dashboard(results_path=missing_json)

    # Warning should be displayed
    mock_st.warning.assert_called_once()
    assert "Evaluation Results Not Found" in mock_st.warning.call_args[0][0]
