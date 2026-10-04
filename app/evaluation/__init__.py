"""
ContextIQ - Evaluation Module
Exposes core evaluation metrics, dataset models, and pipeline evaluator engine.
"""

from app.evaluation.retrieval_evaluator import (
    AdvancedRAGEvaluator,
    ConfigurationSummary,
    DeterministicQueryExpander,
    EvaluationQuery,
    QueryEvaluationResult,
    calculate_citation_metrics,
    calculate_context_reduction_pct,
    calculate_evidence_retention_rate,
    calculate_hit_at_k,
    calculate_mrr_at_k,
    calculate_recall_at_k,
    calculate_refusal_accuracy,
    is_chunk_relevant,
)

__all__ = [
    "AdvancedRAGEvaluator",
    "ConfigurationSummary",
    "DeterministicQueryExpander",
    "EvaluationQuery",
    "QueryEvaluationResult",
    "calculate_citation_metrics",
    "calculate_context_reduction_pct",
    "calculate_evidence_retention_rate",
    "calculate_hit_at_k",
    "calculate_mrr_at_k",
    "calculate_recall_at_k",
    "calculate_refusal_accuracy",
    "is_chunk_relevant",
]
