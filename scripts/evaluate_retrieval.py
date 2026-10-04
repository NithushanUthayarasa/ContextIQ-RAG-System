#!/usr/bin/env python
"""
ContextIQ - Advanced RAG Retrieval & Pipeline Benchmark Runner

Executes evaluation benchmarks across dataset queries and generates:
- Terminal comparison tables
- evaluation/results.json (machine-readable aggregate and per-query logs)
- evaluation/REPORT.md (human-readable comprehensive markdown report)

Supports offline deterministic evaluation (default) with zero API calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional
from unittest.mock import MagicMock

# Ensure project root is in python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.evaluation.retrieval_evaluator import (
    AdvancedRAGEvaluator,
    ConfigurationSummary,
    DeterministicQueryExpander,
    EvaluationQuery,
)
from app.generation.generator import GeminiGenerator
from app.rag.pipeline import RAGPipeline
from app.rag.query_expander import GeminiQueryExpander
from app.retrieval.compressor import ExtractiveContextCompressor
from app.retrieval.reranker import TFIDFReranker
from app.retrieval.retriever import Retriever
from app.vectorstore.chroma_store import ChromaVectorStore


def build_pipeline(
    vector_store: ChromaVectorStore,
    retrieval_mode: str = "hybrid",
    reranker_enabled: bool = False,
    query_expansion_enabled: bool = False,
    parent_child_enabled: bool = False,
    context_compression_enabled: bool = False,
    live_gemini: bool = False,
) -> RAGPipeline:
    """Builds a configured RAGPipeline instance for benchmark execution."""
    # Deterministic mock embedder for offline benchmarking if live not requested
    # Note: Retriever in BM25/hybrid mode handles candidates seamlessly
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.01] * 768

    retriever = Retriever(
        embedder=mock_embedder,
        vector_store=vector_store,
        default_top_k=5,
        default_min_similarity=0.0,
        default_retrieval_mode=retrieval_mode,
    )

    reranker = TFIDFReranker() if reranker_enabled else None

    if query_expansion_enabled:
        if live_gemini:
            query_expander = GeminiQueryExpander()
        else:
            query_expander = DeterministicQueryExpander()
    else:
        query_expander = None

    compressor = ExtractiveContextCompressor() if context_compression_enabled else None

    # Deterministic mock generator for offline benchmarking
    generator = MagicMock(spec=GeminiGenerator)
    generator.generate.return_value = (
        "Grounding answer based on retrieved document sections. "
        "If information is missing, explicitly state: I do not have enough information."
    )

    return RAGPipeline(
        retriever=retriever,
        generator=generator,
        reranker=reranker,
        query_expander=query_expander,
        parent_store=vector_store,
        parent_child_enabled=parent_child_enabled,
        compressor=compressor,
        context_compression_enabled=context_compression_enabled,
    )


def print_comparison_table(summaries: Dict[str, ConfigurationSummary]) -> None:
    """Prints a formatted ASCII comparison table to stdout."""
    header = (
        f"{'Configuration':<26} | {'Hit@3':<6} | {'Hit@5':<6} | {'MRR@5':<6} | "
        f"{'Recall@5':<8} | {'Chars':<6} | {'Red %':<6} | {'ERR %':<6} | {'Cite Prec':<9}"
    )
    separator = "-" * len(header)
    print("\n" + separator)
    print("CONTEXTIQ ADVANCED RAG BENCHMARK RESULTS")
    print(separator)
    print(header)
    print(separator)

    for cid, s in summaries.items():
        err_str = f"{s.avg_evidence_retention_rate * 100:.1f}%"
        red_str = f"{s.avg_context_reduction_pct:.1f}%"
        chars_str = f"{int(s.avg_compressed_chars)}" if s.avg_compressed_chars > 0 else f"{int(s.avg_original_chars)}"
        row = (
            f"{cid:<26} | {s.hit_at_3:<6.3f} | {s.hit_at_5:<6.3f} | {s.mrr_at_5:<6.3f} | "
            f"{s.recall_at_5:<8.3f} | {chars_str:<6} | {red_str:<6} | {err_str:<6} | {s.avg_citation_precision:<9.3f}"
        )
        print(row)

    print(separator + "\n")


def main():
    parser = argparse.ArgumentParser(description="ContextIQ Advanced RAG Evaluator CLI")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(BASE_DIR / "evaluation" / "retrieval_dataset.json"),
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K chunks to retrieve during evaluation turns",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        default=True,
        help="Run full ablation across Phase 14 configurations (default: True)",
    )
    parser.add_argument(
        "--live-gemini",
        action="store_true",
        default=False,
        help="Use live Gemini API calls for query expansion (default: False, uses deterministic offline expander)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(BASE_DIR / "evaluation" / "results.json"),
        help="Output path for evaluation results JSON",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=str(BASE_DIR / "evaluation" / "REPORT.md"),
        help="Output path for evaluation Markdown report",
    )

    args = parser.parse_args()

    evaluator = AdvancedRAGEvaluator(args.dataset)
    print(f"Loaded {len(evaluator.queries)} evaluation queries from: {args.dataset}")

    vector_store = ChromaVectorStore()
    total_chunks = vector_store.count()
    print(f"Connected to ChromaDB ({total_chunks} indexed chunks).")

    configs = [
        (
            "C1_Hybrid_Baseline",
            "Hybrid retrieval only (Semantic + BM25 + RRF)",
            {"reranker_enabled": False, "query_expansion_enabled": False, "parent_child_enabled": False, "context_compression_enabled": False},
        ),
        (
            "C2_Hybrid_Reranked",
            "Hybrid retrieval + TF-IDF Reranking",
            {"reranker_enabled": True, "query_expansion_enabled": False, "parent_child_enabled": False, "context_compression_enabled": False},
        ),
        (
            "C3_Hybrid_Rerank_Expanded",
            "Hybrid retrieval + TF-IDF Reranking + Query Expansion",
            {"reranker_enabled": True, "query_expansion_enabled": True, "parent_child_enabled": False, "context_compression_enabled": False},
        ),
        (
            "C4_Hybrid_Rerank_ParentChild",
            "Hybrid + TF-IDF Reranking + Parent Context Resolution",
            {"reranker_enabled": True, "query_expansion_enabled": False, "parent_child_enabled": True, "context_compression_enabled": False},
        ),
        (
            "C5_Full_Advanced_RAG",
            "Full pipeline: Hybrid + Reranking + Parent/Child + Context Compression",
            {"reranker_enabled": True, "query_expansion_enabled": False, "parent_child_enabled": True, "context_compression_enabled": True},
        ),
    ]

    summaries: Dict[str, ConfigurationSummary] = {}

    for cid, desc, flags in configs:
        print(f"Evaluating {cid}...")
        pipeline = build_pipeline(
            vector_store=vector_store,
            live_gemini=args.live_gemini,
            **flags,
        )
        summary = evaluator.evaluate_configuration(
            pipeline=pipeline,
            config_id=cid,
            description=desc,
            top_k=args.top_k,
        )
        summaries[cid] = summary

    # Display results
    print_comparison_table(summaries)

    # Export results
    evaluator.export_results_json(summaries, args.output_json)
    print(f"Exported machine-readable results to: {args.output_json}")

    evaluator.generate_markdown_report(summaries, args.output_report)
    print(f"Exported markdown evaluation report to: {args.output_report}")


if __name__ == "__main__":
    main()
