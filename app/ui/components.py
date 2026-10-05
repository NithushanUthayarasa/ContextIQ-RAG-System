"""
ContextIQ - UI Components Module
Reusable visual components for Streamlit interface: headers, sidebar, cards, sources, and context viewers.
"""

from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

from app.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    GENERATION_MODEL_NAME,
    is_api_key_configured,
)
from app.retrieval.retriever import RetrievedChunk
from app.ui.styles import CUSTOM_CSS


def apply_custom_styles():
    """Injects custom CSS styles into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header():
    """Renders the top branding banner."""
    st.markdown(
        """
        <div class="contextiq-header">
            <div class="contextiq-title-row">
                <h1 class="contextiq-brand">ContextIQ</h1>
                <span class="contextiq-tagline">RAG-Powered Multi-Document Intelligence System</span>
            </div>
            <p class="contextiq-subtitle">
                Upload documents, retrieve grounded evidence, and ask questions with source-aware answers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(vector_store) -> Dict[str, Any]:
    """
    Renders the sidebar containing system status, document collection,
    retrieval settings, and advanced RAG feature toggles.
    """
    with st.sidebar:
        # ==========================================
        # 1. System Status
        # ==========================================
        st.markdown("### ⚙️ System Status")

        api_ready = is_api_key_configured()
        if api_ready:
            st.markdown(
                '<span class="status-badge online">● Gemini API: Connected</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-badge offline">● Gemini API: Missing Key</span>',
                unsafe_allow_html=True,
            )
            st.caption("Add your `GEMINI_API_KEY` to `.env` or Streamlit secrets.")

        st.markdown(
            '<span class="status-badge online" style="margin-top: 6px;">● Vector Store: ChromaDB</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"**Models:** `{EMBEDDING_MODEL_NAME}` · `{GENERATION_MODEL_NAME}`")

        st.divider()

        # ==========================================
        # 2. Documents & Collection
        # ==========================================
        st.markdown("### 📚 Documents")
        total_chunks = vector_store.count() if vector_store else 0
        indexed_docs = vector_store.list_indexed_documents() if vector_store else []

        if not indexed_docs:
            st.caption("No documents indexed yet.")
        else:
            st.caption(f"**{len(indexed_docs)}** document(s) • **{total_chunks}** total chunks")
            deleting_id = st.session_state.get("deleting_doc_id")

            for doc in indexed_docs:
                doc_id = doc["document_id"]
                source_name = doc["source"]
                pages = doc["page_count"]
                chunks = doc["chunk_count"]
                is_legacy = doc_id.startswith("legacy_")

                st.markdown(f"📄 **{source_name}**")
                st.caption(f"{pages} page(s) • {chunks} chunk(s)")

                if is_legacy:
                    st.caption("*(Legacy V1 record)*")
                else:
                    if deleting_id == doc_id:
                        st.warning(f"Delete `{source_name}`?")
                        col_confirm, col_cancel = st.columns(2)
                        with col_confirm:
                            if st.button(
                                "Confirm",
                                key=f"confirm_{doc_id}",
                                type="primary",
                                use_container_width=True,
                            ):
                                vector_store.delete_by_document_id(doc_id)
                                st.session_state["deleting_doc_id"] = None
                                st.session_state["last_response"] = None
                                st.success(f"Removed `{source_name}`.")
                                st.rerun()
                        with col_cancel:
                            if st.button(
                                "Cancel",
                                key=f"cancel_{doc_id}",
                                use_container_width=True,
                            ):
                                st.session_state["deleting_doc_id"] = None
                                st.rerun()
                    else:
                        if st.button(
                            "🗑️ Delete",
                            key=f"del_{doc_id}",
                            use_container_width=True,
                        ):
                            st.session_state["deleting_doc_id"] = doc_id
                            st.rerun()

                st.markdown(
                    "<hr style='margin: 8px 0; border: none; border-top: 1px dashed #334155;' />",
                    unsafe_allow_html=True,
                )

        if total_chunks > 0:
            if st.button("🗑️ Clear Vector Database", key="clear_all_db_btn", use_container_width=True):
                vector_store.reset()
                st.session_state["current_document"] = None
                st.session_state["current_pages"] = 0
                st.session_state["deleting_doc_id"] = None
                st.session_state["last_response"] = None
                st.success("Vector database cleared.")
                st.rerun()

        # Conversation history controls
        conversation = st.session_state.get("conversation")
        conv_len = len(conversation.get_messages()) if conversation else 0
        if conv_len > 0:
            st.divider()
            st.markdown("### 💬 Conversation")
            st.caption(f"**{conv_len}** message(s) in session")
            if st.button("🗑️ Clear Conversation", key="clear_conv_btn", use_container_width=True):
                if conversation:
                    conversation.clear()
                st.session_state["last_response"] = None
                st.success("Conversation cleared.")
                st.rerun()

        st.divider()

        # ==========================================
        # 3. Retrieval Settings
        # ==========================================
        st.markdown("### 🔍 Retrieval Settings")

        # Retrieval Mode Selector
        valid_modes = ["semantic", "hybrid", "bm25"]
        current_mode = st.session_state.get("retrieval_mode", DEFAULT_RETRIEVAL_MODE)
        mode_idx = valid_modes.index(current_mode) if current_mode in valid_modes else 0

        retrieval_mode = st.selectbox(
            "Retrieval Strategy",
            options=valid_modes,
            index=mode_idx,
            format_func=lambda m: {
                "semantic": "🧠 Semantic Search (Vector)",
                "hybrid": "⚡ Hybrid Search (Vector + BM25)",
                "bm25": "🔤 Keyword Search (BM25)",
            }.get(m, m),
            help="Choose retrieval strategy: Semantic (dense embeddings), BM25 (keyword matching), or Hybrid (Reciprocal Rank Fusion).",
        )
        st.session_state["retrieval_mode"] = retrieval_mode

        top_k = st.slider(
            "Top-K Chunks to Retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.get("top_k", DEFAULT_TOP_K),
        )
        st.session_state["top_k"] = top_k

        min_similarity = st.slider(
            "Minimum Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("min_similarity", DEFAULT_MIN_SIMILARITY)),
            step=0.05,
            help="Minimum cosine similarity required to include chunks in context (0.0 = keep all, 1.0 = exact match).",
        )
        st.session_state["min_similarity"] = min_similarity

        # Document Filter Multiselect
        selected_doc_ids: Optional[List[str]] = None
        if indexed_docs:
            source_counts: Dict[str, int] = {}
            for d in indexed_docs:
                src = d.get("source", "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1

            doc_options: Dict[str, str] = {}
            for d in indexed_docs:
                did = d["document_id"]
                src = d.get("source", "unknown")
                if source_counts[src] > 1 and not did.startswith("legacy_"):
                    label = f"📄 {src} ({did[:8]}...)"
                else:
                    label = f"📄 {src}"
                if label in doc_options:
                    label = f"📄 {src} ({did[:8]})"
                doc_options[label] = did

            existing_selected = [
                lbl for lbl in st.session_state.get("selected_doc_labels", [])
                if lbl in doc_options
            ]

            selected_labels = st.multiselect(
                "Search Scope (Document Filter)",
                options=list(doc_options.keys()),
                default=existing_selected,
                help="Restrict retrieval to one or more documents. Leave empty to search All Documents.",
            )
            st.session_state["selected_doc_labels"] = selected_labels
            if selected_labels:
                selected_doc_ids = [doc_options[lbl] for lbl in selected_labels]
            else:
                selected_doc_ids = None

        st.divider()

        # ==========================================
        # 4. Advanced RAG
        # ==========================================
        st.markdown("### ⚡ Advanced RAG")

        # Query Expansion Toggle
        query_expansion_enabled = st.toggle(
            "Query Expansion",
            value=st.session_state.get("query_expansion_enabled", False),
            help=(
                "Generates alternative search queries via Gemini to broaden candidate pool coverage."
            ),
        )
        st.session_state["query_expansion_enabled"] = query_expansion_enabled

        # Reranker Toggle
        reranker_enabled = st.toggle(
            "TF-IDF Reranking",
            value=st.session_state.get("reranker_enabled", False),
            help=(
                "Fetches an expanded candidate pool and reranks by lexical term overlap."
            ),
        )
        st.session_state["reranker_enabled"] = reranker_enabled

        # Parent/Child Retrieval Toggle
        parent_child_enabled = st.toggle(
            "Parent/Child Retrieval",
            value=st.session_state.get("parent_child_enabled", False),
            help=(
                "Matches against small child chunks and expands to larger parent sections for generation."
            ),
        )
        st.session_state["parent_child_enabled"] = parent_child_enabled

        # Context Compression Toggle
        context_compression_enabled = st.toggle(
            "Context Compression",
            value=st.session_state.get("context_compression_enabled", False),
            help=(
                "Extracts query-relevant sentences from retrieved context, reducing prompt size."
            ),
        )
        st.session_state["context_compression_enabled"] = context_compression_enabled

        # Chunking parameters tucked into clean expander
        with st.expander("⚙️ Advanced Chunking", expanded=False):
            chunk_size = st.number_input(
                "Chunk Size (characters)",
                min_value=200,
                max_value=3000,
                value=st.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE),
                step=100,
            )
            st.session_state["chunk_size"] = chunk_size

            chunk_overlap = st.number_input(
                "Chunk Overlap (characters)",
                min_value=0,
                max_value=max(100, chunk_size - 100),
                value=min(st.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP), chunk_size - 50),
                step=50,
            )
            st.session_state["chunk_overlap"] = chunk_overlap

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "min_similarity": min_similarity,
        "document_ids": selected_doc_ids,
        "retrieval_mode": retrieval_mode,
        "reranker_enabled": reranker_enabled,
        "query_expansion_enabled": query_expansion_enabled,
        "parent_child_enabled": parent_child_enabled,
        "context_compression_enabled": context_compression_enabled,
    }


def render_sources(sources: List[Dict[str, Any]]):
    """Renders cited sources cleanly as responsive badge pills."""
    if not sources:
        return

    # Deduplicate sources while preserving encounter order
    seen = set()
    unique_sources = []
    for s in sources:
        key = (s.get("source"), s.get("page"))
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    pills_html = []
    for s in unique_sources:
        src = s.get("source", "Document")
        page = s.get("page", 1)
        pills_html.append(
            f'<div class="source-pill">📄 <span>{src}</span> · <span class="source-pill-page">Page {page}</span></div>'
        )

    st.markdown(
        f"""
        <div style="margin-top: 0.6rem; margin-bottom: 0.25rem;">
            <div style="font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #94A3B8; margin-bottom: 0.35rem;">
                Sources
            </div>
            <div class="source-container">
                {"".join(pills_html)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_retrieved_context(
    retrieved_chunks: List[RetrievedChunk],
    retrieval_query: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    document_ids: Optional[List[str]] = None,
    retrieval_mode: Optional[str] = None,
    reranking_enabled: bool = False,
    candidates_retrieved: Optional[int] = None,
    query_expansion_enabled: bool = False,
    expanded_queries: Optional[List[str]] = None,
    parent_child_enabled: bool = False,
    child_chunks_retrieved: Optional[int] = None,
    parent_contexts_used: Optional[int] = None,
    context_compression_enabled: bool = False,
    total_chars_original: Optional[int] = None,
    total_chars_compressed: Optional[int] = None,
):
    """
    Renders an expandable inspector for retrieved context chunks, query transformations,
    and ranking scores for technical demonstration.
    """
    if not retrieved_chunks:
        return

    with st.expander("🔍 Retrieval Inspector", expanded=False):
        if retrieval_query:
            st.markdown(f"**Retrieval Query Used:** `{retrieval_query}`")

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            if retrieval_mode:
                st.caption(f"**Strategy:** `{retrieval_mode.upper()}`")
            if similarity_threshold is not None:
                st.caption(f"**Similarity Threshold:** `{similarity_threshold:.2f}`")
        with col_cfg2:
            if document_ids:
                st.caption(f"**Search Scope:** Filtered to {len(document_ids)} document(s)")
            else:
                st.caption("**Search Scope:** All Indexed Documents")

        # Active Enhancements Status
        enhancement_notes = []
        if query_expansion_enabled:
            num_q = len(expanded_queries) if expanded_queries else 1
            enhancement_notes.append(f"Query Expansion ({num_q} queries)")
        if reranking_enabled:
            cand_str = f", {candidates_retrieved} candidates" if candidates_retrieved else ""
            enhancement_notes.append(f"TF-IDF Reranking ({len(retrieved_chunks)} final{cand_str})")
        if parent_child_enabled:
            child_cnt = child_chunks_retrieved if child_chunks_retrieved is not None else len(retrieved_chunks)
            parent_cnt = parent_contexts_used if parent_contexts_used is not None else len(retrieved_chunks)
            enhancement_notes.append(f"Parent/Child ({child_cnt} children → {parent_cnt} parents)")
        if context_compression_enabled:
            red_note = ""
            if total_chars_original and total_chars_compressed:
                pct = ((total_chars_original - total_chars_compressed) / total_chars_original) * 100
                red_note = f", {pct:.1f}% reduced"
            enhancement_notes.append(f"Context Compression{red_note}")

        if enhancement_notes:
            st.caption(f"**Active Enhancements:** {' • '.join(enhancement_notes)}")

        if query_expansion_enabled and expanded_queries and len(expanded_queries) > 1:
            with st.expander("Expanded Queries", expanded=False):
                for idx, q in enumerate(expanded_queries, start=1):
                    st.caption(f"{idx}. `{q}`")

        st.caption(
            f"Showing **{len(retrieved_chunks)}** context chunk(s) retrieved from the index:"
        )

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            doc_badge = (
                f" | <code>{chunk.document_id[:8]}...</code>"
                if getattr(chunk, "document_id", None)
                else ""
            )
            parent_badge = (
                f" | <strong>Parent:</strong> <code>{chunk.parent_id}</code>"
                if getattr(chunk, "parent_id", None)
                else ""
            )
            child_badge = (
                f" | <strong>Child:</strong> <code>{chunk.original_child_id}</code>"
                if getattr(chunk, "original_child_id", None)
                else ""
            )
            comp_badge = ""
            if getattr(chunk, "compression_ratio", None) is not None:
                kept = getattr(chunk, "sentences_kept", "?")
                tot = getattr(chunk, "sentences_total", "?")
                ratio_pct = f"{chunk.compression_ratio * 100:.1f}%"
                comp_badge = f" | <strong>Compressed:</strong> <code>{ratio_pct}</code> ({kept}/{tot} sents)"

            metric_parts = []
            if getattr(chunk, "retrieval_method", None):
                metric_parts.append(f"<strong>Mode:</strong> {chunk.retrieval_method}")
            if chunk.distance is not None and chunk.cosine_similarity is not None:
                metric_parts.append(f"<strong>Dist:</strong> {chunk.distance:.4f}")
                metric_parts.append(f"<strong>Sim:</strong> {chunk.cosine_similarity:.4f}")
            if getattr(chunk, "bm25_score", None) is not None:
                metric_parts.append(f"<strong>BM25:</strong> {chunk.bm25_score:.3f}")
            if getattr(chunk, "rrf_score", None) is not None:
                metric_parts.append(f"<strong>RRF:</strong> {chunk.rrf_score:.5f}")
            if getattr(chunk, "rerank_score", None) is not None:
                metric_parts.append(f"<strong>Rerank:</strong> {chunk.rerank_score:.4f}")
            if getattr(chunk, "original_rank", None) is not None:
                metric_parts.append(f"<strong>OrigRank:</strong> #{chunk.original_rank}")

            metrics_html = " | ".join(metric_parts) if metric_parts else "<span>No scores</span>"

            st.markdown(
                f"""
                <div class="chunk-container">
                    <div class="chunk-meta">
                        <span><strong>Context #{idx}</strong> | 📄 {chunk.source} (Page {chunk.page_number} • Chunk {chunk.chunk_index}){doc_badge}{parent_badge}{child_badge}{comp_badge}</span>
                        <span>{metrics_html}</span>
                    </div>
                    <div class="chunk-text">{chunk.text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_performance_metrics(
    timings: Optional[Any] = None,
    candidates_retrieved: Optional[int] = None,
    final_chunk_count: Optional[int] = None,
    total_chars_original: Optional[int] = None,
    total_chars_compressed: Optional[int] = None,
):
    """
    Renders a compact, expandable Performance drawer showing pipeline stage latencies
    and context throughput statistics for a single RAG turn.
    """
    if timings is None:
        return

    total_ms = getattr(timings, "total_ms", None)
    if total_ms is None:
        return

    with st.expander("⚡ Performance", expanded=False):
        executed_stages: List[Tuple[str, float]] = [("Total", total_ms)]

        rewrite_ms = getattr(timings, "query_rewrite_ms", None)
        if rewrite_ms is not None and rewrite_ms > 0:
            executed_stages.append(("Query Rewrite", rewrite_ms))

        expansion_ms = getattr(timings, "query_expansion_ms", None)
        if expansion_ms is not None and expansion_ms > 0:
            executed_stages.append(("Query Expansion", expansion_ms))

        retrieval_ms = getattr(timings, "retrieval_ms", None)
        if retrieval_ms is not None:
            executed_stages.append(("Retrieval", retrieval_ms))

        rerank_ms = getattr(timings, "reranking_ms", None)
        if rerank_ms is not None and rerank_ms > 0:
            executed_stages.append(("Reranking", rerank_ms))

        parent_ms = getattr(timings, "parent_resolution_ms", None)
        if parent_ms is not None and parent_ms > 0:
            executed_stages.append(("Parent Resolution", parent_ms))

        comp_ms = getattr(timings, "compression_ms", None)
        if comp_ms is not None and comp_ms > 0:
            executed_stages.append(("Compression", comp_ms))

        gen_ms = getattr(timings, "generation_ms", None)
        if gen_ms is not None:
            executed_stages.append(("Generation", gen_ms))

        # Render in compact columns
        cols = st.columns(len(executed_stages))
        for col, (stage_name, ms_val) in zip(cols, executed_stages):
            with col:
                st.metric(
                    label=stage_name,
                    value=f"{ms_val:,.1f} ms" if ms_val < 100 else f"{ms_val:,.0f} ms",
                )

        # Context throughput statistics
        stats_notes = []
        if candidates_retrieved is not None:
            cand_str = f"Candidates retrieved: **{candidates_retrieved}**"
            if final_chunk_count is not None:
                cand_str += f" → Final context chunks: **{final_chunk_count}**"
            stats_notes.append(cand_str)
        elif final_chunk_count is not None:
            stats_notes.append(f"Final context chunks: **{final_chunk_count}**")

        if total_chars_original is not None and total_chars_compressed is not None:
            if total_chars_original != total_chars_compressed:
                saved = total_chars_original - total_chars_compressed
                pct = (saved / total_chars_original) * 100 if total_chars_original > 0 else 0
                stats_notes.append(
                    f"Context characters: **{total_chars_original:,}** → **{total_chars_compressed:,}** (**{pct:.1f}% reduction**)"
                )
            else:
                stats_notes.append(f"Context payload: **{total_chars_original:,} characters**")
        elif total_chars_original is not None:
            stats_notes.append(f"Context payload: **{total_chars_original:,} characters**")

        if stats_notes:
            st.caption(" • ".join(stats_notes))
