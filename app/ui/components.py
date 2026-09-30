"""
ContextIQ - UI Components Module
Reusable visual components for Streamlit interface: headers, sidebar, cards, sources, and context viewers.
"""

from typing import Any, Dict, List
import streamlit as st

from app.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
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
    """Renders the top title and branding banner."""
    st.markdown(
        """
        <div class="main-header">
            <div class="main-title">🧠 ContextIQ</div>
            <div class="main-subtitle">RAG-Powered Document Intelligence System</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(vector_store) -> Dict[str, Any]:
    """
    Renders the sidebar containing system status, model parameters, and document metrics.
    """
    with st.sidebar:
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
            st.caption("Add your GEMINI_API_KEY to the .env file to enable embeddings & LLM.")

        st.markdown(
            '<span class="status-badge online" style="margin-top: 6px;">● Vector Store: ChromaDB</span>',
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown("### 📊 Indexed Document")
        total_chunks = vector_store.count() if vector_store else 0
        current_doc = st.session_state.get("current_document", "None")
        total_pages = st.session_state.get("current_pages", 0)

        st.markdown(f"**File:** `{current_doc}`")
        st.markdown(f"**Pages:** `{total_pages}`")
        st.markdown(f"**Total Chunks in DB:** `{total_chunks}`")

        if total_chunks > 0:
            if st.button("🗑️ Clear Vector Database", use_container_width=True):
                vector_store.reset()
                st.session_state["current_document"] = None
                st.session_state["current_pages"] = 0
                st.session_state["last_response"] = None
                st.success("Vector database cleared.")
                st.rerun()

        st.divider()

        st.markdown("### 🛠️ Configuration")
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

        top_k = st.slider(
            "Top-K Chunks to Retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.get("top_k", DEFAULT_TOP_K),
        )
        st.session_state["top_k"] = top_k

        st.caption(f"**Embeddings:** `{EMBEDDING_MODEL_NAME}`")
        st.caption(f"**Generator:** `{GENERATION_MODEL_NAME}`")

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
    }


def render_sources(sources: List[Dict[str, Any]]):
    """Renders cited source badges."""
    if not sources:
        return

    st.markdown("#### 📚 Sources")
    pills_html = ""
    for s in sources:
        pills_html += f'<span class="source-pill">📄 {s["source"]} — Page {s["page"]}</span>'

    st.markdown(pills_html, unsafe_allow_html=True)


def render_retrieved_context(retrieved_chunks: List[RetrievedChunk]):
    """Renders an expandable inspector for retrieved context chunks."""
    if not retrieved_chunks:
        return

    with st.expander("🔎 View Retrieved Context (Transparency & Debugging)", expanded=False):
        st.caption(
            "Inspecting the raw chunks retrieved from ChromaDB before generation. "
            "Lower distance indicates higher semantic relevance."
        )
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            st.markdown(
                f"""
                <div class="chunk-container">
                    <div class="chunk-meta">
                        <span><strong>Chunk #{idx}</strong> | 📄 {chunk.source} (Page {chunk.page_number})</span>
                        <span><strong>Distance:</strong> {chunk.distance:.4f} | <strong>ID:</strong> {chunk.chunk_id}</span>
                    </div>
                    <div class="chunk-text">{chunk.text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
