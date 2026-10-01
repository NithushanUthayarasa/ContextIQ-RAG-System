"""
ContextIQ - UI Components Module
Reusable visual components for Streamlit interface: headers, sidebar, cards, sources, and context viewers.
"""

from typing import Any, Dict, List, Optional
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

        st.markdown("### 📚 Indexed Documents")
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
    """Renders cited sources on separate lines."""
    if not sources:
        return

    st.markdown("#### 📚 Sources")
    for s in sources:
        st.markdown(f"📄 {s['source']} — Page {s['page']}")


def render_retrieved_context(
    retrieved_chunks: List[RetrievedChunk],
    retrieval_query: Optional[str] = None,
):
    """Renders an expandable inspector for retrieved context chunks and the retrieval query used."""
    if not retrieved_chunks:
        return

    with st.expander("🔎 View Retrieved Context (Transparency & Debugging)", expanded=False):
        if retrieval_query:
            st.markdown(f"**Retrieval Query Used:** `{retrieval_query}`")
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
