"""
ContextIQ - RAG-Powered Document Intelligence System
Main Streamlit Application Entrypoint
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
import streamlit as st

from app.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    UPLOAD_DIR,
    is_api_key_configured,
)
from app.generation.generator import GeminiGenerator, GeminiGenerationError
from app.ingestion.chunker import TextChunker
from app.ingestion.embedder import GeminiEmbedder, GeminiEmbedderError, MissingAPIKeyError
from app.ingestion.pdf_loader import (
    PDFLoader,
    PDFLoaderError,
    InvalidPDFError,
    ScannedOrEmptyPDFError,
)
from app.rag.conversation import Conversation
from app.rag.pipeline import RAGPipeline, RAGPipelineError
from app.rag.query_rewriter import QueryRewriter
from app.retrieval.retriever import Retriever, RetrieverError
from app.ui.components import (
    apply_custom_styles,
    render_header,
    render_retrieved_context,
    render_sidebar,
    render_sources,
)
from app.vectorstore.chroma_store import ChromaVectorStore, VectorStoreError


# --- Ingestion Helper with Multi-Document Guarantees ---
def ingest_pdf_bytes(
    file_name: str,
    file_bytes: bytes,
    vector_store: ChromaVectorStore,
    embedder: GeminiEmbedder,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    upload_dir: Path = UPLOAD_DIR,
) -> Dict[str, Any]:
    """
    Ingests raw PDF bytes with content-based SHA-256 identity, duplicate checking,
    isolated staging, chunking, embedding, and storage in ChromaDB.

    Returns:
        Dict containing status ('indexed' or 'skipped'), document_id, chunks count, etc.
    """
    # 1. Compute complete SHA-256 document identity from file content
    document_id = hashlib.sha256(file_bytes).hexdigest()

    # 2. Duplicate detection BEFORE saving to disk or embedding
    if vector_store.has_document(document_id):
        return {
            "status": "skipped",
            "document_id": document_id,
            "source": file_name,
            "message": "Already indexed",
            "chunks": 0,
            "pages": 0,
            "total_pages": 0,
            "empty_pages": [],
        }

    # 3. Safe staging path using full document_id to avoid filename collisions
    upload_dir.mkdir(parents=True, exist_ok=True)
    staging_path = upload_dir / f"{document_id}.pdf"
    if not staging_path.exists():
        with open(staging_path, "wb") as f:
            f.write(file_bytes)

    # 4. Extract text using PDFLoader preserving human-readable original filename as source
    loader = PDFLoader(staging_path, source_name=file_name)
    load_result = loader.load()

    # 5. Chunk pages with document_id identity
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.split_pages(load_result.pages, document_id=document_id)

    # 6. Generate Gemini embeddings for new chunks
    embeddings = embedder.embed_documents(chunks)

    # 7. Store in ChromaDB
    vector_store.add_chunks(chunks, embeddings)

    return {
        "status": "indexed",
        "document_id": document_id,
        "source": file_name,
        "chunks": len(chunks),
        "pages": load_result.non_empty_page_count,
        "total_pages": load_result.total_pages,
        "empty_pages": load_result.empty_pages,
    }


def handle_chat_turn(
    query: str,
    pipeline: RAGPipeline,
    conversation: Conversation,
    top_k: Optional[int] = None,
) -> RAGResponse:
    """
    Executes a single conversational RAG turn.
    Passes conversation history to RAGPipeline for context-aware query rewriting.
    Records user and assistant messages with source/context/rewritten-query metadata upon success.
    """
    cleaned_query = query.strip()
    history = conversation.get_messages() if conversation else []
    response = pipeline.ask(
        cleaned_query,
        top_k=top_k,
        conversation_messages=history,
    )
    conversation.add_user_message(cleaned_query)
    conversation.add_assistant_message(
        content=response.answer,
        sources=response.sources,
        retrieved_chunks=response.retrieved_chunks,
        retrieval_query=response.retrieval_query,
    )
    return response


# --- Caching Long-Lived Core Services ---
@st.cache_resource(show_spinner=False)
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore()


@st.cache_resource(show_spinner=False)
def get_embedder() -> GeminiEmbedder:
    return GeminiEmbedder()


@st.cache_resource(show_spinner=False)
def get_generator() -> GeminiGenerator:
    return GeminiGenerator()


@st.cache_resource(show_spinner=False)
def get_query_rewriter() -> Optional[QueryRewriter]:
    try:
        return QueryRewriter()
    except Exception:
        return None


def main():
    st.set_page_config(
        page_title="ContextIQ — Document Intelligence",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_styles()
    render_header()

    # Verify API configuration
    if not is_api_key_configured():
        st.error(
            "⚠️ **Gemini API Key Missing**: Please set `GEMINI_API_KEY` in your `.env` file to enable embeddings and generation."
        )

    # Initialize Vector Store
    try:
        vector_store = get_vector_store()
    except Exception as e:
        st.error(f"Failed to initialize vector database: {str(e)}")
        vector_store = None

    # Maintain Session State defaults
    if "current_document" not in st.session_state:
        st.session_state["current_document"] = None
    if "current_pages" not in st.session_state:
        st.session_state["current_pages"] = 0
    if "last_response" not in st.session_state:
        st.session_state["last_response"] = None
    if "deleting_doc_id" not in st.session_state:
        st.session_state["deleting_doc_id"] = None
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = Conversation()

    # Render Sidebar with System Metrics and Hyperparameters
    config = render_sidebar(vector_store)

    # ==========================================
    # SECTION 1: Document Upload & Indexing
    # ==========================================
    st.markdown("### 📄 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload PDF documents to index:",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload standard text-based PDF documents (Scanned/OCR not supported in V1)",
    )

    if uploaded_files:
        total_size_kb = sum(f.size for f in uploaded_files) / 1024
        col_meta1, col_meta2 = st.columns([3, 1])
        with col_meta1:
            names_summary = ", ".join(f"`{f.name}`" for f in uploaded_files[:3])
            if len(uploaded_files) > 3:
                names_summary += f" + {len(uploaded_files) - 3} more"
            st.caption(f"**Selected {len(uploaded_files)} file(s):** {names_summary} ({total_size_kb:.1f} KB)")

        if st.button("🚀 Index Documents", type="primary", use_container_width=True):
            if not is_api_key_configured():
                st.error("Cannot index documents: GEMINI_API_KEY is not configured.")
                return

            embedder = get_embedder()
            summary_stats = {"indexed": 0, "skipped": 0, "failed": 0, "total_chunks": 0}

            with st.status(f"Processing {len(uploaded_files)} Document(s)...", expanded=True) as status:
                for idx, file in enumerate(uploaded_files, start=1):
                    file_name = file.name
                    file_bytes = bytes(file.getbuffer())
                    status.write(f"**[{idx}/{len(uploaded_files)}]** Inspecting `{file_name}`...")

                    try:
                        res = ingest_pdf_bytes(
                            file_name=file_name,
                            file_bytes=file_bytes,
                            vector_store=vector_store,
                            embedder=embedder,
                            chunk_size=config["chunk_size"],
                            chunk_overlap=config["chunk_overlap"],
                        )

                        if res["status"] == "skipped":
                            status.write(f"⏭️ `{file_name}` — Already indexed (skipped).")
                            summary_stats["skipped"] += 1
                        else:
                            empty_note = (
                                f" (Pages {res['empty_pages']} had no text)"
                                if res.get("empty_pages")
                                else ""
                            )
                            status.write(
                                f"✅ `{file_name}` — Indexed {res['chunks']} chunks across {res['pages']} page(s){empty_note}."
                            )
                            summary_stats["indexed"] += 1
                            summary_stats["total_chunks"] += res["chunks"]
                            st.session_state["current_document"] = file_name
                            st.session_state["current_pages"] = res["total_pages"]

                    except ScannedOrEmptyPDFError as e:
                        status.write(f"❌ `{file_name}` — Scanned or image-only PDF: {str(e)}")
                        summary_stats["failed"] += 1
                    except (PDFLoaderError, InvalidPDFError) as e:
                        status.write(f"❌ `{file_name}` — PDF processing error: {str(e)}")
                        summary_stats["failed"] += 1
                    except (GeminiEmbedderError, MissingAPIKeyError) as e:
                        status.write(f"❌ `{file_name}` — Embedding API error: {str(e)}")
                        summary_stats["failed"] += 1
                    except VectorStoreError as e:
                        status.write(f"❌ `{file_name}` — Vector store error: {str(e)}")
                        summary_stats["failed"] += 1
                    except Exception as e:
                        status.write(f"❌ `{file_name}` — Unexpected error: {str(e)}")
                        summary_stats["failed"] += 1

                status.update(
                    label=f"✓ Ingestion Finished: {summary_stats['indexed']} indexed, {summary_stats['skipped']} skipped, {summary_stats['failed']} failed.",
                    state="complete",
                    expanded=False,
                )

            st.session_state["last_response"] = None

            if summary_stats["indexed"] > 0:
                st.success(
                    f"Successfully indexed **{summary_stats['indexed']}** document(s) with **{summary_stats['total_chunks']}** new chunks into ChromaDB."
                )
            if summary_stats["skipped"] > 0:
                st.info(f"Skipped **{summary_stats['skipped']}** already indexed document(s).")
            if summary_stats["failed"] > 0:
                st.warning(f"**{summary_stats['failed']}** document(s) encountered errors.")

    st.divider()

    # ==========================================
    # SECTION 2: Conversational RAG Interface
    # ==========================================
    st.markdown("### 💬 Chat with ContextIQ")

    indexed_count = vector_store.count() if vector_store else 0
    conversation = st.session_state.get("conversation")

    if indexed_count == 0:
        st.info("ℹ️ No documents indexed yet. Upload and index PDF document(s) above to begin chatting.")

    # 1. Render Existing Conversation History
    messages = conversation.get_messages() if conversation else []
    if not messages:
        if indexed_count > 0:
            st.markdown(
                """
                <div style="text-align: center; padding: 2.5rem 1rem; color: #94A3B8; background: #0F172A; border-radius: 8px; border: 1px dashed #334155; margin-bottom: 1.5rem;">
                    <h4 style="color: #F8FAFC; margin-bottom: 0.5rem;">💬 Start a conversation</h4>
                    <p style="margin: 0; font-size: 0.95rem;">Ask a question about your indexed documents below.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        for msg in messages:
            with st.chat_message(msg.role):
                st.markdown(msg.content)
                if msg.role == "assistant":
                    if msg.sources:
                        render_sources(msg.sources)
                    if msg.retrieved_chunks:
                        render_retrieved_context(
                            msg.retrieved_chunks,
                            retrieval_query=msg.retrieval_query,
                        )

    # 2. Chat Input Interaction
    prompt = st.chat_input(
        "Ask a question about your documents...",
        disabled=(indexed_count == 0),
    )

    if prompt and prompt.strip():
        user_query = prompt.strip()

        # Render user message in current run
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate and render assistant response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving relevant context and generating grounded answer..."):
                    embedder = get_embedder()
                    generator = get_generator()
                    rewriter = get_query_rewriter()
                    retriever = Retriever(
                        embedder=embedder,
                        vector_store=vector_store,
                        default_top_k=config["top_k"],
                    )
                    pipeline = RAGPipeline(
                        retriever=retriever,
                        generator=generator,
                        query_rewriter=rewriter,
                    )
                    response = handle_chat_turn(
                        query=user_query,
                        pipeline=pipeline,
                        conversation=conversation,
                        top_k=config["top_k"],
                    )
                    st.session_state["last_response"] = response

                # Render assistant content
                st.markdown(response.answer)

                if response.sources:
                    render_sources(response.sources)

                if response.retrieved_chunks:
                    render_retrieved_context(
                        response.retrieved_chunks,
                        retrieval_query=response.retrieval_query,
                    )

            except (RetrieverError, GeminiGenerationError, RAGPipelineError) as e:
                st.error(f"RAG Error: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected generation error: {str(e)}")


if __name__ == "__main__":
    main()
