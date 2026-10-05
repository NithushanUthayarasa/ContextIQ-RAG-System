"""
ContextIQ - RAG-Powered Document Intelligence System
Main Streamlit Application Entrypoint
"""

import hashlib
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import uuid

# Ensure repository root is on sys.path when executed directly as Streamlit entrypoint
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from app.config import (
    CONTEXT_COMPRESSION_ENABLED,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_PARENT_CHUNK_OVERLAP,
    DEFAULT_PARENT_CHUNK_SIZE,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_TOP_K,
    PARENT_CHILD_ENABLED,
    QUERY_EXPANSION_ENABLED,
    RERANKER_ENABLED,
    RERANKER_CANDIDATE_MULTIPLIER,
    UPLOAD_DIR,
    is_api_key_configured,
)
from app.generation.generator import GeminiGenerator, GeminiGenerationError
from app.ingestion.chunker import TextChunker
from app.ingestion.parent_chunker import ParentChildChunker
from app.ingestion.embedder import GeminiEmbedder, GeminiEmbedderError, MissingAPIKeyError
from app.ingestion.pdf_loader import (
    PDFLoader,
    PDFLoaderError,
    InvalidPDFError,
    EmptyPDFError,
    EncryptedPDFError,
    CorruptPDFError,
    ScannedOrEmptyPDFError,
)
from app.rag.conversation import Conversation
from app.rag.pipeline import RAGPipeline, RAGPipelineError
from app.rag.query_rewriter import QueryRewriter
from app.utils.error_handler import (
    safe_log_exception,
    translate_exception_to_user_message,
)

logger = logging.getLogger("contextiq.main")
from app.retrieval.retriever import Retriever, RetrieverError
from app.retrieval.reranker import TFIDFReranker
from app.retrieval.compressor import ExtractiveContextCompressor
from app.rag.query_expander import GeminiQueryExpander
from app.ui.components import (
    apply_custom_styles,
    render_header,
    render_performance_metrics,
    render_retrieved_context,
    render_sidebar,
    render_sources,
)
from app.ui.eval_dashboard import render_evaluation_dashboard
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
    created_staging = False
    if not staging_path.exists():
        with open(staging_path, "wb") as f:
            f.write(file_bytes)
        created_staging = True

    try:
        # 4. Extract text using PDFLoader preserving human-readable original filename as source
        loader = PDFLoader(staging_path, source_name=file_name)
        load_result = loader.load()

        # 5. Chunk pages into parent sections and child chunks with document_id identity
        parent_chunk_size = max(DEFAULT_PARENT_CHUNK_SIZE, chunk_size * 2)
        parent_chunker = ParentChildChunker(
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=DEFAULT_PARENT_CHUNK_OVERLAP,
            child_chunk_size=chunk_size,
            child_chunk_overlap=chunk_overlap,
        )
        parent_chunks, chunks = parent_chunker.split_pages(load_result.pages, document_id=document_id)

        # 6. Store parent sections in ChromaDB
        vector_store.add_parents(parent_chunks)

        # 7. Generate Gemini embeddings for child chunks
        embeddings = embedder.embed_documents(chunks)

        # 8. Store child chunks in ChromaDB
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
    except Exception as e:
        # Rollback partial ingestion: delete any indexed chunks for this document
        try:
            vector_store.delete_by_document_id(document_id)
        except Exception:
            pass
        # Remove staging file if created during this attempt
        if created_staging and staging_path.exists():
            try:
                staging_path.unlink()
            except Exception:
                pass
        raise


def handle_chat_turn(
    query: str,
    pipeline: RAGPipeline,
    conversation: Conversation,
    top_k: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    document_ids: Optional[List[str]] = None,
    retrieval_mode: Optional[str] = None,
    parent_child_enabled: Optional[bool] = None,
    context_compression_enabled: Optional[bool] = None,
) -> RAGResponse:
    """
    Executes a single conversational RAG turn.
    Passes conversation history to RAGPipeline for context-aware query rewriting.
    Records user and assistant messages with source/context/rewritten-query/document-filter/mode metadata upon success.
    """
    cleaned_query = query.strip()
    history = conversation.get_messages() if conversation else []
    ask_kwargs: Dict[str, Any] = {
        "top_k": top_k,
        "conversation_messages": history,
    }
    if similarity_threshold is not None:
        ask_kwargs["similarity_threshold"] = similarity_threshold
    if document_ids is not None:
        ask_kwargs["document_ids"] = document_ids
    if retrieval_mode is not None:
        ask_kwargs["retrieval_mode"] = retrieval_mode
    if parent_child_enabled is not None:
        ask_kwargs["parent_child_enabled"] = parent_child_enabled
    if context_compression_enabled is not None:
        ask_kwargs["context_compression_enabled"] = context_compression_enabled

    response = pipeline.ask(cleaned_query, **ask_kwargs)
    conversation.add_user_message(cleaned_query)
    conversation.add_assistant_message(
        content=response.answer,
        sources=response.sources,
        retrieved_chunks=response.retrieved_chunks,
        retrieval_query=response.retrieval_query,
        similarity_threshold=response.similarity_threshold,
        document_ids=response.document_ids,
        retrieval_mode=response.retrieval_mode,
        reranking_enabled=getattr(response, "reranking_enabled", False),
        candidates_retrieved=getattr(response, "candidates_retrieved", None),
        query_expansion_enabled=getattr(response, "query_expansion_enabled", False),
        expanded_queries=getattr(response, "expanded_queries", []),
        parent_child_enabled=getattr(response, "parent_child_enabled", False),
        child_chunks_retrieved=getattr(response, "child_chunks_retrieved", None),
        parent_contexts_used=getattr(response, "parent_contexts_used", None),
        context_compression_enabled=getattr(response, "context_compression_enabled", False),
        total_chars_original=getattr(response, "total_chars_original", None),
        total_chars_compressed=getattr(response, "total_chars_compressed", None),
        timings=getattr(response, "timings", None),
    )
    return response


# --- Caching Long-Lived Core Services ---
@st.cache_resource(show_spinner=False)
def get_vector_store(session_id: Optional[str] = None) -> ChromaVectorStore:
    if session_id:
        clean_id = "".join(c for c in session_id if c.isalnum() or c in ("_", "-"))
        collection_name = f"ctx_{clean_id}"
        return ChromaVectorStore(collection_name=collection_name)
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


def render_workspace(
    vector_store: Optional[ChromaVectorStore],
    config: Dict[str, Any],
    upload_dir: Path = UPLOAD_DIR,
):
    """
    Renders document ingestion, conversation history, and chat interaction interface.
    """
    # ==========================================
    # SECTION 1: Document Workspace
    # ==========================================
    st.markdown("### 📄 Document Workspace")

    uploaded_files = st.file_uploader(
        "Upload PDF documents to index:",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload standard text-based PDF documents (Scanned/OCR documents require preprocessing)",
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
                            upload_dir=upload_dir,
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
                        safe_log_exception(logger, f"Ingestion error for {file_name}", e)
                        status.write(f"❌ `{file_name}` — {translate_exception_to_user_message(e, context='pdf')}")
                        summary_stats["failed"] += 1
                    except (EmptyPDFError, EncryptedPDFError, CorruptPDFError, PDFLoaderError, InvalidPDFError) as e:
                        safe_log_exception(logger, f"Ingestion error for {file_name}", e)
                        status.write(f"❌ `{file_name}` — {translate_exception_to_user_message(e, context='pdf')}")
                        summary_stats["failed"] += 1
                    except (GeminiEmbedderError, MissingAPIKeyError) as e:
                        safe_log_exception(logger, f"Embedding API error for {file_name}", e)
                        status.write(f"❌ `{file_name}` — {translate_exception_to_user_message(e, context='embedding')}")
                        summary_stats["failed"] += 1
                    except VectorStoreError as e:
                        safe_log_exception(logger, f"Vector store error for {file_name}", e)
                        status.write(f"❌ `{file_name}` — {translate_exception_to_user_message(e, context='vectorstore')}")
                        summary_stats["failed"] += 1
                    except Exception as e:
                        safe_log_exception(logger, f"Unexpected ingestion error for {file_name}", e)
                        status.write(f"❌ `{file_name}` — {translate_exception_to_user_message(e, context='ingestion')}")
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
        st.info("No documents indexed yet. Upload one or more PDFs above to start asking questions.")

    # 1. Render Existing Conversation History
    messages = conversation.get_messages() if conversation else []
    if not messages:
        if indexed_count > 0:
            st.markdown(
                """
                <div class="empty-state-card">
                    <div class="empty-state-title">💬 Ask a question about your indexed documents</div>
                    <p class="empty-state-subtitle">ContextIQ retrieves relevant evidence passages and generates cited answers.</p>
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
                            similarity_threshold=getattr(msg, "similarity_threshold", None),
                            document_ids=getattr(msg, "document_ids", None),
                            retrieval_mode=getattr(msg, "retrieval_mode", None),
                            reranking_enabled=getattr(msg, "reranking_enabled", False),
                            candidates_retrieved=getattr(msg, "candidates_retrieved", None),
                            query_expansion_enabled=getattr(msg, "query_expansion_enabled", False),
                            expanded_queries=getattr(msg, "expanded_queries", []),
                            parent_child_enabled=getattr(msg, "parent_child_enabled", False),
                            child_chunks_retrieved=getattr(msg, "child_chunks_retrieved", None),
                            parent_contexts_used=getattr(msg, "parent_contexts_used", None),
                            context_compression_enabled=getattr(msg, "context_compression_enabled", False),
                            total_chars_original=getattr(msg, "total_chars_original", None),
                            total_chars_compressed=getattr(msg, "total_chars_compressed", None),
                        )
                    render_performance_metrics(
                        timings=getattr(msg, "timings", None),
                        candidates_retrieved=getattr(msg, "candidates_retrieved", None),
                        final_chunk_count=len(msg.retrieved_chunks) if msg.retrieved_chunks else None,
                        total_chars_original=getattr(msg, "total_chars_original", None),
                        total_chars_compressed=getattr(msg, "total_chars_compressed", None),
                    )

    # 2. Chat Input Interaction
    prompt = st.chat_input(
        "Upload a document above to begin chatting..."
        if indexed_count == 0
        else "Ask a question about your documents...",
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
                        default_min_similarity=config["min_similarity"],
                        default_retrieval_mode=config.get("retrieval_mode", DEFAULT_RETRIEVAL_MODE),
                    )
                    reranker = TFIDFReranker() if config.get("reranker_enabled") else None
                    query_expander = GeminiQueryExpander() if config.get("query_expansion_enabled") else None
                    compressor = (
                        ExtractiveContextCompressor()
                        if config.get("context_compression_enabled")
                        else None
                    )
                    pipeline = RAGPipeline(
                        retriever=retriever,
                        generator=generator,
                        query_rewriter=rewriter,
                        reranker=reranker,
                        reranker_candidate_multiplier=RERANKER_CANDIDATE_MULTIPLIER,
                        query_expander=query_expander,
                        parent_store=vector_store,
                        parent_child_enabled=config.get("parent_child_enabled", PARENT_CHILD_ENABLED),
                        compressor=compressor,
                        context_compression_enabled=config.get("context_compression_enabled", CONTEXT_COMPRESSION_ENABLED),
                    )
                    response = handle_chat_turn(
                        query=user_query,
                        pipeline=pipeline,
                        conversation=conversation,
                        top_k=config["top_k"],
                        similarity_threshold=config["min_similarity"],
                        document_ids=config.get("document_ids"),
                        retrieval_mode=config.get("retrieval_mode"),
                        parent_child_enabled=config.get("parent_child_enabled", PARENT_CHILD_ENABLED),
                        context_compression_enabled=config.get("context_compression_enabled", CONTEXT_COMPRESSION_ENABLED),
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
                        similarity_threshold=response.similarity_threshold,
                        document_ids=response.document_ids,
                        retrieval_mode=response.retrieval_mode,
                        reranking_enabled=getattr(response, "reranking_enabled", False),
                        candidates_retrieved=getattr(response, "candidates_retrieved", None),
                        query_expansion_enabled=getattr(response, "query_expansion_enabled", False),
                        expanded_queries=getattr(response, "expanded_queries", []),
                        parent_child_enabled=getattr(response, "parent_child_enabled", False),
                        child_chunks_retrieved=getattr(response, "child_chunks_retrieved", None),
                        parent_contexts_used=getattr(response, "parent_contexts_used", None),
                        context_compression_enabled=getattr(response, "context_compression_enabled", False),
                        total_chars_original=getattr(response, "total_chars_original", None),
                        total_chars_compressed=getattr(response, "total_chars_compressed", None),
                    )

                render_performance_metrics(
                    timings=getattr(response, "timings", None),
                    candidates_retrieved=getattr(response, "candidates_retrieved", None),
                    final_chunk_count=len(response.retrieved_chunks) if response.retrieved_chunks else None,
                    total_chars_original=getattr(response, "total_chars_original", None),
                    total_chars_compressed=getattr(response, "total_chars_compressed", None),
                )

            except (RetrieverError, GeminiGenerationError, RAGPipelineError, MissingAPIKeyError) as e:
                safe_log_exception(logger, "Chat turn RAG error", e)
                st.error(translate_exception_to_user_message(e, context="generation"))
            except Exception as e:
                safe_log_exception(logger, "Unexpected chat turn error", e)
                st.error(translate_exception_to_user_message(e, context="generation"))


def main():
    st.set_page_config(
        page_title="ContextIQ — Document Intelligence",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_styles()
    render_header()

    # Initialize session identifier for isolated multi-tenant Chroma storage
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex[:12]
    session_id = st.session_state["session_id"]
    session_upload_dir = UPLOAD_DIR / session_id

    # Verify API configuration
    if not is_api_key_configured():
        st.error(
            "⚠️ **Gemini API Key Missing**: Please set `GEMINI_API_KEY` in your `.env` file or Streamlit secrets to enable embeddings and generation."
        )

    # Initialize Vector Store scoped to this user session
    try:
        vector_store = get_vector_store(session_id)
    except Exception as e:
        safe_log_exception(logger, "Vector store initialization error", e)
        st.error(translate_exception_to_user_message(e, context="vectorstore"))
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
    if "min_similarity" not in st.session_state:
        st.session_state["min_similarity"] = DEFAULT_MIN_SIMILARITY
    if "retrieval_mode" not in st.session_state:
        st.session_state["retrieval_mode"] = DEFAULT_RETRIEVAL_MODE
    if "reranker_enabled" not in st.session_state:
        st.session_state["reranker_enabled"] = RERANKER_ENABLED
    if "query_expansion_enabled" not in st.session_state:
        st.session_state["query_expansion_enabled"] = QUERY_EXPANSION_ENABLED
    if "parent_child_enabled" not in st.session_state:
        st.session_state["parent_child_enabled"] = PARENT_CHILD_ENABLED
    if "context_compression_enabled" not in st.session_state:
        st.session_state["context_compression_enabled"] = CONTEXT_COMPRESSION_ENABLED

    # Render Sidebar with System Metrics and Hyperparameters
    config = render_sidebar(vector_store)

    # Top-level Navigation: Conversational Workspace vs. Evaluation Dashboard
    tab_chat, tab_eval = st.tabs(["💬 Workspace & Chat", "📊 Evaluation Dashboard"])

    with tab_chat:
        render_workspace(vector_store, config, upload_dir=session_upload_dir)

    with tab_eval:
        render_evaluation_dashboard()


if __name__ == "__main__":
    main()
