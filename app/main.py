"""
ContextIQ - RAG-Powered Document Intelligence System
Main Streamlit Application Entrypoint
"""

from pathlib import Path
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
from app.rag.pipeline import RAGPipeline, RAGPipelineError
from app.retrieval.retriever import Retriever, RetrieverError
from app.ui.components import (
    apply_custom_styles,
    render_header,
    render_retrieved_context,
    render_sidebar,
    render_sources,
)
from app.vectorstore.chroma_store import ChromaVectorStore, VectorStoreError


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

    # Render Sidebar with System Metrics and Hyperparameters
    config = render_sidebar(vector_store)

    # Maintain Session State defaults
    if "current_document" not in st.session_state:
        st.session_state["current_document"] = None
    if "current_pages" not in st.session_state:
        st.session_state["current_pages"] = 0
    if "last_response" not in st.session_state:
        st.session_state["last_response"] = None

    # ==========================================
    # SECTION 1: Document Upload & Indexing
    # ==========================================
    st.markdown("### 📄 Document Ingestion")

    uploaded_file = st.file_uploader(
        "Upload a PDF document to index:",
        type=["pdf"],
        help="Upload standard text-based PDF documents (Scanned/OCR not supported in V1)",
    )

    if uploaded_file is not None:
        file_size_kb = uploaded_file.size / 1024
        col_meta1, col_meta2 = st.columns([3, 1])
        with col_meta1:
            st.caption(f"**Selected:** `{uploaded_file.name}` ({file_size_kb:.1f} KB)")

        if st.button("🚀 Index Document", type="primary", use_container_width=True):
            if not is_api_key_configured():
                st.error("Cannot index document: GEMINI_API_KEY is not configured.")
                return

            # Save uploaded PDF to staging directory
            save_path = UPLOAD_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                with st.status("Indexing Document...", expanded=True) as status:
                    # 1. PDF Text Extraction
                    status.write("📖 Loading PDF and extracting text with PyMuPDF...")
                    loader = PDFLoader(save_path)
                    load_result = loader.load()

                    if load_result.has_empty_pages:
                        st.info(
                            f"Note: Pages {load_result.empty_pages} contained no extractable text."
                        )

                    status.write(
                        f"✓ Extracted {load_result.non_empty_page_count} page(s) with text (Total {load_result.total_pages} pages)."
                    )

                    # 2. Text Chunking
                    status.write("✂️ Segmenting text into overlapping chunks...")
                    chunker = TextChunker(
                        chunk_size=config["chunk_size"],
                        chunk_overlap=config["chunk_overlap"],
                    )
                    chunks = chunker.split_pages(load_result.pages)
                    status.write(
                        f"✓ Created {len(chunks)} chunks (size: {config['chunk_size']}, overlap: {config['chunk_overlap']})."
                    )

                    # 3. Vector Embeddings
                    status.write("🧠 Generating dense embeddings with Gemini...")
                    embedder = get_embedder()
                    embeddings = embedder.embed_documents(chunks)
                    status.write(f"✓ Generated {len(embeddings)} vector embeddings.")

                    # 4. Storage in ChromaDB
                    status.write("💾 Storing vectors and metadata in ChromaDB...")
                    vector_store.add_chunks(chunks, embeddings)
                    status.update(
                        label="✓ Document Indexed Successfully!",
                        state="complete",
                        expanded=False,
                    )

                # Update session state
                st.session_state["current_document"] = uploaded_file.name
                st.session_state["current_pages"] = load_result.total_pages
                st.session_state["last_response"] = None
                st.success(f"Indexed **{len(chunks)}** chunks from `{uploaded_file.name}` into ChromaDB.")

            except ScannedOrEmptyPDFError as e:
                st.error(f"❌ Scanned or Image-only PDF: {str(e)}")
            except (PDFLoaderError, InvalidPDFError) as e:
                st.error(f"❌ PDF Processing Error: {str(e)}")
            except (GeminiEmbedderError, MissingAPIKeyError) as e:
                st.error(f"❌ Embedding API Error: {str(e)}")
            except VectorStoreError as e:
                st.error(f"❌ Vector Storage Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected Error during indexing: {str(e)}")

    st.divider()

    # ==========================================
    # SECTION 2: Grounded Q&A Interface
    # ==========================================
    st.markdown("### 💬 Ask Your Document")

    indexed_count = vector_store.count() if vector_store else 0
    active_doc = st.session_state.get("current_document")

    if indexed_count == 0:
        st.info("ℹ️ No documents indexed yet. Upload and index a PDF above to begin asking questions.")

    question = st.text_input(
        "Enter your question:",
        placeholder="e.g. What are the key findings of this document?",
        disabled=(indexed_count == 0),
    )

    if st.button("🔍 Ask ContextIQ", type="primary", disabled=(indexed_count == 0)):
        if not question or not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Retrieving relevant context and generating grounded answer..."):
                    embedder = get_embedder()
                    generator = get_generator()
                    retriever = Retriever(
                        embedder=embedder,
                        vector_store=vector_store,
                        default_top_k=config["top_k"],
                    )
                    pipeline = RAGPipeline(retriever=retriever, generator=generator)
                    response = pipeline.ask(question.strip(), top_k=config["top_k"])
                    st.session_state["last_response"] = response

            except (RetrieverError, GeminiGenerationError, RAGPipelineError) as e:
                st.error(f"RAG Error: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected generation error: {str(e)}")

    # ==========================================
    # SECTION 3: Answer & Transparency Display
    # ==========================================
    last_resp = st.session_state.get("last_response")
    if last_resp is not None:
        st.markdown("---")
        st.markdown("### 💡 Answer")
        st.markdown(
            f"""
            <div style="background: #1E293B; border-left: 4px solid #38BDF8; padding: 1.2rem; border-radius: 8px; margin-bottom: 1.2rem;">
                <div style="font-size: 1.05rem; line-height: 1.6; color: #F1F5F9;">{last_resp.answer}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Sources
        render_sources(last_resp.sources)

        # Transparent retrieved context expander
        render_retrieved_context(last_resp.retrieved_chunks)


if __name__ == "__main__":
    main()
