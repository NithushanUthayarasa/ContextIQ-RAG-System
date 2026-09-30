"""
ContextIQ - RAG-Powered Document Intelligence System
Main Application Entry Point
"""

import streamlit as st
from app.config import (
    is_api_key_configured,
    EMBEDDING_MODEL_NAME,
    GENERATION_MODEL_NAME,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
)

def main():
    st.set_page_config(
        page_title="ContextIQ - Document Intelligence",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("ContextIQ — RAG-Powered Document Intelligence System")
    st.info("System initialized. Phase 2 setup complete.")

if __name__ == "__main__":
    main()
