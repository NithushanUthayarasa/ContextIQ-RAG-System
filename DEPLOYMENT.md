# ContextIQ — Streamlit Community Cloud Deployment Guide

**Live Production Application:** [https://contextiq-ai-rag.streamlit.app/](https://contextiq-ai-rag.streamlit.app/)<br>
**Status:** Live & Operational<br>
**Hosting Platform:** Streamlit Community Cloud<br>
**Production Branch:** `main`<br>
**Application Entrypoint:** `app/main.py`

This document details the production deployment architecture, setup specifications, secret configuration, and operational verification procedures for ContextIQ on [Streamlit Community Cloud](https://streamlit.io/cloud).

---

## 1. Verified Deployment Checklist

- [x] **Test Suite**: All 369 unit, integration, and regression tests passing (`python -m pytest -q`).
- [x] **Cloud Import Resolution**: Streamlit Cloud entrypoint path verified (`tests/test_cloud_deployment_readiness.py`).
- [x] **Session Isolation**: Multi-user privacy & isolated vector stores verified (`tests/test_multi_user_session_isolation.py`).
- [x] **Git Cleanliness**: Zero merge conflicts, clean working tree, and no whitespace errors (`git diff --check`).
- [x] **Secret Isolation**: `.env` strictly excluded from git tracking via `.gitignore`.
- [x] **Benchmark Artifacts**: Offline evaluation data committed and tracked (`evaluation/results.json`, `evaluation/REPORT.md`).

---

## 2. Streamlit Community Cloud Configuration

For replicating or maintaining the production deployment:

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Select **Deploy an app** (or **Deploy a public app from GitHub**).
3. Set the deployment parameters:
   - **Repository:** `NithushanUthayarasa/ContextIQ-RAG-System`
   - **Branch:** `main`
   - **Main file path:** `app/main.py`
   - **App URL:** `https://contextiq-ai-rag.streamlit.app/`
4. Runtime environment:
   - **Python Version:** `3.14`
   - **Streamlit Version:** `1.65.0`
5. Open **Advanced settings...** to configure production credentials before launching.

---

## 3. Secret Configuration

In the Streamlit Cloud **Advanced settings** modal under **Secrets**, configure the Google Gemini API key:

```toml
GEMINI_API_KEY = "AIzaSy..."
```

> [!IMPORTANT]
> - Never commit `.env` or real API keys to GitHub.
> - On Streamlit Community Cloud, secrets are managed securely via `st.secrets["GEMINI_API_KEY"]`.
> - ContextIQ automatically resolves `GEMINI_API_KEY` dynamically from `st.secrets` first, falling back to local environment variables during workstation development.

---

## 4. Multi-User Session Isolation & Storage

ContextIQ features a privacy-preserving session architecture for public multi-tenant deployment:

- **Session Isolation**: Each visiting user session receives an ephemeral UUID generated via `st.session_state`.
- **Dynamic Vector Stores**: Vector collections in ChromaDB are session-scoped (`session_{uuid}`), guaranteeing that documents indexed by User A cannot be queried, listed, or accessed by User B.
- **Isolated Upload Staging**: Uploaded files are placed in session-isolated directories (`data/uploads/{session_id}/`).
- **Ephemeral Container Storage**: Community Cloud apps run in containerized environments. Uploaded PDFs and indexed vectors persist for the duration of the container session and reset safely when the container restarts or redeploys.
- **Zero-API Offline Evaluation Dashboard**: The *Evaluation Dashboard* tab operates entirely offline from precomputed benchmark artifacts (`evaluation/results.json`), requiring zero API calls or storage persistence.

---

## 5. Live Production Smoke Test Checklist

Verify the following operations on the live deployment:

1. **Header & Navigation**: Verify the brand header and two main tabs (*💬 Workspace & Chat* and *📊 Evaluation Dashboard*).
2. **System Status**: Check sidebar status badges (`● Gemini API: Connected` and `● Vector Store: ChromaDB`).
3. **Document Ingestion**: Upload a sample PDF and click **Index Documents**. Verify chunking, embedding, and collection metrics update.
4. **Conversational Turn**: Query the uploaded document and verify that the generated answer displays grounded citations with source and page badges (`📄 doc.pdf · Page X`).
5. **Retrieval Inspector**: Expand `🔍 Retrieval Inspector` to examine candidate chunks, similarity distances, BM25 scores, and rank fusion results.
6. **Performance Drawer**: Expand `⚡ Performance` to confirm per-stage latency tracking (Retrieval, Generation, Rewriting, Reranking, Compression).
7. **Evaluation Dashboard**: Switch to *📊 Evaluation Dashboard* and verify KPI cards, ablation tables, and charts render offline.
