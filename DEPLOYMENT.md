# ContextIQ — Streamlit Community Cloud Deployment Guide

This guide outlines the steps to deploy ContextIQ safely to [Streamlit Community Cloud](https://streamlit.io/cloud) after merging `feature/v2-rag` into `main`.

---

## 1. Pre-Deployment Checklist

- [ ] All unit, integration, and regression tests pass (`python -m pytest -q`).
- [ ] No git diff whitespace issues or merge conflicts exist (`git diff --check`).
- [ ] Working tree is clean and `feature/v2-rag` is merged into `main`.
- [ ] `.env` is **not** committed to git (verified in `.gitignore`).
- [ ] `.env.example` contains only placeholder credentials.
- [ ] Benchmark files are tracked and up to date (`evaluation/results.json`, `evaluation/REPORT.md`).

---

## 2. Streamlit Community Cloud Setup

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **Create app** and select **Deploy a public app from GitHub**.
3. Configure the deployment settings:
   - **Repository:** `NithushanUthayarasa/ContextIQ-RAG-System` (or your GitHub fork/repo)
   - **Branch:** `main`
   - **Main file path:** `app/main.py`
   - **App URL:** (Optional custom subdomain, e.g. `contextiq-rag.streamlit.app`)
4. Click **Advanced settings...** before deploying.

---

## 3. Secret Configuration

In the **Advanced settings** modal under **Secrets**, enter your Google Gemini API key:

```toml
GEMINI_API_KEY = "AIzaSy..."
```

> [!NOTE]
> - `.env` files are used exclusively for local workstation development.
> - On Streamlit Community Cloud, secrets are injected via `st.secrets["GEMINI_API_KEY"]`.
> - ContextIQ automatically resolves `GEMINI_API_KEY` from either source.

---

## 4. Storage & Persistence Notes

- **Ephemeral Filesystem:** Streamlit Community Cloud runs in containerized environments. Documents uploaded during a visitor session and indexed into ChromaDB are stored in local container storage (`data/uploads/` and `chroma_db/`).
- **Container Lifecycle:** Uploaded files and vectors reset when the cloud app restarts, sleeps after inactivity, or is redeployed.
- **Evaluation Dashboard:** The *Evaluation Dashboard* tab uses precomputed benchmark artifacts (`evaluation/results.json`) and requires zero cloud storage persistence or external API calls to display.

---

## 5. Post-Deployment Smoke Test

Once deployed, perform the following verification:

1. **Header & Navigation:** Verify the brand banner and two top-level tabs (*💬 Workspace & Chat* and *📊 Evaluation Dashboard*).
2. **System Status:** Check the sidebar — `● Gemini API: Connected` and `● Vector Store: ChromaDB`.
3. **Document Ingestion:** Upload a sample PDF document and click **Index Documents**. Confirm chunking and embedding succeed without errors.
4. **Conversational Turn:** Ask a domain question related to the uploaded PDF. Verify the generated answer includes rounded source badges (`📄 doc.pdf · Page X`).
5. **Retrieval Inspector:** Expand `🔍 Retrieval Inspector` and verify candidate chunks, scores, and active enhancements display properly.
6. **Performance Drawer:** Expand `⚡ Performance` and confirm per-stage timing metrics render.
7. **Evaluation Dashboard:** Switch to *📊 Evaluation Dashboard* and verify KPI cards, Layer Ablation table, and charts render offline without calling Gemini.
