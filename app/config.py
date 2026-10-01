"""
ContextIQ - System Configuration
Centralized configuration loaded from environment variables (.env).
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# Load environment variables from .env
load_dotenv(dotenv_path=ENV_PATH)

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Model Configurations
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 768))
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL_NAME", "gemini-3.5-flash-lite")
MAX_CONTEXT_CHARACTERS = int(os.getenv("MAX_CONTEXT_CHARACTERS", 12000))

# RAG Hyperparameters
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", 1000))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 200))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
DEFAULT_MAX_REWRITE_HISTORY = int(os.getenv("DEFAULT_MAX_REWRITE_HISTORY", 6))

# Minimum Cosine Similarity Threshold for Retrieval Filtering
# Note: This is an initial baseline (0.50) and should be tuned using evaluation data.
DEFAULT_MIN_SIMILARITY = float(os.getenv("DEFAULT_MIN_SIMILARITY", "0.50"))
MIN_RETRIEVAL_SIMILARITY = DEFAULT_MIN_SIMILARITY

# Storage Directories
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "contextiq_documents")

# Ensure runtime directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)


def is_api_key_configured() -> bool:
    """Check if a non-empty Gemini API key is configured."""
    return bool(GEMINI_API_KEY and GEMINI_API_KEY.strip() and not GEMINI_API_KEY.startswith("your_"))
