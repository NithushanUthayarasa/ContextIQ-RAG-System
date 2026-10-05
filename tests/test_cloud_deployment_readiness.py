"""
Unit tests for ContextIQ Streamlit Community Cloud deployment readiness.
Verifies secret resolution, storage directory portability, missing key startup safety,
and evaluation benchmark artifact availability.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.config import (
    BASE_DIR,
    CHROMA_PERSIST_DIR,
    DATA_DIR,
    UPLOAD_DIR,
    get_gemini_api_key,
    is_api_key_configured,
)
from app.ui.eval_dashboard import (
    DEFAULT_REPORT_PATH,
    DEFAULT_RESULTS_PATH,
    load_evaluation_report,
    load_evaluation_results,
)


def test_requirements_contains_critical_cloud_dependencies():
    """Verify requirements.txt declares all direct third-party packages."""
    req_path = BASE_DIR / "requirements.txt"
    assert req_path.is_file()
    content = req_path.read_text(encoding="utf-8").lower()

    assert "streamlit" in content
    assert "google-genai" in content
    assert "pymupdf" in content
    assert "chromadb" in content
    assert "python-dotenv" in content
    assert "pandas" in content


def test_storage_paths_are_portable_path_objects():
    """Verify runtime storage paths use pathlib.Path and are within base project directory."""
    assert isinstance(BASE_DIR, Path)
    assert isinstance(DATA_DIR, Path)
    assert isinstance(UPLOAD_DIR, Path)
    assert isinstance(CHROMA_PERSIST_DIR, Path)


def test_gemini_api_key_resolution_from_streamlit_secrets():
    """Verify that get_gemini_api_key resolves key from st.secrets when env is empty."""
    mock_st = MagicMock()
    mock_st.secrets = {"GEMINI_API_KEY": "AIzaSy_mock_streamlit_cloud_secret_key_12345"}

    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
        with patch("app.config.GEMINI_API_KEY", ""):
            with patch.dict("sys.modules", {"streamlit": mock_st}):
                key = get_gemini_api_key()
                assert key == "AIzaSy_mock_streamlit_cloud_secret_key_12345"
                assert is_api_key_configured() is True


def test_missing_api_key_behavior():
    """Verify is_api_key_configured safely returns False when no key is set."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
        with patch("app.config.GEMINI_API_KEY", ""):
            mock_st = MagicMock()
            mock_st.secrets = {}
            with patch.dict("sys.modules", {"streamlit": mock_st}):
                assert is_api_key_configured() is False


def test_evaluation_dashboard_benchmark_files_exist_and_are_valid():
    """Verify precomputed evaluation artifacts are present and readable for the dashboard."""
    assert DEFAULT_RESULTS_PATH.is_file(), f"Missing benchmark results at {DEFAULT_RESULTS_PATH}"
    assert DEFAULT_REPORT_PATH.is_file(), f"Missing benchmark report at {DEFAULT_REPORT_PATH}"

    data = load_evaluation_results()
    assert data is not None
    assert "configurations" in data
    assert "C1_Hybrid_Baseline" in data["configurations"]
    assert "C5_Full_Advanced_RAG" in data["configurations"]

    report = load_evaluation_report()
    assert report is not None
    assert len(report) > 100
    assert "Evaluation Report" in report or "Benchmark" in report


def test_clean_import_without_api_key():
    """Verify that app modules can be cleanly imported without an active API key."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
        import app.main
        import app.ui.components
        import app.ui.eval_dashboard
        import app.retrieval.retriever

        assert app.main is not None
        assert app.ui.components is not None
        assert app.ui.eval_dashboard is not None


def test_app_main_resolves_when_sys_path_only_contains_app_dir():
    """
    Simulate Streamlit Community Cloud execution where sys.path[0] is the script's
    parent directory ('app') and the repository root is NOT originally on sys.path.
    Verifies that app/main.py self-heals sys.path so 'import app...' succeeds.
    """
    import subprocess
    import sys

    code = """
import sys, runpy
from pathlib import Path

app_dir = str(Path('.').resolve() / 'app')
sys.path = [p for p in sys.path if Path(p).resolve() != Path('.').resolve()]
sys.path.insert(0, app_dir)

mod = runpy.run_path('app/main.py')
assert 'main' in mod, 'main function not found in app/main.py'
print('OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed Streamlit Cloud simulation: {result.stderr}"
    assert "OK" in result.stdout
