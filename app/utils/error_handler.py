"""
ContextIQ - Error Handling & Sanitization Module
Provides user-safe error translation, secret redaction, path sanitization, and structured logging.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("contextiq")

# Regex to match Google API keys (AIza...)
_API_KEY_REGEX = re.compile(r"AIza[0-9A-Za-z-_]{35}")

# Regex to match Windows absolute paths (e.g. C:\Users\... or D:\...)
_WINDOWS_PATH_REGEX = re.compile(r"[A-Za-z]:\\[^ \t\n\r\"']+")

# Regex to match Unix absolute paths (e.g. /home/... or /Users/...)
_UNIX_PATH_REGEX = re.compile(r"/(?:home|Users|tmp|var|etc|root)/[^ \t\n\r\"']+")


def sanitize_error_message(text: Optional[str]) -> str:
    """
    Strips local filesystem paths and potential API keys/secrets from error strings.
    """
    if not text:
        return ""

    sanitized = str(text)

    # Redact Google API keys
    sanitized = _API_KEY_REGEX.sub("[REDACTED_API_KEY]", sanitized)

    # Redact Windows paths
    sanitized = _WINDOWS_PATH_REGEX.sub("[local path]", sanitized)

    # Redact Unix paths
    sanitized = _UNIX_PATH_REGEX.sub("[local path]", sanitized)

    return sanitized.strip()


def translate_exception_to_user_message(exc: Exception, context: str = "") -> str:
    """
    Translates internal exceptions into user-safe, concise category messages.
    Prevents exposing stack traces, raw HTTP responses, secrets, or file paths.

    Categories handled:
    - Configuration Error
    - Authentication Error
    - Rate Limit / Quota Exceeded
    - Network / Temporary Service Error
    - Scanned / Empty PDF
    - Encrypted PDF
    - Corrupt / Invalid PDF
    - Vector Database Error
    - Generic Fallback
    """
    if exc is None:
        return "An unknown error occurred."

    exc_type_name = type(exc).__name__
    raw_msg = str(exc).lower()

    # 1. Configuration & Missing API Key
    if "missingapikey" in exc_type_name.lower() or "api key is missing" in raw_msg or "not configured" in raw_msg:
        return "Gemini API key is not configured. Please add your GEMINI_API_KEY in the settings or environment."

    # 2. Authentication & Permission Errors (HTTP 401, 403, PERMISSION_DENIED)
    if (
        "unauthenticated" in raw_msg
        or "permission_denied" in raw_msg
        or "401" in raw_msg
        or "403" in raw_msg
        or "api_key_invalid" in raw_msg
        or "forbidden" in raw_msg
        or "invalid api key" in raw_msg
    ):
        return "Authentication failed with the AI service. Please verify that your GEMINI_API_KEY is valid."

    # 3. Rate Limit / Quota Failures (HTTP 429, RESOURCE_EXHAUSTED)
    if (
        "429" in raw_msg
        or "resource_exhausted" in raw_msg
        or "quota" in raw_msg
        or "rate limit" in raw_msg
        or "too many requests" in raw_msg
    ):
        return "AI service rate limit or quota exceeded. Please wait a moment and try again."

    # 4. Vector Database / ChromaDB Errors
    if "vectorstore" in exc_type_name.lower() or "chroma" in raw_msg or "sqlite" in raw_msg:
        return "The vector database encountered an error. Please try resetting or re-indexing your documents."

    # 5. Network, Timeout, & Temporary Service Errors (HTTP 503, 504, UNAVAILABLE, DEADLINE_EXCEEDED)
    if (
        "503" in raw_msg
        or "504" in raw_msg
        or "unavailable" in raw_msg
        or "deadline_exceeded" in raw_msg
        or "timeout" in raw_msg
        or "timed out" in raw_msg
        or "network" in raw_msg
        or "connectionerror" in raw_msg
        or "connecterror" in raw_msg
        or "socket" in raw_msg
    ):
        return "AI service is temporarily unavailable or timed out. Please try again in a few moments."

    # 6. Scanned / OCR / Empty Text PDFs
    if "scannedoremptypdf" in exc_type_name.lower() or "no extractable text" in raw_msg:
        return "No extractable text was found in this PDF. Scanned or image-only documents require OCR."

    # 7. Password-Protected / Encrypted PDFs
    if "encryptedpdf" in exc_type_name.lower() or "password" in raw_msg or "encrypted" in raw_msg:
        return "This PDF is password-protected or encrypted, which is currently unsupported."

    # 8. Empty PDF Files (0 bytes or 0 pages)
    if "emptypdf" in exc_type_name.lower() or "0 pages" in raw_msg or "0 bytes" in raw_msg:
        return "The uploaded PDF file is empty (0 bytes or 0 pages)."

    # 9. Corrupt or Invalid PDF Files
    if "corruptpdf" in exc_type_name.lower() or "invalidpdf" in exc_type_name.lower() or "filedataerror" in raw_msg:
        return "The uploaded file appears to be corrupted or is not a valid PDF document."

    # 10. Context-specific fallbacks
    if context == "embedding":
        return "Failed to generate document embeddings. Please check your network connection and retry."
    if context == "generation":
        return "Failed to generate an answer. Please try rephrasing your question or checking your connection."

    return "An unexpected error occurred while processing your request. Please try again."


def safe_log_exception(
    module_logger: logging.Logger,
    context: str,
    exc: Exception,
    level: int = logging.ERROR,
) -> None:
    """
    Safely logs technical exception details to server logs with secret redaction.
    """
    sanitized_msg = sanitize_error_message(str(exc))
    module_logger.log(
        level,
        f"{context}: [{type(exc).__name__}] {sanitized_msg}",
        exc_info=False,
    )
