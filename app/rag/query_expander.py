"""ContextIQ – Query Expansion Module

Provides an optional expansion of a retrieval query into multiple alternative search queries.
The expansion is performed by a Gemini LLM when enabled. The API mirrors the existing
`QueryRewriter` implementation so that the same client handling (environment‑key,
graceful fallback) can be reused.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from google import genai
from google.genai import types

from app.config import (
    GEMINI_API_KEY,
    GENERATION_MODEL_NAME,
    QUERY_EXPANSION_ENABLED,
    QUERY_EXPANSION_MAX_QUERIES,
)


class QueryExpanderError(Exception):
    """Base exception for query‑expansion failures."""

    pass


class BaseQueryExpander(ABC):
    """Abstract base class for query expansion.

    Sub‑classes must implement :meth:`expand` which returns a list of search queries.
    The original query must always be included as the first element.
    """

    @abstractmethod
    def expand(self, query: str, max_queries: int = 3) -> List[str]:
        raise NotImplementedError


class GeminiQueryExpander(BaseQueryExpander):
    """Gemini‑based implementation.

    The LLM is prompted to generate *alternative* search queries that preserve the
    original intent. The response is expected to be a newline‑separated list of
    queries. Any malformed response falls back to the original query only.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, client: Optional[genai.Client] = None):
        self.api_key = api_key if api_key is not None else GEMINI_API_KEY
        self.model_name = model_name or GENERATION_MODEL_NAME
        if client is not None:
            self.client = client
        elif self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"):
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None
        else:
            self.client = None

    def _prompt(self, original_query: str, max_queries: int) -> str:
        return (
            f"You are a search‑query expansion assistant. Produce up to {max_queries - 1} "
            "alternative search queries that capture the same intent as the original. "
            "Return each query on a separate line. Do NOT add explanations, numbers, or any other text.\n"
            f"Original query: {original_query}"
        )

    def _parse_response(self, response_text: str, original_query: str, max_queries: int) -> List[str]:
        # Split on newlines, strip whitespace, discard empties and duplicates.
        lines = [line.strip() for line in response_text.splitlines()]
        # Preserve order, ensure original is first.
        uniq: List[str] = []
        seen = set()
        # Insert original query first.
        if original_query and original_query not in seen:
            uniq.append(original_query)
            seen.add(original_query)
        for line in lines:
            if not line:
                continue
            if line == original_query:
                continue
            if line in seen:
                continue
            uniq.append(line)
            seen.add(line)
            if len(uniq) >= max_queries:
                break
        return uniq

    def expand(self, query: str, max_queries: int = 3) -> List[str]:
        # Validate max_queries – must be >= 1.
        if max_queries < 1:
            raise QueryExpanderError("QUERY_EXPANSION_MAX_QUERIES must be >= 1")
        cleaned = query.strip()
        if not cleaned:
            return []
        # If the expander is disabled or client unavailable, return only the original.
        if not QUERY_EXPANSION_ENABLED or self.client is None:
            return [cleaned]
        prompt = self._prompt(cleaned, max_queries)
        config = types.GenerateContentConfig(system_instruction="You are to generate alternative search queries.", temperature=0.0)
        try:
            response = self.client.models.generate_content(model=self.model_name, contents=prompt, config=config)
            if response and hasattr(response, "text") and response.text:
                expanded = self._parse_response(response.text, cleaned, max_queries)
                # Ensure at least the original query is present.
                return expanded if expanded else [cleaned]
            else:
                return [cleaned]
        except Exception:
            # Any failure results in fallback to the original query only.
            return [cleaned]
