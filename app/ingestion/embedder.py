"""
ContextIQ - Gemini Embedding Service
Generates dense vector embeddings using Google GenAI SDK and gemini-embedding-001.
"""

from typing import List, Optional
from google import genai
from google.genai import types

from app.config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    get_gemini_api_key,
    is_api_key_configured,
)
from app.ingestion.chunker import DocumentChunk


class GeminiEmbedderError(Exception):
    """Base exception for Gemini embedding failures."""
    pass


class MissingAPIKeyError(GeminiEmbedderError):
    """Raised when the Gemini API key is missing or not configured."""
    pass


class EmptyInputError(GeminiEmbedderError, ValueError):
    """Raised when an empty input or query is provided for embedding."""
    pass


class GeminiEmbedder:
    """
    Embedding service wrapping Google GenAI SDK models for document chunks and user queries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        self.api_key = api_key if api_key is not None else get_gemini_api_key()
        if not self.api_key or not self.api_key.strip() or self.api_key.startswith("your_"):
            raise MissingAPIKeyError(
                "Gemini API key is missing or invalid. Please configure GEMINI_API_KEY in your .env file."
            )

        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.dimension = dimension if dimension is not None else EMBEDDING_DIMENSION

        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise GeminiEmbedderError(f"Failed to initialize Google GenAI client: {str(e)}") from e

    def embed_texts(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 50,
    ) -> List[List[float]]:
        """
        Embeds a list of text strings in batches using the configured Gemini embedding model.

        Args:
            texts: List of text strings to embed.
            task_type: GenAI task type ('RETRIEVAL_DOCUMENT' or 'RETRIEVAL_QUERY').
            batch_size: Maximum texts per API batch call (default 50).

        Returns:
            List of float vectors, one per input text.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        config = types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.dimension,
        )

        # Process in batches to respect API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                    config=config,
                )
                if not response or not hasattr(response, "embeddings") or not response.embeddings:
                    raise GeminiEmbedderError(
                        f"Embedding API returned empty response for batch of size {len(batch)}."
                    )

                for emb in response.embeddings:
                    if hasattr(emb, "values") and emb.values is not None:
                        all_embeddings.append(list(emb.values))
                    else:
                        raise GeminiEmbedderError("Malformed embedding response: missing values.")

            except Exception as e:
                if isinstance(e, GeminiEmbedderError):
                    raise
                raise GeminiEmbedderError(
                    f"Gemini Embedding API call failed for model '{self.model_name}': {str(e)}"
                ) from e

        if len(all_embeddings) != len(texts):
            raise GeminiEmbedderError(
                f"Embedding count mismatch: expected {len(texts)}, got {len(all_embeddings)}"
            )

        return all_embeddings

    def embed_documents(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 50,
    ) -> List[List[float]]:
        """
        Generates dense vector embeddings for a list of DocumentChunks.
        Uses task_type="RETRIEVAL_DOCUMENT".

        Args:
            chunks: List of DocumentChunk objects.
            batch_size: Number of chunks per batch request.

        Returns:
            List of embedding vectors corresponding to the input chunks.
        """
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(
            texts=texts,
            task_type="RETRIEVAL_DOCUMENT",
            batch_size=batch_size,
        )

    def embed_query(self, query: str) -> List[float]:
        """
        Generates a dense vector embedding for a single user query.
        Uses task_type="RETRIEVAL_QUERY".

        Args:
            query: The user's question or search query string.

        Returns:
            List of floats representing the query vector.

        Raises:
            EmptyInputError: If query is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise EmptyInputError("Search query must not be empty.")

        cleaned_query = query.strip()
        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=self.dimension,
        )

        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=cleaned_query,
                config=config,
            )
            if (
                not response
                or not hasattr(response, "embeddings")
                or not response.embeddings
                or not hasattr(response.embeddings[0], "values")
            ):
                raise GeminiEmbedderError("Query embedding returned an empty or malformed vector.")

            return list(response.embeddings[0].values)
        except Exception as e:
            if isinstance(e, (EmptyInputError, GeminiEmbedderError)):
                raise
            raise GeminiEmbedderError(
                f"Gemini Embedding API call failed for query with model '{self.model_name}': {str(e)}"
            ) from e
