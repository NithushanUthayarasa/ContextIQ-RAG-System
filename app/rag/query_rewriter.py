"""
ContextIQ - Context-Aware Query Rewriter Module
Reformulates conversational follow-up questions into standalone search queries using Gemini LLM.
"""

from typing import List, Optional
from google import genai
from google.genai import types

from app.config import (
    DEFAULT_MAX_REWRITE_HISTORY,
    GEMINI_API_KEY,
    GENERATION_MODEL_NAME,
)
from app.rag.conversation import ChatMessage


class QueryRewriterError(Exception):
    """Base exception for query rewriter failures."""
    pass


REWRITE_SYSTEM_INSTRUCTION = (
    "You are a search query reformulation assistant.\n"
    "Your task is to rewrite the user's latest follow-up question into a standalone, "
    "context-complete search query suitable for semantic vector retrieval.\n\n"
    "Strict Rules:\n"
    "1. Preserve the user's core intent, key entities, and specific concepts.\n"
    "2. Resolve ambiguous pronouns and references (e.g. 'it', 'they', 'this', 'that', 'its', 'their') "
    "using the provided conversation history.\n"
    "3. Do NOT answer the question. Do NOT generate commentary, preamble, notes, or explanations.\n"
    "4. Return ONLY the reformulated standalone query string.\n"
    "5. If the latest question is already completely standalone, self-contained, or introduces a new topic "
    "unrelated to prior conversation history, return the question EXACTLY as provided without changes."
)


class QueryRewriter:
    """
    Reformulates ambiguous follow-up questions into standalone search queries using Gemini.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_history_messages: int = DEFAULT_MAX_REWRITE_HISTORY,
        client: Optional[genai.Client] = None,
    ):
        self.api_key = api_key if api_key is not None else GEMINI_API_KEY
        self.model_name = model_name or GENERATION_MODEL_NAME
        self.max_history_messages = max_history_messages

        if client is not None:
            self.client = client
        elif self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"):
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None
        else:
            self.client = None

    def format_history(self, messages: List[ChatMessage]) -> str:
        """
        Formats recent conversation messages into a compact text transcript.
        Omits retrieved chunks, embeddings, and PDF data.
        """
        if not messages:
            return ""

        # Limit to the most recent window of messages
        recent = messages[-self.max_history_messages :]
        lines = []
        for msg in recent:
            role_label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role_label}: {msg.content}")

        return "\n".join(lines)

    def rewrite_query(
        self,
        question: str,
        conversation_messages: Optional[List[ChatMessage]] = None,
    ) -> str:
        """
        Rewrites a conversational follow-up question into a standalone retrieval query.

        Args:
            question: Current user question string.
            conversation_messages: Optional sequence of prior ChatMessage objects.

        Returns:
            The reformulated standalone search query, or the original question on fallback.
        """
        if not question or not question.strip():
            return ""

        cleaned_question = question.strip()

        # Step 11: First question optimization — bypass LLM rewrite if no history
        if not conversation_messages:
            return cleaned_question

        # Format compact history
        history_text = self.format_history(conversation_messages)
        if not history_text.strip():
            return cleaned_question

        # If client is not available, fallback to original question
        if self.client is None:
            return cleaned_question

        prompt = (
            f"Conversation History:\n"
            f"{history_text}\n\n"
            f"Latest Question:\n"
            f"{cleaned_question}\n\n"
            f"Standalone Query:"
        )

        config = types.GenerateContentConfig(
            system_instruction=REWRITE_SYSTEM_INSTRUCTION,
            temperature=0.0,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            if response and hasattr(response, "text") and response.text:
                rewritten = response.text.strip()
                # Clean any accidental outer quotes
                if (rewritten.startswith('"') and rewritten.endswith('"')) or (
                    rewritten.startswith("'") and rewritten.endswith("'")
                ):
                    rewritten = rewritten[1:-1].strip()
                if rewritten:
                    return rewritten

            # Step 10: Empty response fallback
            return cleaned_question
        except Exception:
            # Step 10: Fallback to original question on any API/network failure
            return cleaned_question
