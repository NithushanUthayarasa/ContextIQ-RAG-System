"""
ContextIQ - Conversation Memory Module
Manages chat message models and conversation history state.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ConversationError(Exception):
    """Base exception for conversation errors."""
    pass


class InvalidMessageRoleError(ConversationError, ValueError):
    """Raised when an invalid chat message role is provided."""
    pass


class EmptyMessageContentError(ConversationError, ValueError):
    """Raised when message content is empty or whitespace-only."""
    pass


ALLOWED_ROLES = {"user", "assistant"}


@dataclass
class ChatMessage:
    """Represents a single chat message with role, text content, and optional response metadata."""
    role: str
    content: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: List[Any] = field(default_factory=list)
    retrieval_query: Optional[str] = None
    similarity_threshold: Optional[float] = None
    document_ids: Optional[List[str]] = None
    retrieval_mode: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.role, str) or self.role.strip().lower() not in ALLOWED_ROLES:
            raise InvalidMessageRoleError(
                f"Invalid message role '{self.role}'. Allowed roles are: {', '.join(sorted(ALLOWED_ROLES))}."
            )
        self.role = self.role.strip().lower()

        if not isinstance(self.content, str) or not self.content.strip():
            raise EmptyMessageContentError("Message content must not be empty or whitespace-only.")
        self.content = self.content.strip()


@dataclass
class Conversation:
    """Maintains sequential conversation history of ChatMessage objects."""
    messages: List[ChatMessage] = field(default_factory=list)

    def add_user_message(self, content: str) -> ChatMessage:
        """Appends a new user message to the conversation."""
        msg = ChatMessage(role="user", content=content)
        self.messages.append(msg)
        return msg

    def add_assistant_message(
        self,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        retrieved_chunks: Optional[List[Any]] = None,
        retrieval_query: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        document_ids: Optional[List[str]] = None,
        retrieval_mode: Optional[str] = None,
    ) -> ChatMessage:
        """Appends a new assistant message to the conversation with its cited sources and retrieved chunks."""
        msg = ChatMessage(
            role="assistant",
            content=content,
            sources=list(sources) if sources else [],
            retrieved_chunks=list(retrieved_chunks) if retrieved_chunks else [],
            retrieval_query=retrieval_query,
            similarity_threshold=similarity_threshold,
            document_ids=document_ids,
            retrieval_mode=retrieval_mode,
        )
        self.messages.append(msg)
        return msg

    def clear(self) -> None:
        """Clears all messages in the conversation history."""
        self.messages.clear()

    def get_messages(self) -> List[ChatMessage]:
        """Returns a shallow copy of messages to preserve internal state encapsulation."""
        return list(self.messages)

    def __len__(self) -> int:
        return len(self.messages)
