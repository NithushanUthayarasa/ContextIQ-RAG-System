"""
Unit tests for ContextIQ ChatMessage and Conversation models.
Tests verify user/assistant message creation, ordering, validation, clearing,
and state isolation without external dependencies.
"""

import pytest

from app.rag.conversation import (
    ChatMessage,
    Conversation,
    InvalidMessageRoleError,
    EmptyMessageContentError,
)


def test_user_message_creation():
    """Test 1: Verify adding a user message records role 'user' and content accurately."""
    conv = Conversation()
    msg = conv.add_user_message("What is RAG?")

    assert isinstance(msg, ChatMessage)
    assert msg.role == "user"
    assert msg.content == "What is RAG?"

    messages = conv.get_messages()
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "What is RAG?"


def test_assistant_message_creation():
    """Test 2: Verify adding an assistant message records role 'assistant' and content accurately."""
    conv = Conversation()
    msg = conv.add_assistant_message("Retrieval-Augmented Generation enhances LLM responses.")

    assert isinstance(msg, ChatMessage)
    assert msg.role == "assistant"
    assert msg.content == "Retrieval-Augmented Generation enhances LLM responses."

    messages = conv.get_messages()
    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].content == "Retrieval-Augmented Generation enhances LLM responses."


def test_message_ordering_preservation():
    """Test 3: Verify messages preserve exact sequential insertion order."""
    conv = Conversation()
    conv.add_user_message("Question 1")
    conv.add_assistant_message("Answer 1")
    conv.add_user_message("Question 2")
    conv.add_assistant_message("Answer 2")

    messages = conv.get_messages()
    assert len(messages) == 4
    expected = [
        ("user", "Question 1"),
        ("assistant", "Answer 1"),
        ("user", "Question 2"),
        ("assistant", "Answer 2"),
    ]
    for msg, (exp_role, exp_content) in zip(messages, expected):
        assert msg.role == exp_role
        assert msg.content == exp_content


def test_invalid_role_rejection():
    """Test 4: Verify roles other than 'user' and 'assistant' are rejected with error."""
    with pytest.raises(InvalidMessageRoleError):
        ChatMessage(role="system", content="System prompt instructions")

    with pytest.raises(InvalidMessageRoleError):
        ChatMessage(role="bot", content="Bot message")

    with pytest.raises(InvalidMessageRoleError):
        ChatMessage(role="", content="Empty role")

    with pytest.raises(InvalidMessageRoleError):
        ChatMessage(role="admin", content="Admin action")


def test_empty_content_rejection():
    """Test 5: Verify empty or whitespace-only message content is rejected with error."""
    with pytest.raises(EmptyMessageContentError):
        ChatMessage(role="user", content="")

    with pytest.raises(EmptyMessageContentError):
        ChatMessage(role="user", content="   \t \n  ")

    with pytest.raises(EmptyMessageContentError):
        conv = Conversation()
        conv.add_user_message("")

    with pytest.raises(EmptyMessageContentError):
        conv = Conversation()
        conv.add_assistant_message("   ")


def test_clear_conversation():
    """Test 6: Verify clear() empties all conversation history."""
    conv = Conversation()
    conv.add_user_message("Hello")
    conv.add_assistant_message("Hi there!")
    assert len(conv) == 2

    conv.clear()
    assert len(conv) == 0
    assert conv.get_messages() == []


def test_state_isolation_between_conversations():
    """Test 7: Verify two independent Conversation instances do not share state."""
    conv_a = Conversation()
    conv_b = Conversation()

    conv_a.add_user_message("Message in Conv A")

    assert len(conv_a) == 1
    assert len(conv_b) == 0
    assert conv_b.get_messages() == []

    # Also verify modifying the returned list from get_messages does not mutate internal state
    msgs = conv_a.get_messages()
    msgs.clear()
    assert len(conv_a.get_messages()) == 1
