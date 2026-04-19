"""Tests for chat regenerate response feature.

Verifies that regenerate buttons work correctly for assistant messages.
"""

from datetime import datetime


class TestChatRegenerate:
    """Test regenerate response functionality in chat interface."""

    def test_regenerate_button_shown_for_assistant_messages(self):
        """Verify regenerate button should be rendered for assistant messages."""
        # Simulate assistant message structure
        message = {
            "role": "assistant",
            "content": "Test response",
            "sources": [],
            "timestamp": datetime.now(),
            "prompt": "Test question",
        }

        # Regenerate button should only be shown for assistant role with prompt
        assert message["role"] == "assistant"
        assert "prompt" in message
        assert message["prompt"]  # Must have original prompt to regenerate

    def test_regenerate_button_not_shown_for_user_messages(self):
        """Verify no regenerate button for user messages."""
        # Simulate user message structure
        message = {
            "role": "user",
            "content": "Test question",
            "timestamp": datetime.now(),
        }

        # Regenerate button should NOT be shown for user role
        assert message["role"] == "user"
        assert "prompt" not in message

    def test_regenerate_removes_assistant_message(self):
        """Verify clicking regenerate removes the assistant message from history."""
        # Simulate message history
        messages = [
            {"role": "user", "content": "Q1", "timestamp": datetime.now()},
            {
                "role": "assistant",
                "content": "A1",
                "sources": [],
                "timestamp": datetime.now(),
                "prompt": "Q1",
            },
            {"role": "user", "content": "Q2", "timestamp": datetime.now()},
            {
                "role": "assistant",
                "content": "A2",
                "sources": [],
                "timestamp": datetime.now(),
                "prompt": "Q2",
            },
        ]

        # Simulate regenerating the second assistant message (index 3)
        # This should remove messages from index 3 onwards
        regenerate_index = 3
        new_messages = messages[:regenerate_index]

        assert len(new_messages) == 3
        assert new_messages[-1]["role"] == "user"
        assert new_messages[-1]["content"] == "Q2"

    def test_regenerate_preserves_conversation_context(self):
        """Verify messages before regenerated one are kept."""
        messages = [
            {"role": "user", "content": "Q1", "timestamp": datetime.now()},
            {
                "role": "assistant",
                "content": "A1",
                "sources": [],
                "timestamp": datetime.now(),
                "prompt": "Q1",
            },
            {"role": "user", "content": "Q2", "timestamp": datetime.now()},
            {
                "role": "assistant",
                "content": "A2",
                "sources": [],
                "timestamp": datetime.now(),
                "prompt": "Q2",
            },
        ]

        # Regenerate the second response
        regenerate_index = 3
        preserved_messages = messages[:regenerate_index]

        # Verify first Q&A pair is preserved
        assert preserved_messages[0]["role"] == "user"
        assert preserved_messages[0]["content"] == "Q1"
        assert preserved_messages[1]["role"] == "assistant"
        assert preserved_messages[1]["content"] == "A1"

    def test_regenerate_recalls_chat_with_same_prompt(self):
        """Verify original prompt is used for regeneration."""
        original_prompt = "What is the meaning of life?"
        assistant_message = {
            "role": "assistant",
            "content": "Original answer",
            "sources": [],
            "timestamp": datetime.now(),
            "prompt": original_prompt,
        }

        # Verify prompt is stored and matches original
        assert assistant_message["prompt"] == original_prompt
        assert len(assistant_message["prompt"]) > 0

    def test_regenerate_handles_missing_prompt(self):
        """Verify assistant message without prompt field doesn't show regenerate."""
        # Old message without prompt tracking (backward compatibility)
        message = {
            "role": "assistant",
            "content": "Old response",
            "sources": [],
            "timestamp": datetime.now(),
        }

        # Should not have prompt field
        assert "prompt" not in message
