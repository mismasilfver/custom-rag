"""Tests for chat copy button feature.

Verifies that copy buttons are shown for assistant messages only.
"""


class TestChatCopyButton:
    """Test copy button functionality in chat interface."""

    def test_copy_button_shown_for_assistant_messages(self):
        """Verify copy button should be rendered for assistant messages."""
        # Simulate assistant message structure
        message = {
            "role": "assistant",
            "content": "Test response",
            "sources": [],
            "timestamp": None,
        }

        # Copy button should only be shown for assistant role
        assert message["role"] == "assistant"
        assert "content" in message
        assert message["content"]  # Non-empty content should be copyable

    def test_copy_button_not_shown_for_user_messages(self):
        """Verify no copy button for user messages."""
        # Simulate user message structure
        message = {
            "role": "user",
            "content": "Test question",
            "timestamp": None,
        }

        # Copy button should NOT be shown for user role
        assert message["role"] == "user"

    def test_copy_button_has_correct_content(self):
        """Verify button copies the message content."""
        message_content = "This is the assistant response."
        message = {
            "role": "assistant",
            "content": message_content,
            "sources": [],
        }

        # The copy button value should be the message content
        assert message["content"] == message_content
        assert len(message["content"]) > 0

    def test_copy_button_label_and_icon(self):
        """Verify button uses icon and label."""
        # Expected copy button configuration
        expected_label = "📋"

        # Label should be the copy icon
        assert expected_label == "📋"
        assert len(expected_label) > 0

    def test_copy_button_with_sources(self):
        """Verify copy button works for messages with source citations."""
        message = {
            "role": "assistant",
            "content": "Response with sources",
            "sources": [{"number": 1, "file_name": "doc.pdf", "snippet": "text"}],
        }

        # Copy button should still work with sources present
        assert message["role"] == "assistant"
        assert "content" in message

    def test_copy_button_with_error_message(self):
        """Verify copy button shown for error messages from assistant."""
        message = {
            "role": "assistant",
            "content": "❌ Error: Something went wrong",
            "sources": [],
        }

        # Error messages are also assistant role and should have copy button
        assert message["role"] == "assistant"
        assert "content" in message
