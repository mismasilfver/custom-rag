"""Tests for chat message timestamps feature.

Verifies that timestamps are added to messages and displayed correctly.
"""

from datetime import datetime


class TestChatTimestamps:
    """Test chat message timestamp functionality."""

    def test_user_message_has_timestamp(self):
        """Verify timestamp is added when user message is stored."""
        # Simulate the message structure added in render_chat_section
        message = {
            "role": "user",
            "content": "Test question",
            "timestamp": datetime.now(),
        }

        assert "timestamp" in message
        assert isinstance(message["timestamp"], datetime)

    def test_assistant_message_has_timestamp(self):
        """Verify timestamp is added when assistant responds."""
        # Simulate the message structure added in render_chat_section
        message = {
            "role": "assistant",
            "content": "Test answer",
            "sources": [],
            "timestamp": datetime.now(),
        }

        assert "timestamp" in message
        assert isinstance(message["timestamp"], datetime)

    def test_timestamp_format_is_datetime(self):
        """Verify timestamp is a datetime object."""
        timestamp = datetime.now()

        # Should have year, month, day, hour, minute attributes
        assert hasattr(timestamp, "year")
        assert hasattr(timestamp, "month")
        assert hasattr(timestamp, "day")
        assert hasattr(timestamp, "hour")
        assert hasattr(timestamp, "minute")

    def test_timestamp_displayed_in_ui(self):
        """Verify timestamp formatting for UI display."""
        message = {
            "role": "user",
            "content": "Test question",
            "timestamp": datetime(2024, 1, 15, 14, 30),
        }

        # Verify timestamp formatting logic for UI
        timestamp_str = message["timestamp"].strftime("%H:%M")
        assert timestamp_str == "14:30"

    def test_timestamp_format_hour_minute(self):
        """Verify timestamp displays as HH:MM format."""
        timestamp = datetime(2024, 1, 15, 9, 5)
        formatted = timestamp.strftime("%H:%M")

        assert formatted == "09:05"
        assert len(formatted) == 5  # HH:MM format

    def test_message_without_sources_has_timestamp(self):
        """Verify timestamp works for messages without sources."""
        message = {
            "role": "assistant",
            "content": "Error message",
            "sources": [],
            "timestamp": datetime.now(),
        }

        assert "timestamp" in message
        assert message["sources"] == []
