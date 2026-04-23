"""Tests for conversation export functionality."""

from datetime import datetime

from rag_engine import RAGEngine


class TestExportConversationToMarkdown:
    """Tests for export_conversation_to_markdown method."""

    def test_empty_conversation_returns_empty_string(self):
        """Empty messages list should return empty string."""
        engine = RAGEngine()
        result = engine.export_conversation_to_markdown([])
        assert result == ""

    def test_basic_conversation_formatting(self):
        """Single user-assistant exchange formatted correctly."""
        engine = RAGEngine()
        messages = [
            {
                "role": "user",
                "content": "Hello",
                "timestamp": datetime(2025, 4, 23, 20, 20, 30),
            },
            {
                "role": "assistant",
                "content": "Hi there!",
                "timestamp": datetime(2025, 4, 23, 20, 20, 35),
            },
        ]

        result = engine.export_conversation_to_markdown(messages)

        assert "# Chat Export" in result
        assert "**User:** Hello" in result
        assert "**Assistant:** Hi there!" in result
        assert "20:20" in result or "2025" in result

    def test_conversation_without_sources_omits_sources_section(self):
        """Sources section not included when include_sources=False."""
        engine = RAGEngine()
        messages = [
            {
                "role": "assistant",
                "content": "Answer",
                "timestamp": datetime.now(),
                "sources": [
                    {
                        "number": 1,
                        "file_name": "doc.pdf",
                        "page_label": "5",
                        "snippet": "text",
                    }
                ],
            },
        ]

        result = engine.export_conversation_to_markdown(messages, include_sources=False)

        assert "**Sources:**" not in result
        assert "doc.pdf" not in result

    def test_conversation_with_sources_includes_sources_section(self):
        """Sources section included when include_sources=True."""
        engine = RAGEngine()
        messages = [
            {
                "role": "assistant",
                "content": "Answer",
                "timestamp": datetime.now(),
                "sources": [
                    {
                        "number": 1,
                        "file_name": "doc.pdf",
                        "page_label": "5",
                        "snippet": "source text",
                    },
                    {
                        "number": 2,
                        "file_name": "other.pdf",
                        "page_label": None,
                        "snippet": "more text",
                    },
                ],
            },
        ]

        result = engine.export_conversation_to_markdown(messages, include_sources=True)

        assert "**Sources:**" in result
        assert "[1] doc.pdf, Page 5" in result
        assert "source text" in result
        assert "[2] other.pdf" in result
        assert "more text" in result

    def test_source_without_page_label_omits_page_info(self):
        """Sources without page_label don't show page info."""
        engine = RAGEngine()
        messages = [
            {
                "role": "assistant",
                "content": "Answer",
                "timestamp": datetime.now(),
                "sources": [
                    {
                        "number": 1,
                        "file_name": "doc.txt",
                        "page_label": None,
                        "snippet": "text",
                    }
                ],
            },
        ]

        result = engine.export_conversation_to_markdown(messages, include_sources=True)

        assert "[1] doc.txt" in result
        assert ", Page" not in result

    def test_conversation_with_project_name_in_title(self):
        """Project name included in markdown title."""
        engine = RAGEngine()
        messages = [{"role": "user", "content": "Hi", "timestamp": datetime.now()}]

        result = engine.export_conversation_to_markdown(
            messages, project_name="my-project"
        )

        assert "my-project" in result
