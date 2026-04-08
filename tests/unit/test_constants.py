"""Tests for constants module."""

from constants import CITATION_PROMPT_TEMPLATE, SUPPORTED_EXTENSIONS


class TestSupportedExtensions:
    """Test SUPPORTED_EXTENSIONS constant."""

    def test_contains_expected_extensions(self):
        """Should contain expected file extensions."""
        expected = {".pdf", ".doc", ".docx", ".txt"}
        assert SUPPORTED_EXTENSIONS == expected

    def test_is_frozen_set(self):
        """Should be a frozenset for immutability."""
        assert isinstance(SUPPORTED_EXTENSIONS, frozenset)


class TestCitationPromptTemplate:
    """Test CITATION_PROMPT_TEMPLATE constant."""

    def test_contains_context_placeholder(self):
        """Should contain context_str placeholder."""
        assert "{context_str}" in CITATION_PROMPT_TEMPLATE

    def test_contains_query_placeholder(self):
        """Should contain query_str placeholder."""
        assert "{query_str}" in CITATION_PROMPT_TEMPLATE

    def test_mentions_citations(self):
        """Should mention citation format like [1], [2]."""
        assert "[1], [2]" in CITATION_PROMPT_TEMPLATE

    def test_mentions_references_section(self):
        """Should mention References section."""
        assert "References" in CITATION_PROMPT_TEMPLATE
