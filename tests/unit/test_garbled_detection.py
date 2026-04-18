"""
Unit tests for is_snippet_garbled heuristic.

Run with: ./venv/bin/python -m pytest tests/unit/test_garbled_detection.py -v
"""


class TestIsSnippetGarbled:
    """Tests for the is_snippet_garbled heuristic function."""

    def test_clean_english_text_is_not_garbled(self):
        from rag_engine import is_snippet_garbled

        text = "The quick brown fox jumps over the lazy dog."
        assert is_snippet_garbled(text) is False

    def test_typical_pdf_citation_snippet_is_not_garbled(self):
        from rag_engine import is_snippet_garbled

        text = (
            "Our Process Michael and I have been doing various informal "
            "experiments with ChatGPT since it first came out."
        )
        assert is_snippet_garbled(text) is False

    def test_font_encoded_garbage_is_garbled(self):
        from rag_engine import is_snippet_garbled

        text = "!4\u0194 SD J4!b !EH)r )r0E N DHH 9HCH)EL 8zss]~zk=4k \u02aaJ6#"
        assert is_snippet_garbled(text) is True

    def test_mostly_symbols_and_numbers_is_garbled(self):
        from rag_engine import is_snippet_garbled

        text = "!9TC yUjC rl 1Cf K\u3161%\u5855 \u0361m \u0761 +P( <TL|RJ_ y\u00b0 } SX"
        assert is_snippet_garbled(text) is True

    def test_empty_string_is_not_garbled(self):
        from rag_engine import is_snippet_garbled

        assert is_snippet_garbled("") is False

    def test_text_with_some_unicode_but_mostly_readable_is_not_garbled(self):
        from rag_engine import is_snippet_garbled

        text = "The caf\u00e9 served cr\u00eapes and caf\u00e9 au lait in the morning."
        assert is_snippet_garbled(text) is False

    def test_short_snippet_with_low_ascii_ratio_is_garbled(self):
        from rag_engine import is_snippet_garbled

        text = "\u1234\u5678\u9abc\u1234\u5678"
        assert is_snippet_garbled(text) is True
