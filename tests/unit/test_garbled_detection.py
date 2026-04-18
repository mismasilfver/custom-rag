"""
Unit tests for is_snippet_garbled and sources_contain_garbled heuristics.

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


class TestSourcesContainGarbled:
    """Tests for the sources_contain_garbled helper."""

    def test_returns_false_for_empty_sources(self):
        from rag_engine import sources_contain_garbled

        assert sources_contain_garbled([]) is False

    def test_returns_false_when_all_snippets_are_clean(self):
        from rag_engine import sources_contain_garbled

        sources = [
            {"number": 1, "snippet": "This is clean readable text."},
            {"number": 2, "snippet": "Another readable sentence here."},
        ]
        assert sources_contain_garbled(sources) is False

    def test_returns_true_when_any_snippet_is_garbled(self):
        from rag_engine import sources_contain_garbled

        sources = [
            {"number": 1, "snippet": "This is clean readable text."},
            {
                "number": 2,
                "snippet": "!4\u0194 SD J4!b !EH)r )r0E N DHH 9HCH)EL",
            },
        ]
        assert sources_contain_garbled(sources) is True

    def test_returns_true_when_all_snippets_are_garbled(self):
        from rag_engine import sources_contain_garbled

        sources = [
            {"number": 1, "snippet": "!4\u0194 SD J4!b !EH)r )r0E"},
            {"number": 2, "snippet": "Z^r O{UpKf3{~\u04ca ,J\u0159eXOD"},
        ]
        assert sources_contain_garbled(sources) is True

    def test_handles_missing_snippet_key(self):
        from rag_engine import sources_contain_garbled

        sources = [{"number": 1}]
        assert sources_contain_garbled(sources) is False
