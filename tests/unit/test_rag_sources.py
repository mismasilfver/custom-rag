"""
Unit tests for the new query_with_sources functionality.
Run with: ./venv/bin/python -m pytest tests/unit/test_rag_sources.py -v
"""

from unittest.mock import MagicMock, patch


class TestCleanTextForDisplay:
    """Tests for _clean_text_for_display helper function."""

    def test_clean_text_removes_binary_characters(self):
        from rag_engine import _clean_text_for_display

        # Simulate binary PDF content with garbage bytes
        binary_content = "Hello\x00World\x01\x02\x03Test"
        cleaned = _clean_text_for_display(binary_content)
        assert "\x00" not in cleaned
        assert "Hello World Test" in cleaned

    def test_clean_text_handles_unicode(self):
        from rag_engine import _clean_text_for_display

        unicode_content = "Hello 世界 🌍 Test"
        cleaned = _clean_text_for_display(unicode_content)
        assert "Hello 世界 🌍 Test" in cleaned

    def test_clean_text_truncates_long_content(self):
        from rag_engine import _clean_text_for_display

        long_content = "X" * 500
        cleaned = _clean_text_for_display(long_content, max_length=100)
        assert len(cleaned) <= 103  # 100 + "..."
        assert cleaned.endswith("...")

    def test_clean_text_normalizes_whitespace(self):
        from rag_engine import _clean_text_for_display

        messy_content = "Hello    \n\n\t   World"
        cleaned = _clean_text_for_display(messy_content)
        assert cleaned == "Hello World"

    def test_clean_text_handles_empty_content(self):
        from rag_engine import _clean_text_for_display

        assert _clean_text_for_display("") == ""
        assert _clean_text_for_display(None) == ""


class TestRAGEngineQueryWithSources:
    """Tests for query_with_sources method."""

    def test_query_with_sources_returns_dict_with_answer_and_sources(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        # Mock the index and query engine
        mock_index = MagicMock()
        mock_response = MagicMock()
        mock_response.source_nodes = []
        mock_response.__str__ = lambda self: "Test answer"

        mock_query_engine = MagicMock()
        mock_query_engine.query.return_value = mock_response

        mock_index.as_query_engine.return_value = mock_query_engine
        engine._index = mock_index

        # Also need to mock _initialize_llm
        with patch.object(engine, "_initialize_llm", return_value=MagicMock()):
            result = engine.query_with_sources("What is this?")

        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
        assert result["answer"] == "Test answer"
        assert result["sources"] == []

    def test_query_with_sources_extracts_source_metadata(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        # Create mock source nodes with metadata
        mock_node1 = MagicMock()
        mock_node1.node.metadata = {
            "file_name": "doc1.pdf",
            "page_label": "5",
        }
        mock_node1.node.get_content.return_value = "This is the content of document 1"
        mock_node1.score = 0.95

        mock_node2 = MagicMock()
        mock_node2.node.metadata = {
            "file_name": "doc2.txt",
        }
        # Content over 200 chars to trigger truncation
        long_content = "X" * 250
        mock_node2.node.get_content.return_value = long_content
        mock_node2.score = 0.87

        mock_response = MagicMock()
        mock_response.source_nodes = [mock_node1, mock_node2]
        mock_response.__str__ = lambda self: "Test answer with sources"

        mock_query_engine = MagicMock()
        mock_query_engine.query.return_value = mock_response

        mock_index = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine
        engine._index = mock_index

        with patch.object(engine, "_initialize_llm", return_value=MagicMock()):
            result = engine.query_with_sources("What is this?")

        assert len(result["sources"]) == 2

        # Check first source
        source1 = result["sources"][0]
        assert source1["number"] == 1
        assert source1["file_name"] == "doc1.pdf"
        assert source1["page_label"] == "5"
        assert source1["score"] == 0.95
        assert "This is the content" in source1["snippet"]

        # Check second source (long content that gets truncated and cleaned)
        source2 = result["sources"][1]
        assert source2["number"] == 2
        assert source2["file_name"] == "doc2.txt"
        assert source2["page_label"] is None
        assert source2["score"] == 0.87
        # Should be truncated and cleaned: 200 X's + "..."
        assert source2["snippet"] == "X" * 200 + "..."

    def test_query_with_sources_cleans_binary_content(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        # Create mock source node with binary/garbage content
        # (simulating PDF extraction issues)
        mock_node = MagicMock()
        mock_node.node.metadata = {"file_name": "problematic.pdf"}
        # Simulate content with null bytes and control characters
        mock_node.node.get_content.return_value = "Readable\x00text\x01with\x02garbage"
        mock_node.score = 0.9

        mock_response = MagicMock()
        mock_response.source_nodes = [mock_node]
        mock_response.__str__ = lambda self: "Answer"

        mock_query_engine = MagicMock()
        mock_query_engine.query.return_value = mock_response

        mock_index = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine
        engine._index = mock_index

        with patch.object(engine, "_initialize_llm", return_value=MagicMock()):
            result = engine.query_with_sources("What?")

        assert len(result["sources"]) == 1
        source = result["sources"][0]
        # The snippet should be cleaned - no null bytes
        assert "\x00" not in source["snippet"]
        assert "Readable text with garbage" in source["snippet"]

    def test_query_with_sources_handles_missing_metadata(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        # Create mock source node with minimal metadata
        mock_node = MagicMock()
        mock_node.node.metadata = {}  # Empty metadata
        mock_node.node.get_content.return_value = "Short content"
        mock_node.score = None

        mock_response = MagicMock()
        mock_response.source_nodes = [mock_node]
        mock_response.__str__ = lambda self: "Answer"

        mock_query_engine = MagicMock()
        mock_query_engine.query.return_value = mock_response

        mock_index = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine
        engine._index = mock_index

        with patch.object(engine, "_initialize_llm", return_value=MagicMock()):
            result = engine.query_with_sources("What?")

        assert len(result["sources"]) == 1
        source = result["sources"][0]
        assert source["file_name"] == "Unknown"
        assert source["page_label"] is None
        assert source["score"] is None

    def test_query_with_sources_handles_no_source_nodes(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        # Response with no source_nodes attribute
        mock_response = MagicMock()
        del mock_response.source_nodes  # Remove the attribute entirely
        mock_response.__str__ = lambda self: "Answer without sources"

        mock_query_engine = MagicMock()
        mock_query_engine.query.return_value = mock_response

        mock_index = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine
        engine._index = mock_index

        with patch.object(engine, "_initialize_llm", return_value=MagicMock()):
            result = engine.query_with_sources("What?")

        assert result["answer"] == "Answer without sources"
        assert result["sources"] == []
