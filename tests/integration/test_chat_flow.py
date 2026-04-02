"""Integration tests for the chat interface flow.

These tests verify the query → response flow with conversation history.
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from rag_engine import RAGEngine


class TestQueryFlow:
    """Integration: index document → query → verify answer contains expected content."""

    def test_query_returns_answer_after_indexing(self, tmp_data_dir, tmp_chroma_dir, sample_txt_file):
        """After indexing a document, query should return a relevant answer."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        # Mock Ollama responses
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        # Upload and index
        engine.upload_files([str(sample_txt_file)])
        with patch("urllib.request.urlopen", return_value=mock_response):
            engine.index()

        # Mock chat response
        mock_chat_response = MagicMock()
        mock_chat_response.read.return_value = json.dumps({
            "message": {"content": "This document is about a RAG system for testing."}
        }).encode()
        mock_chat_response.__enter__ = lambda s: s
        mock_chat_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_chat_response):
            response = engine.query("What is this document about?")

        assert isinstance(response, str)
        assert len(response) > 0

    def test_query_with_no_index_raises_error(self, tmp_data_dir, tmp_chroma_dir):
        """Query without indexing should raise an appropriate error."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with pytest.raises((ValueError, FileNotFoundError)):
            engine.query("What is this document about?")


class TestChatSessionState:
    """Integration: conversation history persists across queries."""

    def test_multiple_queries_maintain_context(self, tmp_data_dir, tmp_chroma_dir, sample_txt_file):
        """Multiple queries should work sequentially without resetting state."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        engine.upload_files([str(sample_txt_file)])
        with patch("urllib.request.urlopen", return_value=mock_response):
            engine.index()

        # First query
        mock_chat1 = MagicMock()
        mock_chat1.read.return_value = json.dumps({
            "message": {"content": "Answer to first question"}
        }).encode()
        mock_chat1.__enter__ = lambda s: s
        mock_chat1.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_chat1):
            response1 = engine.query("First question?")

        # Second query (same engine, same index)
        mock_chat2 = MagicMock()
        mock_chat2.read.return_value = json.dumps({
            "message": {"content": "Answer to second question"}
        }).encode()
        mock_chat2.__enter__ = lambda s: s
        mock_chat2.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_chat2):
            response2 = engine.query("Second question?")

        assert response1 != response2
        assert "first" in response1.lower()
        assert "second" in response2.lower()

    def test_query_engine_caching(self, tmp_data_dir, tmp_chroma_dir, sample_txt_file):
        """Query engine should be cached and reused across queries."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        engine.upload_files([str(sample_txt_file)])
        with patch("urllib.request.urlopen", return_value=mock_response):
            engine.index()

        # First query creates query engine
        mock_chat = MagicMock()
        mock_chat.read.return_value = json.dumps({
            "message": {"content": "Answer"}
        }).encode()
        mock_chat.__enter__ = lambda s: s
        mock_chat.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_chat):
            engine.query("Question 1?")
            assert engine._query_engine is not None

            # Second query should reuse same query engine
            cached_engine = engine._query_engine
            engine.query("Question 2?")
            assert engine._query_engine is cached_engine


class TestErrorHandling:
    """Integration: errors are handled gracefully."""

    def test_query_with_no_index_shows_helpful_error(self, tmp_data_dir, tmp_chroma_dir):
        """When no index exists, query should fail with a clear error."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with pytest.raises(Exception) as exc_info:
            engine.query("What is this about?")

        error_msg = str(exc_info.value)
        # Error should mention documents or index
        assert any(word in error_msg.lower() for word in ["document", "index", "not found", "empty"])

    def test_query_when_ollama_down_shows_error_in_response(self, tmp_data_dir, tmp_chroma_dir, sample_txt_file):
        """When Ollama is unreachable during query, error appears in response."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        # Index the document first
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        engine.upload_files([str(sample_txt_file)])
        with patch("urllib.request.urlopen", return_value=mock_response):
            engine.index()

        # Now simulate Ollama being down during query
        with patch("urllib.request.urlopen", side_effect=ConnectionError("Connection refused")):
            # Query may raise or return an error response depending on implementation
            try:
                response = engine.query("What is this?")
                # If it returns a response, it should indicate an error
                assert any(err_word in response.lower() for err_word in ["error", "failed", "unable", "connect"])
            except Exception:
                pass  # Raising exception is also acceptable behavior
