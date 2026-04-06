"""Integration tests for the chat interface flow.

These tests verify the query → response flow with conversation history.
"""
from unittest.mock import patch, MagicMock

import pytest

from rag_engine import RAGEngine


@pytest.fixture(autouse=True)
def mock_ollama_deps():
    """Mock the external llama-index dependencies that try to connect to Ollama."""
    with patch("llama_index.embeddings.ollama.OllamaEmbedding") as mock_embed, \
         patch("llama_index.llms.ollama.Ollama") as mock_llm:
        
        # Configure mock embedding model
        mock_embed_instance = MagicMock()
        # Mock the __call__ method for the embedding model if needed
        mock_embed.return_value = mock_embed_instance
        
        # Configure mock LLM
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        
        yield mock_embed_instance, mock_llm_instance

class TestQueryFlow:
    """Integration: index document → query → verify answer contains expected content."""

    def test_query_returns_answer_after_indexing(self, tmp_data_dir, tmp_chroma_dir, sample_txt_file):
        """After indexing a document, query should return a relevant answer."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        # Upload and index
        engine.upload_files([str(sample_txt_file)])
        
        # Mock index creation and query engine
        with patch.object(engine, '_build_index', return_value=True):
            engine._index = MagicMock()
            mock_query_engine = MagicMock()
            mock_query_engine.query.return_value = "This document is about a RAG system for testing."
            engine._index.as_query_engine.return_value = mock_query_engine
            
            engine.index()
            answer = engine.query("What is this document about?")
            
            assert answer == "This document is about a RAG system for testing."

    def test_query_with_no_index_raises_error(self, tmp_data_dir, tmp_chroma_dir):
        """Querying an empty engine should raise an error."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        
        with patch.object(engine, '_build_index', side_effect=FileNotFoundError("Data directory not found")):
            with pytest.raises(Exception):
                engine.query("Hello")


class TestChatSessionState:
    """Integration: conversation history persists across queries."""

    def test_multiple_queries_maintain_context(self, tmp_data_dir, tmp_chroma_dir, sample_txt_file):
        """Multiple queries should work sequentially without resetting state."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        engine.upload_files([str(sample_txt_file)])
        
        with patch.object(engine, '_build_index', return_value=True):
            engine._index = MagicMock()
            mock_query_engine = MagicMock()
            mock_query_engine.query.side_effect = ["I am a RAG system.", "I just told you, I am a RAG system."]
            engine._index.as_query_engine.return_value = mock_query_engine
            
            engine.index()

            ans1 = engine.query("What are you?")
            assert "RAG system" in ans1

            ans2 = engine.query("Can you repeat what you are?")
            assert "I am a RAG system" in ans2

    def test_query_engine_caching(self, tmp_data_dir, tmp_chroma_dir, sample_txt_file):
        """The query engine should be cached after first use."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        engine.upload_files([str(sample_txt_file)])
        
        with patch.object(engine, '_build_index', return_value=True):
            engine._index = MagicMock()
            mock_query_engine = MagicMock()
            mock_query_engine.query.return_value = "Test response"
            engine._index.as_query_engine.return_value = mock_query_engine
            
            engine.index()

            # First query builds the engine
            assert engine._query_engine is None
            engine.query("First query")
            assert engine._query_engine is not None

            # Second query reuses it
            first_engine_id = id(engine._query_engine)
            engine.query("Second query")
            assert id(engine._query_engine) == first_engine_id

            # Verify as_query_engine was only called once
            engine._index.as_query_engine.assert_called_once()


class TestErrorHandling:
    """Integration: failure modes when Ollama or ChromaDB are unavailable."""

    def test_query_with_no_index_shows_helpful_error(self, tmp_data_dir, tmp_chroma_dir):
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        
        with patch.object(engine, '_build_index', side_effect=ValueError("No documents found")):
            with pytest.raises(ValueError, match="No documents found"):
                engine.query("Hello")

    def test_query_when_ollama_down_shows_error_in_response(self, tmp_data_dir, tmp_chroma_dir, sample_txt_file):
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        engine.upload_files([str(sample_txt_file)])
        
        with patch.object(engine, '_build_index', return_value=True):
            engine._index = MagicMock()
            mock_query_engine = MagicMock()
            mock_query_engine.query.side_effect = Exception("Connection refused")
            engine._index.as_query_engine.return_value = mock_query_engine
            
            engine.index()

            with pytest.raises(Exception, match="Connection refused"):
                engine.query("Hello?")
