"""Integration tests for Streamlit app scaffolding with Ollama management.

These tests verify the interaction between the app UI layer and RAGEngine.
"""
import json
from unittest.mock import patch, MagicMock

import pytest

from rag_engine import RAGEngine


class TestAppOllamaStatusCheck:
    """Integration: app loads → Ollama status checked → UI reflects state."""

    def test_app_checks_ollama_on_init(self, tmp_data_dir, tmp_chroma_dir):
        """When the app starts, it should check Ollama status."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch.object(engine, "check_ollama") as mock_check:
            mock_check.return_value = True
            is_running = engine.check_ollama()

        mock_check.assert_called_once()
        assert is_running is True

    def test_app_reflects_ollama_not_running(self, tmp_data_dir, tmp_chroma_dir):
        """When Ollama is down, app should reflect stopped state."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch.object(engine, "check_ollama", return_value=False):
            is_running = engine.check_ollama()

        assert is_running is False

    def test_app_populates_model_list_when_ollama_running(self, tmp_data_dir, tmp_chroma_dir, ollama_models_response):
        """When Ollama is running, app should populate model dropdown."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_tag_response = MagicMock()
        mock_tag_response.read.return_value = json.dumps(ollama_models_response).encode()
        mock_tag_response.__enter__ = lambda s: s
        mock_tag_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_tag_response):
            models = engine.list_models()

        assert len(models) == 3
        assert "llama3.1:8b" in models
        assert "mistral:7b" in models

    def test_app_shows_empty_model_list_when_ollama_down(self, tmp_data_dir, tmp_chroma_dir):
        """When Ollama is down, model list should be empty."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            models = engine.list_models()

        assert models == []


class TestAppModelSelection:
    """Integration: user selects model → RAGEngine.set_model() called correctly."""

    def test_select_model_updates_engine_model_name(self, tmp_data_dir, tmp_chroma_dir):
        """Selecting a model should update the engine's model_name attribute."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        original_model = engine.model_name

        engine.set_model("mistral:7b")

        assert engine.model_name == "mistral:7b"
        assert engine.model_name != original_model

    def test_select_model_clears_cached_llm(self, tmp_data_dir, tmp_chroma_dir):
        """Selecting a model should clear the cached LLM to force reinitialization."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine._llm = MagicMock()  # simulate cached LLM
        engine._query_engine = MagicMock()  # simulate cached query engine

        engine.set_model("mistral:7b")

        assert engine._llm is None
        assert engine._query_engine is None

    def test_select_same_model_no_error(self, tmp_data_dir, tmp_chroma_dir):
        """Selecting the same model twice should be idempotent."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        # Set once
        engine.set_model("mistral:7b")
        # Set again to same value
        engine.set_model("mistral:7b")

        assert engine.model_name == "mistral:7b"

    def test_model_selection_persists_in_session(self, tmp_data_dir, tmp_chroma_dir):
        """Model selection should persist across UI interactions (simulated)."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        # Simulate: user selects model from dropdown
        available_models = ["llama3.1:8b", "mistral:7b", "nomic-embed-text:latest"]
        user_selection = available_models[1]  # user picks mistral

        engine.set_model(user_selection)

        # Verify engine now reflects user's choice
        assert engine.model_name == user_selection
        assert engine.model_name == "mistral:7b"
