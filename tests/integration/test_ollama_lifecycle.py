import json
from unittest.mock import MagicMock, patch

import pytest

from rag_engine import RAGEngine


class TestOllamaCheckAndModelList:
    """Integration: check Ollama status → list models → select model."""

    def test_check_ollama_then_list_models(
        self, tmp_data_dir, tmp_chroma_dir, ollama_models_response
    ):
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_tag_response = MagicMock()
        mock_tag_response.read.return_value = json.dumps(
            ollama_models_response
        ).encode()
        mock_tag_response.__enter__ = lambda s: s
        mock_tag_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen") as mock_urlopen:
            # First call: check_ollama (just needs to succeed)
            # Second call: list_models (returns model data)
            mock_urlopen.side_effect = [MagicMock(), mock_tag_response]

            is_running = engine.check_ollama()
            models = engine.list_models()

        assert is_running is True
        assert "llama3.1:8b" in models
        assert len(models) == 3

    def test_select_model_from_listed_models(
        self, tmp_data_dir, tmp_chroma_dir, ollama_models_response
    ):
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_tag_response = MagicMock()
        mock_tag_response.read.return_value = json.dumps(
            ollama_models_response
        ).encode()
        mock_tag_response.__enter__ = lambda s: s
        mock_tag_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_tag_response):
            models = engine.list_models()

        # User picks a different model
        engine.set_model(models[2])

        assert engine.model_name == "mistral:7b"
        assert engine._llm is None  # cached LLM cleared

    def test_start_then_stop_ollama(self, tmp_data_dir, tmp_chroma_dir):
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch("rag_engine.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 99999
            mock_popen.return_value = mock_process

            with patch.object(engine, "check_ollama", side_effect=[False, True]):
                with patch("rag_engine.time.sleep"):
                    started = engine.start_ollama()

            assert started is True
            assert engine._ollama_process is not None

            engine.stop_ollama()

            mock_process.terminate.assert_called_once()
            assert engine._ollama_process is None
