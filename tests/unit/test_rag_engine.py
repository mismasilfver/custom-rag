import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open


class TestRAGEngineCheckOllama:
    """Tests for checking Ollama server connectivity."""

    def test_check_ollama_returns_true_when_server_is_reachable(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            assert engine.check_ollama() is True

    def test_check_ollama_returns_false_when_server_is_unreachable(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            assert engine.check_ollama() is False


class TestRAGEngineOllamaLifecycle:
    """Tests for starting and stopping Ollama."""

    def test_start_ollama_launches_subprocess(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch("rag_engine.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            # First call returns False (not running), second returns True (started)
            with patch.object(engine, "check_ollama", side_effect=[False, True]):
                with patch("rag_engine.time.sleep"):
                    result = engine.start_ollama()

            assert result is True
            mock_popen.assert_called_once()

    def test_start_ollama_returns_true_if_already_running(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch.object(engine, "check_ollama", return_value=True):
            result = engine.start_ollama()

        assert result is True

    def test_stop_ollama_terminates_process(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        mock_process = MagicMock()
        engine._ollama_process = mock_process

        engine.stop_ollama()

        mock_process.terminate.assert_called_once()

    def test_stop_ollama_does_nothing_when_no_process(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine._ollama_process = None

        engine.stop_ollama()  # should not raise


class TestRAGEngineListModels:
    """Tests for listing available Ollama models."""

    def test_list_models_returns_model_names(self, tmp_data_dir, tmp_chroma_dir, ollama_models_response):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(ollama_models_response).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            models = engine.list_models()

        assert models == ["llama3.1:8b", "nomic-embed-text:latest", "mistral:7b"]

    def test_list_models_returns_empty_list_when_ollama_unreachable(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            models = engine.list_models()

        assert models == []


class TestRAGEngineSetModel:
    """Tests for switching the LLM model."""

    def test_set_model_updates_model_name(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine.set_model("mistral:7b")

        assert engine.model_name == "mistral:7b"

    def test_set_model_clears_cached_llm(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine._llm = MagicMock()  # simulate a cached LLM

        engine.set_model("mistral:7b")

        assert engine._llm is None


class TestRAGEngineFileManagement:
    """Tests for uploading and listing files in the data directory."""

    def test_upload_files_copies_to_data_dir(self, tmp_data_dir, tmp_chroma_dir, sample_txt_file):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine.upload_files([str(sample_txt_file)])

        uploaded = list(tmp_data_dir.iterdir())
        assert len(uploaded) == 1
        assert uploaded[0].name == "sample.txt"

    def test_upload_files_rejects_unsupported_extension(self, tmp_data_dir, tmp_chroma_dir, tmp_path):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        bad_file = tmp_path / "malware.exe"
        bad_file.write_text("not a document")

        engine.upload_files([str(bad_file)])

        assert list(tmp_data_dir.iterdir()) == []

    def test_upload_multiple_files(self, tmp_data_dir, tmp_chroma_dir, sample_txt_file, sample_pdf_file):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine.upload_files([str(sample_txt_file), str(sample_pdf_file)])

        uploaded_names = sorted(f.name for f in tmp_data_dir.iterdir())
        assert uploaded_names == ["sample.pdf", "sample.txt"]

    def test_list_data_files_returns_file_names(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        (tmp_data_dir / "doc1.pdf").write_bytes(b"fake")
        (tmp_data_dir / "doc2.txt").write_text("hello")

        files = engine.list_data_files()

        assert sorted(files) == ["doc1.pdf", "doc2.txt"]

    def test_list_data_files_returns_empty_when_no_files(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        files = engine.list_data_files()

        assert files == []


class TestRAGEngineReset:
    """Tests for resetting chroma_db and data folder."""

    def test_reset_deletes_chroma_dir(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        tmp_chroma_dir.mkdir()
        (tmp_chroma_dir / "some_file").write_text("data")

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine.reset()

        assert not tmp_chroma_dir.exists()

    def test_reset_deletes_document_files_from_data_dir(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        (tmp_data_dir / "doc.pdf").write_bytes(b"fake")
        (tmp_data_dir / "notes.txt").write_text("hello")
        (tmp_data_dir / "keep.json").write_text("{}")  # non-document file

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine.reset()

        remaining = [f.name for f in tmp_data_dir.iterdir()]
        assert remaining == ["keep.json"]

    def test_reset_succeeds_when_nothing_exists(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        result = engine.reset()

        assert result is True
