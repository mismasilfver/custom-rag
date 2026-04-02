import json
import pytest
import shutil
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory, cleaned up after each test."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def tmp_chroma_dir(tmp_path):
    """Temporary ChromaDB directory, cleaned up after each test."""
    chroma_dir = tmp_path / "chroma_db"
    return chroma_dir


@pytest.fixture
def sample_txt_file(tmp_path):
    """A small .txt file for testing uploads and indexing."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a sample document for testing the RAG system.")
    return file_path


@pytest.fixture
def sample_pdf_file(tmp_path):
    """A dummy .pdf file (just bytes) for testing file management."""
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake pdf content for testing")
    return file_path


@pytest.fixture
def mock_ollama_responses():
    """Factory for mock Ollama HTTP responses."""
    def _make_response(status_code=200, body=None):
        mock_response = MagicMock()
        mock_response.status = status_code
        mock_response.read.return_value = json.dumps(body or {}).encode()
        return mock_response
    return _make_response


@pytest.fixture
def ollama_models_response():
    """Typical response from Ollama /api/tags endpoint."""
    return {
        "models": [
            {"name": "llama3.1:8b", "size": 4_700_000_000},
            {"name": "nomic-embed-text:latest", "size": 274_000_000},
            {"name": "mistral:7b", "size": 4_100_000_000},
        ]
    }
