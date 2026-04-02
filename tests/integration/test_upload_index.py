"""Integration tests for file management + indexing UI flows.

These tests verify the upload → index → reset integration flow.
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from rag_engine import RAGEngine


class TestUploadIndexFlow:
    """Integration: upload file → index → verify ChromaDB has documents."""

    def test_upload_file_then_index_creates_collection(self, tmp_data_dir, tmp_chroma_dir, sample_txt_file):
        """Upload a text file and index it; ChromaDB collection should exist with documents."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        # Mock Ollama responses for embedding generation
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        # Upload the file
        engine.upload_files([str(sample_txt_file)])
        assert len(engine.list_data_files()) == 1

        # Index with mocked Ollama calls
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = mock_response
            engine.index()

        # Verify ChromaDB collection was created
        import chromadb
        client = chromadb.PersistentClient(path=str(tmp_chroma_dir))
        collections = client.list_collections()
        assert any(c.name == "documents" for c in collections)

    def test_upload_rejects_unsupported_extensions(self, tmp_data_dir, tmp_chroma_dir, tmp_path):
        """Upload should reject .exe, .sh, etc. and only accept supported document types."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        good_file = tmp_path / "valid.txt"
        good_file.write_text("valid content")
        bad_file = tmp_path / "script.exe"
        bad_file.write_bytes(b"malicious content")

        engine.upload_files([str(good_file), str(bad_file)])

        # Only the valid file should be uploaded
        files = engine.list_data_files()
        assert files == ["valid.txt"]

    def test_reindex_clears_and_rebuilds_index(self, tmp_data_dir, tmp_chroma_dir, sample_txt_file):
        """Reindex should clear existing collection and rebuild from scratch."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        # First index
        engine.upload_files([str(sample_txt_file)])
        with patch("urllib.request.urlopen", return_value=mock_response):
            engine.index()

        # Reindex
        with patch("urllib.request.urlopen", return_value=mock_response):
            engine.reindex()

        # Collection should still exist after reindex
        import chromadb
        client = chromadb.PersistentClient(path=str(tmp_chroma_dir))
        collections = client.list_collections()
        assert any(c.name == "documents" for c in collections)


class TestResetIntegrationFlow:
    """Integration: populate data → reset → verify both cleaned."""

    def test_reset_after_upload_clears_everything(self, tmp_data_dir, tmp_chroma_dir, sample_txt_file):
        """After upload and index, reset should clear data and chroma."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        # Upload and index
        engine.upload_files([str(sample_txt_file)])
        with patch("urllib.request.urlopen", return_value=mock_response):
            engine.index()

        # Reset
        engine.reset()

        # Data folder should be empty of documents
        assert engine.list_data_files() == []
        # Chroma folder should be deleted
        assert not tmp_chroma_dir.exists()

    def test_can_upload_after_reset(self, tmp_data_dir, tmp_chroma_dir, tmp_path):
        """After reset, new files can be uploaded and indexed."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        # First batch
        first_file = tmp_path / "first.txt"
        first_file.write_text("first document")
        engine.upload_files([str(first_file)])

        # Reset
        engine.reset()

        # Second batch
        second_file = tmp_path / "second.txt"
        second_file.write_text("second document")
        engine.upload_files([str(second_file)])

        files = engine.list_data_files()
        assert files == ["second.txt"]

    def test_reset_preserves_non_document_files(self, tmp_data_dir, tmp_chroma_dir):
        """Reset should only delete .pdf, .doc, .docx, .txt files."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        # Create various files
        (Path(tmp_data_dir) / "notes.txt").write_text("a document")
        (Path(tmp_data_dir) / "config.json").write_text('{"key": "value"}')
        (Path(tmp_data_dir) / "README.md").write_text("# Readme")

        engine.reset()

        remaining = engine.list_data_files()
        # Only document files should be deleted; others remain
        assert "notes.txt" not in remaining
        assert "config.json" in remaining
        assert "README.md" in remaining
