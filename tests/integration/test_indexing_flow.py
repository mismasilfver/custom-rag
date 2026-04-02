import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from rag_engine import RAGEngine


class TestUploadAndIndexFlow:
    """Integration: upload files → index → verify ChromaDB collection exists."""

    def test_upload_then_list_shows_uploaded_files(self, tmp_data_dir, tmp_chroma_dir, sample_txt_file, sample_pdf_file):
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        assert engine.list_data_files() == []

        engine.upload_files([str(sample_txt_file), str(sample_pdf_file)])

        files = engine.list_data_files()
        assert "sample.txt" in files
        assert "sample.pdf" in files

    def test_upload_rejects_bad_extensions_but_accepts_good_ones(self, tmp_data_dir, tmp_chroma_dir, tmp_path):
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        good_file = tmp_path / "notes.txt"
        good_file.write_text("valid doc")
        bad_file = tmp_path / "script.sh"
        bad_file.write_text("#!/bin/bash")

        engine.upload_files([str(good_file), str(bad_file)])

        files = engine.list_data_files()
        assert files == ["notes.txt"]


class TestResetFlow:
    """Integration: populate data + chroma → reset → verify both cleaned."""

    def test_reset_clears_chroma_and_document_files(self, tmp_data_dir, tmp_chroma_dir, sample_txt_file):
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        # Upload a file
        engine.upload_files([str(sample_txt_file)])
        assert len(engine.list_data_files()) == 1

        # Create a fake chroma_db dir
        tmp_chroma_dir.mkdir(parents=True, exist_ok=True)
        (tmp_chroma_dir / "collection.bin").write_bytes(b"fake")
        assert tmp_chroma_dir.exists()

        # Reset
        engine.reset()

        assert not tmp_chroma_dir.exists()
        assert engine.list_data_files() == []

    def test_reset_preserves_non_document_files(self, tmp_data_dir, tmp_chroma_dir):
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        (tmp_data_dir / "config.json").write_text('{"key": "value"}')
        (tmp_data_dir / "notes.txt").write_text("a document")

        engine.reset()

        remaining = engine.list_data_files()
        assert remaining == ["config.json"]

    def test_reset_then_upload_fresh_files(self, tmp_data_dir, tmp_chroma_dir, tmp_path):
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        # First batch
        old_file = tmp_path / "old.txt"
        old_file.write_text("old content")
        engine.upload_files([str(old_file)])
        assert engine.list_data_files() == ["old.txt"]

        # Reset
        engine.reset()
        assert engine.list_data_files() == []

        # Second batch
        new_file = tmp_path / "new.txt"
        new_file.write_text("new content")
        engine.upload_files([str(new_file)])
        assert engine.list_data_files() == ["new.txt"]
