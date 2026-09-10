"""Unit tests for the EpubReader custom reader."""

from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import Document

from epub_reader import EpubReader


class TestEpubReader:
    """Tests for EpubReader text extraction from EPUB files."""

    def test_load_data_returns_document(self, sample_epub_file):
        """EpubReader must return a list containing at least one Document."""
        reader = EpubReader()
        docs = reader.load_data(sample_epub_file)

        assert len(docs) >= 1
        assert isinstance(docs[0], Document)

    def test_load_data_extracts_expected_text(self, sample_epub_file):
        """The extracted Document text must include the chapter body content."""
        reader = EpubReader()
        docs = reader.load_data(sample_epub_file)

        combined_text = "\n".join(doc.text for doc in docs)
        assert "This is sample chapter content for EPUB testing." in combined_text

    def test_load_data_preserves_file_name_metadata(self, sample_epub_file):
        """file_name metadata must be preserved when passed via extra_info."""
        reader = EpubReader()
        docs = reader.load_data(
            sample_epub_file, extra_info={"file_name": "sample.epub"}
        )

        for doc in docs:
            assert doc.metadata.get("file_name") == "sample.epub"

    def test_load_data_handles_empty_extra_info(self, sample_epub_file):
        """EpubReader must not fail when extra_info is omitted."""
        reader = EpubReader()
        docs = reader.load_data(sample_epub_file)

        assert len(docs) >= 1
        for doc in docs:
            assert "file_name" in doc.metadata or doc.metadata == {}

    def test_load_data_raises_for_corrupt_epub(self, tmp_path):
        """A corrupt/non-EPUB file must raise so the caller can log and skip it."""
        corrupt_file = tmp_path / "corrupt.epub"
        corrupt_file.write_bytes(b"not an epub")

        reader = EpubReader()
        with pytest.raises(Exception):
            reader.load_data(corrupt_file)

    def test_load_data_returns_empty_list_when_no_document_items(self, tmp_path):
        """An EPUB with no document items must produce an empty document list."""
        empty_epub = tmp_path / "empty.epub"
        empty_epub.write_bytes(b"dummy")

        with patch("ebooklib.epub.read_epub") as mock_read_epub:
            mock_book = MagicMock()
            mock_book.get_items.return_value = []
            mock_read_epub.return_value = mock_book

            reader = EpubReader()
            docs = reader.load_data(empty_epub)

        assert docs == []
