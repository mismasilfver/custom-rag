"""Unit tests for per-file document loading error handling in RAGEngine."""

from unittest.mock import MagicMock, patch


class TestLoadDocumentsWithErrorHandling:
    """Tests for _load_documents_with_error_handling resilience."""

    def test_skips_corrupt_file_and_loads_good_file(self, tmp_data_dir, tmp_chroma_dir):
        """A corrupt .pdf must be skipped so a good .txt file still indexes."""
        from rag_engine import RAGEngine

        (tmp_data_dir / "bad.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
        (tmp_data_dir / "good.txt").write_text("hello world")

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with (
            patch("rag_engine.PyMuPDFReader") as mock_pdf_cls,
            patch("rag_engine.SimpleDirectoryReader") as mock_sdr_cls,
        ):
            bad_reader = MagicMock()
            bad_reader.load_data.side_effect = Exception("corrupt pdf")
            mock_pdf_cls.return_value = bad_reader

            mock_sdr_instance = MagicMock()
            mock_sdr_instance.load_data.return_value = [MagicMock(text="hello world")]
            mock_sdr_cls.return_value = mock_sdr_instance

            docs, failed_files = engine._load_documents_with_error_handling(
                tmp_data_dir
            )

        assert len(docs) == 1
        assert "bad.pdf" in failed_files

    def test_skips_corrupt_epub_and_loads_remaining_files(
        self, tmp_data_dir, tmp_chroma_dir, sample_txt_file
    ):
        """A corrupt .epub must be skipped so other supported files still load."""
        from rag_engine import RAGEngine

        (tmp_data_dir / "bad.epub").write_bytes(b"not an epub")
        (tmp_data_dir / "good.txt").write_text("hello world")

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        docs, failed_files = engine._load_documents_with_error_handling(tmp_data_dir)

        assert "bad.epub" in failed_files
        # The .txt file is loaded by SimpleDirectoryReader.
        assert len(docs) >= 1

    def test_returns_failed_files_when_all_files_unreadable(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        """If every file fails, the failed list contains all file names."""
        from rag_engine import RAGEngine

        (tmp_data_dir / "bad.pdf").write_bytes(b"%PDF-1.4 fake")

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch("rag_engine.PyMuPDFReader") as mock_pdf_cls:
            bad_reader = MagicMock()
            bad_reader.load_data.side_effect = Exception("corrupt pdf")
            mock_pdf_cls.return_value = bad_reader

            docs, failed_files = engine._load_documents_with_error_handling(
                tmp_data_dir
            )

        assert docs == []
        assert failed_files == ["bad.pdf"]


class TestBuildIndexSkipsCorruptFiles:
    """Tests verifying _build_index still builds when some files fail."""

    def test_build_index_succeeds_when_some_files_are_unreadable(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        """Index creation must succeed if at least one file can be read."""
        from rag_engine import RAGEngine

        (tmp_data_dir / "bad.pdf").write_bytes(b"%PDF-1.4 fake")
        (tmp_data_dir / "good.txt").write_text("hello world")

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_client = MagicMock()
        mock_client.list_collections.return_value = []
        mock_client.get_or_create_collection.return_value = MagicMock()

        with (
            patch("rag_engine.chromadb.PersistentClient", return_value=mock_client),
            patch("rag_engine.PyMuPDFReader") as mock_pdf_cls,
            patch("rag_engine.SimpleDirectoryReader") as mock_sdr_cls,
            patch("rag_engine.ChromaVectorStore"),
            patch("rag_engine.StorageContext"),
            patch("rag_engine.VectorStoreIndex") as mock_vsi,
            patch.object(engine, "_initialize_embed_model", return_value=MagicMock()),
            patch.object(engine, "_initialize_llm", return_value=MagicMock()),
            patch("rag_engine.Settings"),
        ):
            bad_reader = MagicMock()
            bad_reader.load_data.side_effect = Exception("corrupt pdf")
            mock_pdf_cls.return_value = bad_reader

            mock_sdr_instance = MagicMock()
            mock_sdr_instance.load_data.return_value = [MagicMock(text="hello world")]
            mock_sdr_cls.return_value = mock_sdr_instance

            mock_vsi.from_documents.return_value = MagicMock()

            result = engine._build_index(force=True)

        assert result is True
        mock_vsi.from_documents.assert_called_once()
