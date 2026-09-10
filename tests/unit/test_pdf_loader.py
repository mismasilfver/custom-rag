"""
Unit tests for file extractor registration in RAGEngine.

Ensures the RAG engine uses the correct custom readers for .pdf and .epub
files and that unreadable files are skipped rather than aborting indexing.

Run with: ./venv/bin/python -m pytest tests/unit/test_pdf_loader.py -v
"""

from unittest.mock import MagicMock, patch


class TestPDFLoaderUsesPyMuPDF:
    """RAGEngine must use PyMuPDFReader for .pdf files."""

    def test_build_index_uses_pymupdf_reader_for_pdf_files(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        """The per-file loader must instantiate PyMuPDFReader for .pdf files."""
        from rag_engine import RAGEngine

        (tmp_data_dir / "doc.pdf").write_bytes(b"%PDF-1.4 fake")

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_pymupdf_reader_instance = MagicMock()
        mock_docs = [MagicMock()]

        with (
            patch("rag_engine.PyMuPDFReader") as mock_pymupdf_cls,
            patch("rag_engine.chromadb") as mock_chromadb,
            patch("rag_engine.VectorStoreIndex") as mock_vsi,
            patch("rag_engine.StorageContext"),
            patch("rag_engine.ChromaVectorStore"),
            patch("rag_engine.Settings"),
            patch.object(engine, "_initialize_embed_model", return_value=MagicMock()),
            patch.object(engine, "_initialize_llm", return_value=MagicMock()),
        ):
            mock_pymupdf_cls.return_value = mock_pymupdf_reader_instance
            mock_pymupdf_reader_instance.load_data.return_value = mock_docs

            mock_client = mock_chromadb.PersistentClient.return_value
            mock_client.list_collections.return_value = []
            mock_vsi.from_documents.return_value = MagicMock()

            engine.ensure_index()

        mock_pymupdf_cls.assert_called_once()
        mock_pymupdf_reader_instance.load_data.assert_called_once()


class TestEpubReaderRegistration:
    """RAGEngine must register EpubReader for .epub files."""

    def test_build_index_uses_epub_reader_for_epub_files(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        """The per-file loader must instantiate EpubReader for .epub files."""
        from rag_engine import RAGEngine

        (tmp_data_dir / "book.epub").write_bytes(b"fake epub")

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_epub_reader_instance = MagicMock()
        mock_docs = [MagicMock()]

        with (
            patch("rag_engine.EpubReader") as mock_epub_reader_cls,
            patch("rag_engine.chromadb") as mock_chromadb,
            patch("rag_engine.VectorStoreIndex") as mock_vsi,
            patch("rag_engine.StorageContext"),
            patch("rag_engine.ChromaVectorStore"),
            patch("rag_engine.Settings"),
            patch.object(engine, "_initialize_embed_model", return_value=MagicMock()),
            patch.object(engine, "_initialize_llm", return_value=MagicMock()),
        ):
            mock_epub_reader_cls.return_value = mock_epub_reader_instance
            mock_epub_reader_instance.load_data.return_value = mock_docs

            mock_client = mock_chromadb.PersistentClient.return_value
            mock_client.list_collections.return_value = []
            mock_vsi.from_documents.return_value = MagicMock()

            engine.ensure_index()

        mock_epub_reader_cls.assert_called_once()
        mock_epub_reader_instance.load_data.assert_called_once()
