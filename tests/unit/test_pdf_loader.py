"""
Unit tests for PDF loading with PyMuPDFReader.

Ensures the RAG engine uses PyMuPDFReader (not the default pypdf backend)
when loading .pdf files, to avoid font-encoding garbled text.

Run with: ./venv/bin/python -m pytest tests/unit/test_pdf_loader.py -v
"""

from unittest.mock import MagicMock, patch


class TestPDFLoaderUsesPyMuPDF:
    """RAGEngine must use PyMuPDFReader as the PDF file extractor."""

    def test_build_index_passes_pymupdf_reader_as_pdf_extractor(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        """SimpleDirectoryReader must receive PyMuPDFReader for .pdf files."""
        from rag_engine import RAGEngine

        (tmp_data_dir / "doc.pdf").write_bytes(b"%PDF-1.4 fake")

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_pymupdf_reader_instance = MagicMock()
        mock_docs = [MagicMock()]

        with (
            patch("rag_engine.PyMuPDFReader") as mock_pymupdf_cls,
            patch("rag_engine.SimpleDirectoryReader") as mock_sdr_cls,
            patch("rag_engine.chromadb") as mock_chromadb,
            patch("rag_engine.VectorStoreIndex") as mock_vsi,
            patch("rag_engine.StorageContext"),
            patch("rag_engine.ChromaVectorStore"),
            patch("rag_engine.Settings"),
            patch.object(engine, "_initialize_embed_model", return_value=MagicMock()),
            patch.object(engine, "_initialize_llm", return_value=MagicMock()),
        ):
            mock_pymupdf_cls.return_value = mock_pymupdf_reader_instance

            mock_sdr_instance = MagicMock()
            mock_sdr_instance.load_data.return_value = mock_docs
            mock_sdr_cls.return_value = mock_sdr_instance

            mock_client = mock_chromadb.PersistentClient.return_value
            mock_client.list_collections.return_value = []
            mock_vsi.from_documents.return_value = MagicMock()

            engine.ensure_index()

        mock_pymupdf_cls.assert_called_once()

        _, kwargs = mock_sdr_cls.call_args
        file_extractor = kwargs.get("file_extractor", {})
        assert (
            ".pdf" in file_extractor
        ), "SimpleDirectoryReader must have a .pdf entry in file_extractor"
        assert (
            file_extractor[".pdf"] is mock_pymupdf_reader_instance
        ), "The .pdf extractor must be a PyMuPDFReader instance"  # noqa: E501
