"""
Unit tests for reindex_with_markdown method on RAGEngine.

Run with: ./venv/bin/python -m pytest tests/unit/test_reindex_with_markdown.py -v
"""

from pathlib import Path
from unittest.mock import patch


class TestReindexWithMarkdown:
    """Tests for RAGEngine.reindex_with_markdown."""

    def test_converts_each_pdf_to_markdown(self, tmp_data_dir, tmp_chroma_dir):
        """Each PDF in data_dir must be converted via pymupdf4llm.to_markdown."""
        from rag_engine import RAGEngine

        (tmp_data_dir / "doc1.pdf").write_bytes(b"%PDF-1.4 fake")
        (tmp_data_dir / "doc2.pdf").write_bytes(b"%PDF-1.4 fake")

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with (
            patch("rag_engine.pymupdf4llm") as mock_pymupdf4llm,
            patch.object(engine, "rebuild_index"),
        ):
            mock_pymupdf4llm.to_markdown.return_value = "# Markdown content"
            engine.reindex_with_markdown()

        assert mock_pymupdf4llm.to_markdown.call_count == 2
        called_paths = {
            call_args[0][0] for call_args in mock_pymupdf4llm.to_markdown.call_args_list
        }
        assert any("doc1.pdf" in str(p) for p in called_paths)
        assert any("doc2.pdf" in str(p) for p in called_paths)

    def test_skips_non_pdf_files(self, tmp_data_dir, tmp_chroma_dir):
        """Only .pdf files should be converted; .txt and .docx are passed through."""
        from rag_engine import RAGEngine

        (tmp_data_dir / "notes.txt").write_text("plain text")
        (tmp_data_dir / "report.pdf").write_bytes(b"%PDF-1.4 fake")

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with (
            patch("rag_engine.pymupdf4llm") as mock_pymupdf4llm,
            patch.object(engine, "rebuild_index"),
        ):
            mock_pymupdf4llm.to_markdown.return_value = "# content"
            engine.reindex_with_markdown()

        assert mock_pymupdf4llm.to_markdown.call_count == 1

    def test_writes_markdown_files_to_temp_dir(self, tmp_data_dir, tmp_chroma_dir):
        """Converted markdown must be written to a temporary staging directory."""
        from rag_engine import RAGEngine

        (tmp_data_dir / "doc.pdf").write_bytes(b"%PDF-1.4 fake")

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        staging_snapshots = []

        def capture_rebuild():
            staging_path = Path(engine.data_dir)
            md_files = list(staging_path.glob("*.md"))
            staging_snapshots.append(
                {
                    "dir": staging_path,
                    "md_contents": [f.read_text() for f in md_files],
                }
            )

        with (
            patch("rag_engine.pymupdf4llm") as mock_pymupdf4llm,
            patch.object(engine, "rebuild_index", side_effect=capture_rebuild),
        ):
            mock_pymupdf4llm.to_markdown.return_value = "# content"
            engine.reindex_with_markdown()

        assert len(staging_snapshots) == 1
        snapshot = staging_snapshots[0]
        assert (
            snapshot["dir"] != tmp_data_dir
        ), "rebuild_index must run against a temp dir, not the original data_dir"
        assert snapshot["md_contents"] == ["# content"]

    def test_data_dir_restored_after_reindex(self, tmp_data_dir, tmp_chroma_dir):
        """engine.data_dir must be restored to original after reindex."""
        from rag_engine import RAGEngine

        (tmp_data_dir / "doc.pdf").write_bytes(b"%PDF-1.4 fake")

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        original_data_dir = engine.data_dir

        with (
            patch("rag_engine.pymupdf4llm") as mock_pymupdf4llm,
            patch.object(engine, "rebuild_index"),
        ):
            mock_pymupdf4llm.to_markdown.return_value = "# content"
            engine.reindex_with_markdown()

        assert engine.data_dir == original_data_dir

    def test_returns_true_on_success(self, tmp_data_dir, tmp_chroma_dir):
        """reindex_with_markdown must return True when conversion succeeds."""
        from rag_engine import RAGEngine

        (tmp_data_dir / "doc.pdf").write_bytes(b"%PDF-1.4 fake")

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with (
            patch("rag_engine.pymupdf4llm") as mock_pymupdf4llm,
            patch.object(engine, "rebuild_index"),
        ):
            mock_pymupdf4llm.to_markdown.return_value = "# content"
            result = engine.reindex_with_markdown()

        assert result is True

    def test_non_pdf_files_are_copied_to_temp_dir(self, tmp_data_dir, tmp_chroma_dir):
        """Non-PDF supported files must be copied into the staging dir alongside md."""
        from rag_engine import RAGEngine

        (tmp_data_dir / "notes.txt").write_text("plain text notes")
        (tmp_data_dir / "report.pdf").write_bytes(b"%PDF-1.4 fake")

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with (
            patch("rag_engine.pymupdf4llm") as mock_pymupdf4llm,
            patch.object(engine, "rebuild_index"),
        ):
            mock_pymupdf4llm.to_markdown.return_value = "# converted"
            engine.reindex_with_markdown()

        staging_dir = Path(engine.data_dir)
        assert (staging_dir / "notes.txt").exists()
