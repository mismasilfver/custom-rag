"""Unit tests for filename handling during upload.

These tests verify that uploaded files are saved with their original
filenames when using the tuple format (temp_path, original_name).
"""

from rag_engine import RAGEngine


class TestUploadFilenameWithTuple:
    """Test that upload_files uses original filename from tuple."""

    def test_upload_uses_original_filename_from_tuple(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        """When passing (temp_path, original_name), save as original_name."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        # Create a temp file with any name
        temp_file = tmp_data_dir / ".." / "tmp7cbu_j96_test.pdf"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file.write_text("test content")

        # Upload with tuple format (temp_path, original_name)
        engine.upload_files([(str(temp_file), "my_document.pdf")])

        # File should be saved with the original_name from tuple
        files = engine.list_data_files()
        assert "my_document.pdf" in files
        assert "tmp7cbu_j96_test.pdf" not in files

    def test_upload_tuple_handles_underscores_in_filename(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        """Filenames with underscores handled correctly via tuple."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        temp_file = tmp_data_dir / ".." / "any_temp_name.txt"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file.write_text("test content")

        # Upload with tuple containing underscores in original name
        engine.upload_files([(str(temp_file), "test_file_v2_final.txt")])

        files = engine.list_data_files()
        assert "test_file_v2_final.txt" in files


class TestUploadBackwardCompatibility:
    """Test that upload_files still accepts plain file paths."""

    def test_upload_accepts_plain_path(self, tmp_data_dir, tmp_chroma_dir):
        """Plain file paths should still work (uses path basename)."""
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        # Create a file with a simple name
        simple_file = tmp_data_dir / ".." / "simple_name.txt"
        simple_file.parent.mkdir(parents=True, exist_ok=True)
        simple_file.write_text("test content")

        # Upload with plain path (backward compatibility)
        engine.upload_files([str(simple_file)])

        # Should use the basename from the path
        files = engine.list_data_files()
        assert "simple_name.txt" in files
