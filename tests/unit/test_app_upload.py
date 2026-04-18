"""Unit tests for Streamlit UI upload handling.

These tests verify that file upload logic in app.py behaves correctly
and doesn't cause infinite loops.
"""


class TestFileUploadRerunRegression:
    """Regression test for infinite loop bug in file upload handling."""

    def test_upload_section_does_not_contain_st_rerun(self):
        """Verify the file upload handling does NOT call st.rerun().

        This is a regression test for a bug where st.rerun() after file upload
        caused an infinite loop because st.file_uploader retains its state
        across reruns, causing repeated processing of the same files.

        The fix was to remove the st.rerun() call after successful upload.
        """
        # Read the app.py source code
        with open("app.py", "r") as f:
            source = f.read()

        # Find the render_file_section function
        section_start = source.find("def render_file_section(engine):")
        assert section_start != -1, "render_file_section function not found"

        # Find the next function definition (end of render_file_section)
        next_def = source.find("\ndef ", section_start + 1)
        if next_def == -1:
            section_code = source[section_start:]
        else:
            section_code = source[section_start:next_def]

        # Find the upload handling block - look for the pattern where we process
        # uploaded files and check that st.rerun() is NOT called there
        upload_block_start = section_code.find("if uploaded_files:")
        assert upload_block_start != -1, "uploaded_files handling not found"

        # Get the code from upload block to the end of the function or next block
        remaining_code = section_code[upload_block_start:]

        # Find where the upload block ends (next top-level block or function end)
        # In the fixed code, the upload block is followed by listing current files
        next_block = remaining_code.find("\n    # List current files")
        if next_block == -1:
            upload_block = remaining_code
        else:
            upload_block = remaining_code[:next_block]

        # The key assertion: st.rerun() should NOT be in the upload block
        assert "st.rerun()" not in upload_block, (
            "st.rerun() found in upload handling - this causes infinite loops! "
            "Remove it to fix the bug."
        )

    def test_upload_section_calls_upload_files(self):
        """Verify upload handling calls engine.upload_files()."""
        with open("app.py", "r") as f:
            source = f.read()

        section_start = source.find("def render_file_section(engine):")
        section_end = source.find("\ndef ", section_start + 1)
        if section_end == -1:
            section_code = source[section_start:]
        else:
            section_code = source[section_start:section_end]

        # Verify the upload handling exists and calls engine.upload_files
        assert "if uploaded_files:" in section_code
        assert "engine.upload_files(file_info)" in section_code
