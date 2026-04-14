import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from project_manager import ProjectManager


@pytest.fixture
def tmp_projects_dir(tmp_path):
    """Fixture to provide a temporary projects directory."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    return str(projects_dir)


class TestProjectManager:

    def test_init_creates_base_dir(self, tmp_path):
        base_dir = tmp_path / "custom_projects"
        assert not base_dir.exists()

        ProjectManager(base_dir=str(base_dir))

        assert base_dir.exists()
        assert base_dir.is_dir()

    def test_create_project(self, tmp_projects_dir):
        pm = ProjectManager(base_dir=tmp_projects_dir)

        project_name = "test_project"
        success = pm.create_project(project_name)

        assert success is True

        project_dir = Path(tmp_projects_dir) / project_name
        assert project_dir.exists()
        assert (project_dir / "data").exists()
        assert (project_dir / "chroma_db").exists()

    def test_create_project_invalid_name(self, tmp_projects_dir):
        pm = ProjectManager(base_dir=tmp_projects_dir)

        assert pm.create_project("") is False
        assert pm.create_project("invalid/name") is False
        assert pm.create_project("name.with.dots") is False
        assert pm.create_project("  spaces  ") is False

    def test_create_project_already_exists(self, tmp_projects_dir):
        pm = ProjectManager(base_dir=tmp_projects_dir)

        assert pm.create_project("existing_project") is True
        assert pm.create_project("existing_project") is False

    def test_create_project_from_existing_folder(self, tmp_projects_dir):
        """Test create_project converts existing folder to valid project."""
        pm = ProjectManager(base_dir=tmp_projects_dir)

        # Create a folder with some files (simulating user-created folder)
        existing_folder = Path(tmp_projects_dir) / "existing_folder"
        existing_folder.mkdir()
        (existing_folder / "document.pdf").touch()

        # Should convert the existing folder to a valid project
        success = pm.create_project("existing_folder")
        assert success is True

        # Verify it's now a valid project
        assert "existing_folder" in pm.list_projects()

        # Verify the project structure
        assert (existing_folder / "data").exists()
        assert (existing_folder / "chroma_db").exists()

        # Document files are moved to data/ folder
        assert (existing_folder / "data" / "document.pdf").exists()
        assert not (existing_folder / "document.pdf").exists()

    def test_create_project_from_existing_folder_with_subdirs(self, tmp_projects_dir):
        """Test converting folder that already has some subdirectories."""
        pm = ProjectManager(base_dir=tmp_projects_dir)

        # Create folder with files in a subdirectory
        existing_folder = Path(tmp_projects_dir) / "partial_project"
        existing_folder.mkdir()
        (existing_folder / "random_subdir").mkdir()
        (existing_folder / "random_subdir" / "file.txt").touch()

        success = pm.create_project("partial_project")
        assert success is True

        # Verify project structure created alongside existing content
        assert (existing_folder / "data").exists()
        assert (existing_folder / "chroma_db").exists()
        assert (existing_folder / "random_subdir" / "file.txt").exists()

    def test_create_project_moves_document_files_to_data(self, tmp_projects_dir):
        """Test that document files are moved to data/ when converting folder."""
        pm = ProjectManager(base_dir=tmp_projects_dir)

        # Create folder with document files at root
        existing_folder = Path(tmp_projects_dir) / "docs_project"
        existing_folder.mkdir()
        (existing_folder / "report.pdf").touch()
        (existing_folder / "notes.txt").touch()
        (existing_folder / "unsupported.exe").touch()

        success = pm.create_project("docs_project")
        assert success is True

        # Verify project structure
        assert (existing_folder / "data").exists()
        assert (existing_folder / "chroma_db").exists()

        # Verify document files moved to data/
        assert (existing_folder / "data" / "report.pdf").exists()
        assert (existing_folder / "data" / "notes.txt").exists()

        # Unsupported file should stay at root
        assert (existing_folder / "unsupported.exe").exists()

        # Document files should no longer be at root
        assert not (existing_folder / "report.pdf").exists()
        assert not (existing_folder / "notes.txt").exists()

    def test_list_projects(self, tmp_projects_dir):
        pm = ProjectManager(base_dir=tmp_projects_dir)

        assert pm.list_projects() == []

        pm.create_project("project1")
        pm.create_project("project2")

        # Create a dummy file that should be ignored
        (Path(tmp_projects_dir) / "not_a_project.txt").touch()

        projects = pm.list_projects()
        assert len(projects) == 2
        assert "project1" in projects
        assert "project2" in projects

    def test_delete_project(self, tmp_projects_dir):
        pm = ProjectManager(base_dir=tmp_projects_dir)

        pm.create_project("to_delete")
        assert "to_delete" in pm.list_projects()

        success = pm.delete_project("to_delete")
        assert success is True
        assert "to_delete" not in pm.list_projects()

        project_dir = Path(tmp_projects_dir) / "to_delete"
        assert not project_dir.exists()

    def test_delete_nonexistent_project(self, tmp_projects_dir):
        pm = ProjectManager(base_dir=tmp_projects_dir)
        assert pm.delete_project("does_not_exist") is False

    def test_get_project_paths(self, tmp_projects_dir):
        pm = ProjectManager(base_dir=tmp_projects_dir)
        pm.create_project("my_project")

        paths = pm.get_project_paths("my_project")
        assert paths is not None

        expected_data_dir = str(Path(tmp_projects_dir) / "my_project" / "data")
        expected_chroma_dir = str(Path(tmp_projects_dir) / "my_project" / "chroma_db")

        assert paths["data_dir"] == expected_data_dir
        assert paths["chroma_dir"] == expected_chroma_dir

    def test_get_project_paths_nonexistent(self, tmp_projects_dir):
        pm = ProjectManager(base_dir=tmp_projects_dir)
        assert pm.get_project_paths("does_not_exist") is None

    def test_migrate_legacy_data(self, tmp_path):
        # Setup legacy directories
        legacy_data = tmp_path / "data"
        legacy_chroma = tmp_path / "chroma_db"
        legacy_data.mkdir()
        legacy_chroma.mkdir()

        # Add some dummy files
        (legacy_data / "test.txt").touch()
        (legacy_chroma / "chroma.sqlite3").touch()

        # Setup ProjectManager with a base dir in the same tmp_path
        projects_dir = tmp_path / "projects"
        pm = ProjectManager(base_dir=str(projects_dir))

        # Perform migration
        success = pm.migrate_legacy_data(
            legacy_data_dir=str(legacy_data),
            legacy_chroma_dir=str(legacy_chroma),
            target_project_name="default",
        )

        assert success is True

        # Verify migration
        assert "default" in pm.list_projects()
        paths = pm.get_project_paths("default")

        assert Path(paths["data_dir"], "test.txt").exists()
        assert Path(paths["chroma_dir"], "chroma.sqlite3").exists()

        # Verify legacy dirs are removed
        assert not legacy_data.exists()
        assert not legacy_chroma.exists()

    def test_migrate_legacy_data_no_legacy(self, tmp_projects_dir):
        pm = ProjectManager(base_dir=tmp_projects_dir)

        success = pm.migrate_legacy_data(
            legacy_data_dir="/does/not/exist/data",
            legacy_chroma_dir="/does/not/exist/chroma",
            target_project_name="default",
        )

        assert success is False
        assert "default" not in pm.list_projects()

    def test_create_project_returns_false_on_filesystem_error(self, tmp_projects_dir):
        pm = ProjectManager(base_dir=tmp_projects_dir)

        with patch("pathlib.Path.mkdir", side_effect=OSError("disk full")):
            result = pm.create_project("new_project")

        assert result is False

    def test_list_projects_returns_empty_when_base_dir_removed(self, tmp_projects_dir):
        pm = ProjectManager(base_dir=tmp_projects_dir)
        pm.create_project("proj1")

        shutil.rmtree(tmp_projects_dir)

        assert pm.list_projects() == []

    def test_delete_project_returns_false_on_filesystem_error(self, tmp_projects_dir):
        pm = ProjectManager(base_dir=tmp_projects_dir)
        pm.create_project("to_delete")

        with patch("shutil.rmtree", side_effect=OSError("permission denied")):
            result = pm.delete_project("to_delete")

        assert result is False

    def test_migrate_legacy_data_skips_gitkeep_file(self, tmp_path):
        legacy_data = tmp_path / "data"
        legacy_data.mkdir()
        (legacy_data / ".gitkeep").touch()
        (legacy_data / "real_doc.txt").write_text("content")

        projects_dir = tmp_path / "projects"
        pm = ProjectManager(base_dir=str(projects_dir))

        success = pm.migrate_legacy_data(
            legacy_data_dir=str(legacy_data),
            legacy_chroma_dir="/does/not/exist/chroma",
            target_project_name="default",
        )

        assert success is True
        paths = pm.get_project_paths("default")
        assert Path(paths["data_dir"], "real_doc.txt").exists()
        assert not Path(paths["data_dir"], ".gitkeep").exists()

    def test_migrate_legacy_data_returns_false_when_create_project_fails(
        self, tmp_path
    ):
        legacy_data = tmp_path / "data"
        legacy_data.mkdir()
        (legacy_data / "doc.txt").write_text("content")

        projects_dir = tmp_path / "projects"
        pm = ProjectManager(base_dir=str(projects_dir))

        with patch.object(pm, "create_project", return_value=False):
            success = pm.migrate_legacy_data(
                legacy_data_dir=str(legacy_data),
                legacy_chroma_dir="/does/not/exist/chroma",
                target_project_name="invalid name!",
            )

        assert success is False

    def test_migrate_legacy_data_returns_false_on_move_error(self, tmp_path):
        legacy_data = tmp_path / "data"
        legacy_data.mkdir()
        (legacy_data / "doc.txt").write_text("content")

        projects_dir = tmp_path / "projects"
        pm = ProjectManager(base_dir=str(projects_dir))

        with patch("shutil.move", side_effect=OSError("move failed")):
            success = pm.migrate_legacy_data(
                legacy_data_dir=str(legacy_data),
                legacy_chroma_dir="/does/not/exist/chroma",
                target_project_name="default",
            )

        assert success is False
