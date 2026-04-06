from pathlib import Path

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
