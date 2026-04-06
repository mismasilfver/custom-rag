import pytest
from pathlib import Path

from rag_engine import RAGEngine
from project_manager import ProjectManager

@pytest.fixture
def tmp_projects_dir(tmp_path):
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    return str(projects_dir)

class TestEngineProjectSwitching:

    def test_engine_initialization_with_project_paths(self, tmp_projects_dir):
        pm = ProjectManager(base_dir=tmp_projects_dir)
        pm.create_project("proj1")
        paths = pm.get_project_paths("proj1")
        
        engine = RAGEngine(
            data_dir=paths["data_dir"],
            chroma_dir=paths["chroma_dir"]
        )
        
        assert engine.data_dir == paths["data_dir"]
        assert engine.chroma_dir == paths["chroma_dir"]
        
        # Test that it respects the paths when interacting with data files
        test_file = Path(paths["data_dir"]) / "test.txt"
        test_file.touch()
        
        files = engine.list_data_files()
        assert "test.txt" in files
        assert len(files) == 1

    def test_engine_isolation_between_projects(self, tmp_projects_dir):
        pm = ProjectManager(base_dir=tmp_projects_dir)
        pm.create_project("proj1")
        pm.create_project("proj2")
        
        paths1 = pm.get_project_paths("proj1")
        paths2 = pm.get_project_paths("proj2")
        
        engine1 = RAGEngine(data_dir=paths1["data_dir"], chroma_dir=paths1["chroma_dir"])
        engine2 = RAGEngine(data_dir=paths2["data_dir"], chroma_dir=paths2["chroma_dir"])
        
        # Add file to proj1
        (Path(paths1["data_dir"]) / "file1.txt").touch()
        
        # Add file to proj2
        (Path(paths2["data_dir"]) / "file2.txt").touch()
        
        # Verify isolation
        files1 = engine1.list_data_files()
        files2 = engine2.list_data_files()
        
        assert "file1.txt" in files1
        assert "file2.txt" not in files1
        
        assert "file2.txt" in files2
        assert "file1.txt" not in files2

    def test_engine_reset_only_affects_current_project(self, tmp_projects_dir):
        pm = ProjectManager(base_dir=tmp_projects_dir)
        pm.create_project("proj1")
        pm.create_project("proj2")
        
        paths1 = pm.get_project_paths("proj1")
        paths2 = pm.get_project_paths("proj2")
        
        engine1 = RAGEngine(data_dir=paths1["data_dir"], chroma_dir=paths1["chroma_dir"])
        engine2 = RAGEngine(data_dir=paths2["data_dir"], chroma_dir=paths2["chroma_dir"])
        
        # Setup files and mock chroma dirs
        (Path(paths1["data_dir"]) / "file1.txt").touch()
        Path(paths1["chroma_dir"]).mkdir(exist_ok=True)
        (Path(paths1["chroma_dir"]) / "db.sqlite3").touch()
        
        (Path(paths2["data_dir"]) / "file2.txt").touch()
        Path(paths2["chroma_dir"]).mkdir(exist_ok=True)
        (Path(paths2["chroma_dir"]) / "db.sqlite3").touch()
        
        # Reset engine 1
        engine1.reset()
        
        # Verify engine 1 is cleared
        assert len(engine1.list_data_files()) == 0
        assert not Path(paths1["chroma_dir"]).exists()
        
        # Verify engine 2 is untouched
        assert len(engine2.list_data_files()) == 1
        assert "file2.txt" in engine2.list_data_files()
        assert Path(paths2["chroma_dir"]).exists()
