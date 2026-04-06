import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ProjectManager:
    """Manages separate RAG projects on the filesystem."""

    def __init__(self, base_dir="projects"):
        """Initialize ProjectManager with a base directory."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _is_valid_name(self, name: str) -> bool:
        """Check if a project name is valid (alphanumeric, dashes, underscores)."""
        if not name or not name.strip():
            return False
        import re
        return bool(re.match(r'^[\w-]+$', name))

    def create_project(self, project_name: str) -> bool:
        """Create a new project directory structure.
        
        Args:
            project_name: Name of the project to create
            
        Returns:
            bool: True if created successfully, False if invalid name or already exists
        """
        if not self._is_valid_name(project_name):
            logger.warning(f"Invalid project name: '{project_name}'")
            return False
            
        project_dir = self.base_dir / project_name
        
        if project_dir.exists():
            logger.warning(f"Project '{project_name}' already exists.")
            return False
            
        try:
            # Create main project dir
            project_dir.mkdir()
            
            # Create subdirectories
            (project_dir / "data").mkdir()
            (project_dir / "chroma_db").mkdir()
            
            logger.info(f"Created project '{project_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to create project '{project_name}': {e}")
            return False

    def list_projects(self) -> list:
        """List all valid project names."""
        if not self.base_dir.exists():
            return []
            
        projects = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / "data").exists() and (item / "chroma_db").exists():
                projects.append(item.name)
                
        return sorted(projects)

    def delete_project(self, project_name: str) -> bool:
        """Delete a project and all its contents.
        
        Args:
            project_name: Name of the project to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        if project_name not in self.list_projects():
            logger.warning(f"Cannot delete project '{project_name}': does not exist or is invalid")
            return False
            
        project_dir = self.base_dir / project_name
        
        try:
            shutil.rmtree(project_dir)
            logger.info(f"Deleted project '{project_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete project '{project_name}': {e}")
            return False

    def get_project_paths(self, project_name: str) -> dict:
        """Get the paths for a project's data and chroma directories.
        
        Args:
            project_name: Name of the project
            
        Returns:
            dict: Containing 'data_dir' and 'chroma_dir' paths, or None if project invalid
        """
        if project_name not in self.list_projects():
            return None
            
        project_dir = self.base_dir / project_name
        return {
            "data_dir": str(project_dir / "data"),
            "chroma_dir": str(project_dir / "chroma_db")
        }

    def migrate_legacy_data(self, legacy_data_dir="data", legacy_chroma_dir="chroma_db", target_project_name="default") -> bool:
        """Migrate legacy root-level data and chroma_db to a new project.
        
        Args:
            legacy_data_dir: Path to legacy data directory
            legacy_chroma_dir: Path to legacy chroma_db directory
            target_project_name: Name of the project to create for migration
            
        Returns:
            bool: True if migration performed successfully, False if no legacy data found
        """
        data_path = Path(legacy_data_dir)
        chroma_path = Path(legacy_chroma_dir)
        
        has_legacy_data = data_path.exists() and data_path.is_dir()
        has_legacy_chroma = chroma_path.exists() and chroma_path.is_dir()
        
        if not has_legacy_data and not has_legacy_chroma:
            return False
            
        logger.info(f"Found legacy data. Migrating to project '{target_project_name}'...")
        
        # Create the target project
        if target_project_name not in self.list_projects():
            success = self.create_project(target_project_name)
            if not success:
                logger.error(f"Failed to create migration project '{target_project_name}'")
                return False
                
        project_dir = self.base_dir / target_project_name
        target_data = project_dir / "data"
        target_chroma = project_dir / "chroma_db"
        
        try:
            # Move data contents
            if has_legacy_data:
                for item in data_path.iterdir():
                    if item.name == ".gitkeep":
                        continue
                    shutil.move(str(item), str(target_data / item.name))
                shutil.rmtree(data_path)
                
            # Move chroma contents
            if has_legacy_chroma:
                for item in chroma_path.iterdir():
                    shutil.move(str(item), str(target_chroma / item.name))
                shutil.rmtree(chroma_path)
                
            logger.info("Migration successful.")
            return True
        except Exception as e:
            logger.error(f"Error during migration: {e}")
            return False
