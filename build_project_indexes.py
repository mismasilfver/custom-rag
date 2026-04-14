"""One-shot script: convert raw project folders to valid projects and build ChromaDB
indexes.
"""

import logging
import sys

from project_manager import ProjectManager
from rag_engine import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECTS_BASE = "projects"
PROJECTS = ["AItesting", "Howtodecide", "Softwaretestdesign", "Takingtestingseriously"]


def main():
    pm = ProjectManager(base_dir=PROJECTS_BASE)

    for project_name in PROJECTS:
        logger.info(f"=== Processing project: {project_name} ===")

        converted = pm.create_project(project_name)
        if converted:
            logger.info(f"[{project_name}] Converted/created successfully.")
        else:
            logger.info(
                f"[{project_name}] Already a valid project, skipping conversion."
            )

        paths = pm.get_project_paths(project_name)
        if not paths:
            logger.error(
                f"[{project_name}] Could not retrieve paths — skipping index build."
            )
            continue

        logger.info(
            f"[{project_name}] Building ChromaDB index at '{paths['chroma_dir']}'..."
        )
        engine = RAGEngine(data_dir=paths["data_dir"], chroma_dir=paths["chroma_dir"])
        try:
            engine.ensure_index()
            logger.info(f"[{project_name}] Index built successfully.")
        except Exception as e:
            logger.error(f"[{project_name}] Index build failed: {e}")
            sys.exit(1)

    logger.info("=== All projects processed. ===")


if __name__ == "__main__":
    main()
