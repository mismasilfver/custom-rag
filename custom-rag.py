import argparse
import logging
import sys

from project_manager import ProjectManager
from rag_engine import RAGEngine, sources_contain_garbled

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logging.root.setLevel(logging.INFO)
logging.root.handlers = [_handler]
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def format_sources_for_display(sources):
    """Format source information for CLI display."""
    if not sources:
        return ""

    lines = ["\n" + "-" * 40, "Sources:"]
    for source in sources:
        page_info = f", Page {source['page_label']}" if source.get("page_label") else ""
        lines.append(f"\n[{source['number']}] {source['file_name']}{page_info}")
        lines.append(f"    {source['snippet'][:100]}...")
    lines.append("-" * 40)
    return "\n".join(lines)


def interactive_query_mode(engine):
    """Run interactive query mode - ask questions until user quits"""
    print("\n" + "=" * 60)
    print("Interactive Query Mode")
    print("=" * 60)
    print("Type your questions below (or 'quit' to exit):")
    print("-" * 60)

    while True:
        try:
            query = input("\nYour question: ").strip()

            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not query:
                continue

            result = engine.query_with_sources(query)
            print(f"\nAnswer: {result['answer']}")

            if result["sources"]:
                print(format_sources_for_display(result["sources"]))
                if sources_contain_garbled(result["sources"]):
                    _prompt_markdown_reindex(engine)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Error: {str(e)}")


def _prompt_markdown_reindex(engine):
    """Ask the user if they want to re-index using Markdown conversion."""
    print("\n⚠️  Some source snippets appear garbled (font encoding issue).")
    try:
        answer = (
            input(
                "Would you like to re-index using PDF→Markdown conversion"
                " for better results? [y/N] "
            )
            .strip()
            .lower()
        )
    except (EOFError, KeyboardInterrupt):
        return
    if answer == "y":
        print("Re-indexing with Markdown conversion...")
        engine.reindex_with_markdown()
        print("Re-indexing complete. Please run your query again.")


def run_test_queries(engine):
    """Run predefined test queries against the RAG system."""
    test_queries = [
        "What is the key finding of this document?",
    ]

    print("\nRAG System Test Results")
    print("=" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 40)

        try:
            result = engine.query_with_sources(query)
            print(f"Response: {result['answer']}")

            if result["sources"]:
                print(format_sources_for_display(result["sources"]))

            print("Status: SUCCESS")
        except Exception as e:
            logger.error(f"Query {i} failed: {e}")
            print(f"Error: {str(e)}")
            print("Status: FAILED")

        print("-" * 40)


def main():
    # Run legacy migration on startup
    pm = ProjectManager()
    if pm.migrate_legacy_data(target_project_name="default"):
        logger.info("Migrated legacy data to 'default' project.")

    parser = argparse.ArgumentParser(description="RAG System with ChromaDB persistence")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete chroma_db and clear data folder documents",
    )
    parser.add_argument(
        "--reindex", action="store_true", help="Force re-creation of embeddings"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive query mode"
    )
    parser.add_argument(
        "--project",
        "-p",
        type=str,
        default="default",
        help="Project to use (creates it if it doesn't exist)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available projects and exit",
    )
    args = parser.parse_args()

    # Handle --list before any other operations
    if args.list:
        projects = pm.list_projects()
        print("\nAvailable projects:")
        print("-" * 40)
        if projects:
            for project in projects:
                marker = " *" if project == "default" else ""
                print(f"  - {project}{marker}")
        else:
            print("  (no projects found)")
        print("-" * 40)
        print("\nUse --project <name> to target a specific project")
        example_project = projects[0] if projects else "myproject"
        print(f"Example: python custom-rag.py --project {example_project} --reindex")
        sys.exit(0)

    # Ensure project exists
    if args.project not in pm.list_projects():
        pm.create_project(args.project)
        logger.info(f"Created new project: '{args.project}'")

    project_paths = pm.get_project_paths(args.project)
    if not project_paths:
        logger.error(f"Failed to get paths for project '{args.project}'")
        sys.exit(1)

    engine = RAGEngine(
        data_dir=project_paths["data_dir"], chroma_dir=project_paths["chroma_dir"]
    )

    logger.info("=" * 60)
    logger.info(f"Starting RAG Pipeline (Project: {args.project})")
    logger.info("=" * 60)

    if args.reset:
        reset_success = engine.reset()
        if reset_success:
            print(
                "\nReset complete. Add new files to the data folder"
                " and run without --reset to re-index."
            )
        sys.exit(0 if reset_success else 1)

    if not engine.check_ollama():
        logger.error(
            "Ollama server is not running. Please start it with 'ollama serve'"
        )
        sys.exit(1)

    try:
        if args.reindex:
            logger.info("--reindex flag set: Will recreate embeddings")
            engine.rebuild_index()
        else:
            engine.ensure_index()

        if args.interactive:
            interactive_query_mode(engine)
        else:
            run_test_queries(engine)

        print("\nRAG system is working correctly!")
        if not args.interactive:
            print("\nTo reset and start fresh with new documents:")
            print(
                f"  ./venv/bin/python custom-rag.py"
                f" --reset --project {args.project}"
            )
            print("\nTo ask questions interactively, run:")
            print(
                f"  ./venv/bin/python custom-rag.py"
                f" --interactive --project {args.project}"
            )
            print("\nTo force re-index (if you add new documents):")
            print(
                f"  ./venv/bin/python custom-rag.py --reindex --project {args.project}"
            )

    except Exception as e:
        logger.error(f"System Error: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"\nRAG system failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
