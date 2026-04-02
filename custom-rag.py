import argparse
import logging
import sys

from rag_engine import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


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

            if query.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            if not query:
                continue

            response = engine.query(query)
            print(f"\nAnswer: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Error: {str(e)}")


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
            response = engine.query(query)
            print(f"Response: {response}")
            print(f"Status: SUCCESS")
        except Exception as e:
            logger.error(f"Query {i} failed: {e}")
            print(f"Error: {str(e)}")
            print(f"Status: FAILED")

        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="RAG System with ChromaDB persistence")
    parser.add_argument("--reset", action="store_true", help="Delete chroma_db and clear data folder documents")
    parser.add_argument("--reindex", action="store_true", help="Force re-creation of embeddings")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive query mode")
    args = parser.parse_args()

    engine = RAGEngine()

    logger.info("=" * 60)
    logger.info("Starting RAG Pipeline")
    logger.info("=" * 60)

    if args.reset:
        reset_success = engine.reset()
        if reset_success:
            print("\nReset complete. Add new files to the data folder and run without --reset to re-index.")
        sys.exit(0 if reset_success else 1)

    if not engine.check_ollama():
        logger.error("Ollama server is not running. Please start it with 'ollama serve'")
        sys.exit(1)

    try:
        if args.reindex:
            logger.info("--reindex flag set: Will recreate embeddings")
            engine.reindex()
        else:
            engine.index()

        if args.interactive:
            interactive_query_mode(engine)
        else:
            run_test_queries(engine)

        print("\nRAG system is working correctly!")
        if not args.interactive:
            print("\nTo reset and start fresh with new documents:")
            print("  ./venv/bin/python custom-rag.py --reset")
            print("\nTo ask questions interactively, run:")
            print("  ./venv/bin/python custom-rag.py --interactive")
            print("\nTo force re-index (if you add new documents):")
            print("  ./venv/bin/python custom-rag.py --reindex")

    except Exception as e:
        logger.error(f"System Error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"\nRAG system failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()