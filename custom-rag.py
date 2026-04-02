from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import logging
import shutil
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def reset_data(data_dir="data", chroma_dir="./chroma_db"):
    """Delete chroma_db folder and remove all pdf, doc, docx, txt files from data folder"""
    logger.info("=" * 60)
    logger.info("Resetting RAG data")
    logger.info("=" * 60)

    # Delete chroma_db folder
    chroma_path = Path(chroma_dir)
    if chroma_path.exists():
        logger.info(f"Deleting ChromaDB folder: {chroma_path.absolute()}")
        shutil.rmtree(chroma_path)
        logger.info("ChromaDB folder deleted successfully")
    else:
        logger.info("ChromaDB folder does not exist, nothing to delete")

    # Delete document files from data folder
    data_path = Path(data_dir)
    if data_path.exists():
        doc_extensions = {".pdf", ".doc", ".docx", ".txt"}
        deleted_count = 0
        for file_path in data_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in doc_extensions:
                logger.info(f"Deleting file: {file_path.name}")
                file_path.unlink()
                deleted_count += 1
        logger.info(f"Deleted {deleted_count} document files from '{data_dir}'")
    else:
        logger.info(f"Data directory '{data_dir}' does not exist, nothing to clean")

    logger.info("Reset complete. You can now add new files and re-index.")
    return True

def check_ollama_connection(host="http://localhost:11434", timeout=5):
    """Check if Ollama server is reachable"""
    import urllib.request
    try:
        urllib.request.urlopen(host, timeout=timeout)
        logger.info(f"Ollama server is reachable at {host}")
        return True
    except Exception as e:
        logger.error(f"Cannot connect to Ollama at {host}: {e}")
        return False

# Check Ollama connection before initializing models
if not check_ollama_connection():
    logger.error("Ollama server is not running. Please start it with 'ollama serve'")
    sys.exit(1)

logger.info("Initializing embedding model...")


# Initialize the embedding model
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    request_timeout=300.0,  # Increased timeout for large documents
)

# Initialize the LLM with optimized settings
llm = Ollama(
    model="llama3.1:8b",  # Confirm with `ollama list`
    request_timeout=300.0,
    temperature=0.1,          # Lower temperature for more factual responses
)

logger.info("LLM initialized successfully")
logger.info("Setting global configurations...")

# Set global configurations
Settings.embed_model = embed_model
Settings.llm = llm

CHROMA_DB_DIR = "./chroma_db"

def get_or_create_index(data_dir="data", force_reindex=False):
    """Get existing index from ChromaDB or create new one if not exists"""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection_name = "documents"

    # Check if collection exists and has data
    existing_collections = chroma_client.list_collections()
    collection_exists = any(c.name == collection_name for c in existing_collections)

    if collection_exists and not force_reindex:
        logger.info(f"Found existing ChromaDB collection '{collection_name}'")
        logger.info("Loading existing index (no embedding generation needed)...")

        # Load existing collection
        collection = chroma_client.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        logger.info("Existing index loaded successfully")
        return index

    # Create new index
    logger.info(f"No existing index found. Creating new index from documents in '{data_dir}'...")

    # Check if data directory exists
    if not Path(data_dir).exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found. Please create it and add your PDF files.")

    logger.info(f"Data directory exists: {Path(data_dir).absolute()}")

    # Load documents
    logger.info("Loading documents...")
    docs = SimpleDirectoryReader(data_dir).load_data()
    logger.info(f"Loaded {len(docs)} documents")

    if not docs:
        raise ValueError(f"No documents found in {data_dir}")

    # Create new collection
    logger.info("Creating new ChromaDB collection and generating embeddings (this may take a while)...")
    collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index (this generates embeddings and stores in ChromaDB)
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model
    )
    logger.info(f"Index created and saved to ChromaDB at '{CHROMA_DB_DIR}'")

    return index

def create_query_engine(index, similarity_top_k=3):
    """Create query engine with specified retrieval parameters"""
    logger.info(f"Creating query engine with similarity_top_k={similarity_top_k}...")

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=similarity_top_k,
        response_mode="compact"
    )

    logger.info("Query engine created successfully")
    return query_engine


def interactive_query_mode(query_engine):
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

            logger.info(f"Processing query: '{query}'")
            response = query_engine.query(query)
            print(f"\nAnswer: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Error: {str(e)}")

def test_rag_system(interactive=False, force_reindex=False):
    """Test the RAG system with sample queries"""
    logger.info("Starting test_rag_system()...")

    try:
        # Get or create index (loads existing if available)
        logger.info("Step 1: Loading or creating index...")
        index = get_or_create_index(force_reindex=force_reindex)
        logger.info("Step 1 complete: Index ready")

        # Create query engine
        logger.info("Step 2: Creating query engine...")
        query_engine = create_query_engine(index)
        logger.info("Step 2 complete: Query engine ready")

        if interactive:
            interactive_query_mode(query_engine)
            return True

        logger.info("Step 3: Running test queries...")
        test_queries = [
            "What is the key finding of this document?",
        ]

        print("\nRAG System Test Results")
        print("=" * 50)

        for i, query in enumerate(test_queries, 1):
            print(f"\nTest {i}: {query}")
            print("-" * 40)

            logger.info(f"Executing query {i}: '{query}'")
            try:
                logger.info("  Sending query to query_engine...")
                response = query_engine.query(query)
                logger.info(f"  Query {i} completed successfully")
                print(f"Response: {response}")
                print(f"Status: SUCCESS")
            except Exception as e:
                logger.error(f"  Query {i} failed: {e}")
                print(f"Error: {str(e)}")
                print(f"Status: FAILED")

            print("-" * 40)

        logger.info("All test queries completed")
        return True

    except Exception as e:
        logger.error(f"System Error in test_rag_system: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"System Error: {str(e)}")
        return False

# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG System with ChromaDB persistence")
    parser.add_argument("--reset", action="store_true", help="Delete chroma_db and clear data folder documents")
    parser.add_argument("--reindex", action="store_true", help="Force re-creation of embeddings")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive query mode")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Starting RAG Pipeline")
    logger.info("=" * 60)

    if args.reset:
        reset_success = reset_data()
        if reset_success:
            print("\nReset complete. Add new files to the data folder and run without --reset to re-index.")
        sys.exit(0 if reset_success else 1)

    if args.reindex:
        logger.info("--reindex flag set: Will recreate embeddings")
    if args.interactive:
        logger.info("--interactive flag set: Will enter interactive mode after setup")

    success = test_rag_system(interactive=args.interactive, force_reindex=args.reindex)

    if success:
        print("\nRAG system is working correctly!")
        if not args.interactive:
            print("\nTo reset and start fresh with new documents:")
            print("  ./venv/bin/python custom-rag.py --reset")
            print("\nTo ask questions interactively, run:")
            print("  ./venv/bin/python custom-rag.py --interactive")
            print("\nTo force re-index (if you add new documents):")
            print("  ./venv/bin/python custom-rag.py --reindex")
    else:
        print("\nRAG system test failed. Check the error messages above.")