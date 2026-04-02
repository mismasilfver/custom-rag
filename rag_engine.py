import json
import logging
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt"}


class RAGEngine:
    """Core RAG engine with lazy initialization. No side effects on import."""

    def __init__(
        self,
        data_dir="data",
        chroma_dir="./chroma_db",
        ollama_host="http://localhost:11434",
        model_name="llama3.1:8b",
        embed_model_name="nomic-embed-text:latest",
    ):
        self.data_dir = data_dir
        self.chroma_dir = chroma_dir
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.embed_model_name = embed_model_name

        self._ollama_process = None
        self._llm = None
        self._embed_model = None
        self._index = None
        self._query_engine = None

    # ── Ollama lifecycle ──────────────────────────────────────────────

    def check_ollama(self, timeout=5):
        """Check if Ollama server is reachable."""
        try:
            urllib.request.urlopen(self.ollama_host, timeout=timeout)
            logger.info(f"Ollama server is reachable at {self.ollama_host}")
            return True
        except Exception as e:
            logger.error(f"Cannot connect to Ollama at {self.ollama_host}: {e}")
            return False

    def start_ollama(self):
        """Start Ollama server if not already running."""
        if self.check_ollama():
            logger.info("Ollama is already running")
            return True

        logger.info("Starting Ollama server...")
        self._ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        for _ in range(10):
            time.sleep(1)
            if self.check_ollama():
                logger.info(f"Ollama started (PID {self._ollama_process.pid})")
                return True

        logger.error("Ollama failed to start within timeout")
        return False

    def stop_ollama(self):
        """Stop Ollama server if we started it."""
        if self._ollama_process is None:
            logger.info("No Ollama process to stop")
            return

        logger.info(f"Stopping Ollama (PID {self._ollama_process.pid})...")
        self._ollama_process.terminate()
        self._ollama_process.wait(timeout=10)
        self._ollama_process = None
        logger.info("Ollama stopped")

    # ── Model management ──────────────────────────────────────────────

    def list_models(self):
        """List available Ollama models via the /api/tags endpoint."""
        try:
            url = f"{self.ollama_host}/api/tags"
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def set_model(self, model_name):
        """Switch the LLM model. Clears cached LLM so it reinitializes on next use."""
        self.model_name = model_name
        self._llm = None
        self._query_engine = None
        logger.info(f"Model set to '{model_name}'")

    # ── File management ───────────────────────────────────────────────

    def upload_files(self, file_paths):
        """Copy files with supported extensions into the data directory."""
        data_path = Path(self.data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        for file_path in file_paths:
            src = Path(file_path)
            if src.suffix.lower() not in SUPPORTED_EXTENSIONS:
                logger.warning(f"Skipping unsupported file type: {src.name}")
                continue
            dest = data_path / src.name
            shutil.copy2(str(src), str(dest))
            logger.info(f"Uploaded: {src.name}")

    def list_data_files(self):
        """Return list of file names in the data directory."""
        data_path = Path(self.data_dir)
        if not data_path.exists():
            return []
        return sorted(f.name for f in data_path.iterdir() if f.is_file())

    # ── Indexing ──────────────────────────────────────────────────────

    def _get_embed_model(self):
        """Lazy-initialize the embedding model."""
        if self._embed_model is None:
            from llama_index.embeddings.ollama import OllamaEmbedding

            self._embed_model = OllamaEmbedding(
                model_name=self.embed_model_name,
                request_timeout=300.0,
            )
        return self._embed_model

    def _get_llm(self):
        """Lazy-initialize the LLM."""
        if self._llm is None:
            from llama_index.llms.ollama import Ollama

            self._llm = Ollama(
                model=self.model_name,
                request_timeout=300.0,
                temperature=0.1,
            )
        return self._llm

    def index(self):
        """Create index from documents in data directory. Skips if index already exists."""
        return self._build_index(force=False)

    def reindex(self):
        """Force re-creation of index from documents in data directory."""
        return self._build_index(force=True)

    def _build_index(self, force=False):
        """Internal: build or load the vector index."""
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb

        embed_model = self._get_embed_model()
        llm = self._get_llm()
        Settings.embed_model = embed_model
        Settings.llm = llm

        chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        collection_name = "documents"

        existing_collections = chroma_client.list_collections()
        collection_exists = any(c.name == collection_name for c in existing_collections)

        if collection_exists and not force:
            logger.info(f"Loading existing ChromaDB collection '{collection_name}'")
            collection = chroma_client.get_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=embed_model,
            )
            logger.info("Existing index loaded successfully")
        else:
            data_path = Path(self.data_dir)
            if not data_path.exists():
                raise FileNotFoundError(f"Data directory '{self.data_dir}' not found.")

            logger.info("Loading documents and generating embeddings...")
            docs = SimpleDirectoryReader(self.data_dir).load_data()
            if not docs:
                raise ValueError(f"No documents found in {self.data_dir}")

            collection = chroma_client.get_or_create_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_documents(
                docs,
                storage_context=storage_context,
                embed_model=embed_model,
            )
            logger.info(f"Index created and saved to ChromaDB at '{self.chroma_dir}'")

        self._query_engine = None
        return True

    # ── Querying ──────────────────────────────────────────────────────

    def query(self, question, similarity_top_k=3):
        """Query the index and return the response string."""
        if self._index is None:
            self.index()

        if self._query_engine is None:
            self._query_engine = self._index.as_query_engine(
                llm=self._get_llm(),
                similarity_top_k=similarity_top_k,
                response_mode="compact",
            )

        logger.info(f"Processing query: '{question}'")
        response = self._query_engine.query(question)
        return str(response)

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self):
        """Delete chroma_db folder and remove document files from data folder."""
        chroma_path = Path(self.chroma_dir)
        if chroma_path.exists():
            logger.info(f"Deleting ChromaDB folder: {chroma_path.absolute()}")
            shutil.rmtree(chroma_path)
            logger.info("ChromaDB folder deleted successfully")

        data_path = Path(self.data_dir)
        if data_path.exists():
            deleted_count = 0
            for file_path in data_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    logger.info(f"Deleting file: {file_path.name}")
                    file_path.unlink()
                    deleted_count += 1
            logger.info(f"Deleted {deleted_count} document files from '{self.data_dir}'")

        self._index = None
        self._query_engine = None
        logger.info("Reset complete.")
        return True
