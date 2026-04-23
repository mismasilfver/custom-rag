import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import chromadb
import pymupdf4llm
from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from constants import CITATION_PROMPT_TEMPLATE, SUPPORTED_EXTENSIONS

COLLECTION_NAME = "documents"

logger = logging.getLogger(__name__)


def _clean_text_for_display(text, max_length=200):
    """Clean text content for display by removing binary/non-printable characters.

    Args:
        text: Raw text content from document
        max_length: Maximum length for snippet

    Returns:
        Cleaned text string safe for display
    """
    if not text:
        return ""

    # Replace null bytes and other control characters
    # Keep only printable ASCII and common Unicode characters
    cleaned = "".join(
        char if (ord(char) >= 32 or char in "\n\r\t") and ord(char) < 0x110000 else " "
        for char in text
    )

    # Remove multiple consecutive whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Strip and truncate
    cleaned = cleaned.strip()
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."

    return cleaned


def is_snippet_garbled(text, min_ascii_letter_ratio=0.7):
    """Detect whether a text snippet appears garbled due to font encoding issues.

    Uses the ratio of ASCII letters (a-z, A-Z) to total non-whitespace
    characters as a readability signal. Clean prose is typically >70% ASCII
    letters; font-encoded garbage drops well below that threshold.

    Args:
        text: The snippet string to evaluate.
        min_ascii_letter_ratio: Minimum ratio of ASCII letters required to
            consider the text readable. Default 0.7 (70%).

    Returns:
        True if the snippet appears garbled, False otherwise.
    """
    if not text:
        return False

    non_whitespace = [c for c in text if not c.isspace()]
    if not non_whitespace:
        return False

    ascii_letters = sum(1 for c in non_whitespace if c.isascii() and c.isalpha())
    ratio = ascii_letters / len(non_whitespace)
    return ratio < min_ascii_letter_ratio


def sources_contain_garbled(sources):
    """Return True if any source snippet appears garbled.

    Args:
        sources: List of source dicts as returned by query_with_sources,
            each optionally containing a 'snippet' key.

    Returns:
        True if at least one snippet is garbled, False otherwise.
    """
    return any(is_snippet_garbled(s.get("snippet", "")) for s in sources)


class RAGEngine:
    """Core RAG engine with lazy initialization. No side effects on import."""

    def __init__(
        self,
        data_dir="data",
        chroma_dir="./chroma_db",
        ollama_host="http://localhost:11434",
        model_name="phi4:latest",
        embed_model_name="bge-m3:latest",
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
        self._chat_engine = None
        self._chat_store = None

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
        env = os.environ.copy()
        env["OLLAMA_KEEP_ALIVE"] = "30m"
        self._ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
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
        self._chat_engine = None
        self._chat_store = None
        logger.info(f"Model set to '{model_name}'")

    # ── File management ───────────────────────────────────────────────

    def upload_files(self, file_info):
        """Copy files with supported extensions into the data directory.

        Args:
            file_info: List of tuples (temp_file_path, original_filename) or
                      list of file paths (backward compatibility)
        """
        data_path = Path(self.data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        for item in file_info:
            # Handle both tuple format and legacy path-only format
            if isinstance(item, tuple):
                file_path, original_name = item
            else:
                file_path = item
                original_name = Path(item).name

            src = Path(file_path)
            if src.suffix.lower() not in SUPPORTED_EXTENSIONS:
                logger.warning(f"Skipping unsupported file type: {original_name}")
                continue

            dest = data_path / original_name
            shutil.copy2(str(src), str(dest))
            logger.info(f"Uploaded: {original_name}")

    def list_data_files(self):
        """Return list of file names in the data directory."""
        data_path = Path(self.data_dir)
        if not data_path.exists():
            return []
        return sorted(f.name for f in data_path.iterdir() if f.is_file())

    # ── Indexing ──────────────────────────────────────────────────────

    def _initialize_embed_model(self):
        """Lazy-initialize the embedding model."""
        if self._embed_model is None:
            from llama_index.embeddings.ollama import OllamaEmbedding

            self._embed_model = OllamaEmbedding(
                model_name=self.embed_model_name,
                request_timeout=300.0,
            )
        return self._embed_model

    def _initialize_llm(self):
        """Lazy-initialize the LLM."""
        if self._llm is None:
            from llama_index.llms.ollama import Ollama

            self._llm = Ollama(
                model=self.model_name,
                request_timeout=300.0,
                temperature=0.1,
            )
        return self._llm

    def ensure_index(self):
        """Create index from documents in data directory.

        Skips if index already exists (idempotent).
        """
        return self._build_index(force=False)

    def rebuild_index(self):
        """Force re-creation of index from documents in data directory."""
        return self._build_index(force=True)

    def reindex_with_markdown(self):
        """Convert PDFs in data_dir to Markdown and rebuild the index.

        Uses pymupdf4llm to convert each PDF to Markdown, writes the results
        to a temporary staging directory alongside any non-PDF supported files,
        then rebuilds the index against that staging directory.

        Returns:
            True on success.
        """
        data_path = Path(self.data_dir)
        original_data_dir = self.data_dir

        with tempfile.TemporaryDirectory() as staging_dir:
            staging_path = Path(staging_dir)

            for f in data_path.iterdir():
                if not f.is_file():
                    continue
                if f.suffix.lower() == ".pdf":
                    md_text = pymupdf4llm.to_markdown(str(f))
                    (staging_path / f"{f.stem}.md").write_text(
                        md_text, encoding="utf-8"
                    )
                elif f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    shutil.copy2(str(f), str(staging_path / f.name))

            self.data_dir = staging_dir
            try:
                self.rebuild_index()
            finally:
                self.data_dir = original_data_dir

        return True

    def _build_index(self, force=False):
        """Internal: build or load the vector index."""

        chroma_client = chromadb.PersistentClient(path=self.chroma_dir)

        existing_collections = chroma_client.list_collections()
        collection_exists = any(c.name == COLLECTION_NAME for c in existing_collections)

        if not collection_exists or force:
            data_path = Path(self.data_dir)
            if not data_path.exists():
                raise FileNotFoundError(f"Data directory '{self.data_dir}' not found.")
            supported_files = [
                f
                for f in data_path.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            if not supported_files:
                raise ValueError(
                    f"No supported documents found in '{self.data_dir}'. "
                    "Add files and run with --reindex, or use --project to "
                    "target a different project."
                )

        embed_model = self._initialize_embed_model()
        llm = self._initialize_llm()
        Settings.embed_model = embed_model
        Settings.llm = llm

        if collection_exists and not force:
            logger.info(f"Loading existing ChromaDB collection '{COLLECTION_NAME}'")
            collection = chroma_client.get_collection(COLLECTION_NAME)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=embed_model,
            )
            logger.info("Existing index loaded successfully")
        else:
            logger.info("Loading documents and generating embeddings...")
            docs = SimpleDirectoryReader(
                self.data_dir,
                file_extractor={".pdf": PyMuPDFReader()},
            ).load_data()
            if not docs:
                raise ValueError(f"No documents found in {self.data_dir}")
            collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
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

    def _ensure_index_and_query_engine(self, similarity_top_k=3, qa_prompt=None):
        """Ensure index and query engine are initialized.

        Args:
            similarity_top_k: Number of similar documents to retrieve
            qa_prompt: Optional PromptTemplate for custom query prompts
        """
        if self._index is None:
            self.ensure_index()

        if self._query_engine is None:
            query_engine_kwargs = {
                "llm": self._initialize_llm(),
                "similarity_top_k": similarity_top_k,
                "response_mode": "compact",
            }
            if qa_prompt is not None:
                query_engine_kwargs["text_qa_template"] = qa_prompt
            self._query_engine = self._index.as_query_engine(**query_engine_kwargs)

    def query(self, question, similarity_top_k=3):
        """Query the index and return the response string."""
        self._ensure_index_and_query_engine(similarity_top_k=similarity_top_k)

        logger.info(f"Processing query: '{question}'")
        response = self._query_engine.query(question)
        return str(response)

    def query_with_sources(self, question, similarity_top_k=3):
        """Query the index and return response with source information.

        Returns a dict with:
        - answer: The LLM response string
        - sources: List of source dicts with number, file_name,
          page_label, snippet, score
        """
        qa_prompt = PromptTemplate(CITATION_PROMPT_TEMPLATE)

        self._ensure_index_and_query_engine(
            similarity_top_k=similarity_top_k, qa_prompt=qa_prompt
        )

        logger.info(f"Processing query with sources: '{question}'")
        response = self._query_engine.query(question)

        # Extract source information from response
        sources = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            for i, source_node in enumerate(response.source_nodes, 1):
                node = source_node.node
                metadata = node.metadata

                # Get content snippet and clean it for display
                content = node.get_content()
                snippet = _clean_text_for_display(content, max_length=200)

                source = {
                    "number": i,
                    "file_name": metadata.get("file_name", "Unknown"),
                    "page_label": metadata.get("page_label"),
                    "snippet": snippet,
                    "score": getattr(source_node, "score", None),
                }
                sources.append(source)

        return {
            "answer": str(response),
            "sources": sources,
        }

    # ── Chat ──────────────────────────────────────────────────────────

    def get_chat_engine(self, chat_history_path, system_prompt=None, token_limit=None):
        """Return a ContextChatEngine backed by a persisted SimpleChatStore.

        The instance is cached on ``_chat_engine`` and reused across calls.
        Call ``clear_chat_history`` to reset it.

        Args:
            chat_history_path: Path to the JSON file used to persist chat history.
            system_prompt: Optional system prompt override.
            token_limit: Optional token limit for ChatMemoryBuffer.

        Returns:
            ContextChatEngine instance.
        """
        if self._chat_engine is not None:
            return self._chat_engine

        from constants import CHAT_SYSTEM_PROMPT, CHAT_TOKEN_LIMIT

        if self._index is None:
            self.ensure_index()

        effective_system_prompt = (
            system_prompt if system_prompt is not None else CHAT_SYSTEM_PROMPT
        )
        effective_token_limit = (
            token_limit if token_limit is not None else CHAT_TOKEN_LIMIT
        )

        self._chat_store = SimpleChatStore.from_persist_path(chat_history_path)
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=effective_token_limit,
            chat_store=self._chat_store,
            chat_store_key="default",
        )

        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            llm=self._initialize_llm(),
            system_prompt=effective_system_prompt,
        )
        logger.info(f"Chat engine initialised (token_limit={effective_token_limit})")
        return self._chat_engine

    def chat(self, message, chat_history_path):
        """Send a message to the chat engine and return answer with sources.

        Returns a dict with:
        - answer: The LLM response string
        - sources: List of source dicts with number, file_name, page_label,
          snippet, score
        """
        chat_engine = self.get_chat_engine(chat_history_path)
        logger.info(f"Chat message: '{message}'")
        response = chat_engine.chat(message)

        sources = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            for i, source_node in enumerate(response.source_nodes, 1):
                node = source_node.node
                metadata = node.metadata
                snippet = _clean_text_for_display(node.get_content(), max_length=200)
                sources.append(
                    {
                        "number": i,
                        "file_name": metadata.get("file_name", "Unknown"),
                        "page_label": metadata.get("page_label"),
                        "snippet": snippet,
                        "score": getattr(source_node, "score", None),
                    }
                )

        if self._chat_store is not None:
            self._chat_store.persist(chat_history_path)

        return {"answer": str(response), "sources": sources}

    def load_chat_messages(self, chat_history_path):
        """Load persisted chat messages for display in the UI.

        Reads the SimpleChatStore JSON at ``chat_history_path`` and returns
        a list of dicts suitable for ``st.session_state.messages``.
        Returns an empty list if the file does not exist.

        Args:
            chat_history_path: Path to the persisted chat history JSON file.

        Returns:
            List of dicts with 'role' and 'content' keys.
        """
        if not Path(chat_history_path).exists():
            return []

        store = SimpleChatStore.from_persist_path(chat_history_path)
        return [
            {"role": str(msg.role), "content": msg.content}
            for msg in store.get_messages("default")
        ]

    def clear_chat_history(self, chat_history_path):
        """Reset the chat engine memory and delete the persisted history file.

        Args:
            chat_history_path: Path to the JSON file to delete.
        """
        if self._chat_engine is not None:
            self._chat_engine.reset()
            self._chat_engine = None

        self._chat_store = None

        history_file = Path(chat_history_path)
        if history_file.exists():
            history_file.unlink()
            logger.info(f"Deleted chat history file: {chat_history_path}")

    def export_conversation_to_markdown(
        self, messages, include_sources=False, project_name=None
    ):
        """Export conversation messages as markdown string.

        Args:
            messages: List of message dicts with 'role', 'content', 'timestamp',
                     and optional 'sources' keys.
            include_sources: Whether to include source references in output.
            project_name: Optional project name for the title.

        Returns:
            Markdown formatted string.
        """
        if not messages:
            return ""

        lines = []

        # Title with project name and timestamp
        project = project_name or "export"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"# Chat Export - {project} - {now}")
        lines.append("")
        lines.append("---")
        lines.append("")

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp")

            if role == "user":
                lines.append(f"**User:** {content}")
            elif role == "assistant":
                lines.append(f"**Assistant:** {content}")
            else:
                lines.append(f"**{role.capitalize()}:** {content}")

            lines.append("")

            # Add timestamp if available
            if timestamp:
                if isinstance(timestamp, datetime):
                    time_str = timestamp.strftime("%H:%M")
                else:
                    time_str = str(timestamp)
                lines.append(f"*{time_str}*")
                lines.append("")

            # Add sources if requested and available
            if include_sources and role == "assistant":
                sources = msg.get("sources", [])
                if sources:
                    lines.append("**Sources:**")
                    for src in sources:
                        num = src.get("number", 0)
                        file_name = src.get("file_name", "Unknown")
                        page_label = src.get("page_label")
                        snippet = src.get("snippet", "")

                        if page_label:
                            lines.append(f"- [{num}] {file_name}, Page {page_label}")
                        else:
                            lines.append(f"- [{num}] {file_name}")

                        if snippet:
                            lines.append(f"  > {snippet}")

                    lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

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
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
                ):
                    logger.info(f"Deleting file: {file_path.name}")
                    file_path.unlink()
                    deleted_count += 1
            logger.info(
                f"Deleted {deleted_count} document files from '{self.data_dir}'"
            )

        self._index = None
        self._query_engine = None
        self._chat_engine = None
        self._chat_store = None
        logger.info("Reset complete.")
        return True
