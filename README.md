# Custom RAG with Ollama

A local Retrieval-Augmented Generation (RAG) system using LlamaIndex, Ollama, and ChromaDB. This repository enables you to query your documents locally using open-source LLMs without sending data to external APIs.

Notice, you're still stopped by ollama's internal guardrails when trying to query a PDF say about art of psychological manipulation and protecting yourself from manipulation. Something I found out. 😂

## Features

- **Web-based UI**: Streamlit interface for document management, indexing, and chat
- **Ollama Integration**: Start/stop Ollama server directly from the UI, with model selection
- **File Management**: Upload documents via drag-and-drop, delete with confirmation dialogs
- **Chat Interface**: Conversational Q&A with conversation history and clear chat option
- **Local Document Processing**: Load and index PDF, Word, and text documents from a local directory
- **Persistent Embeddings**: ChromaDB stores embeddings locally, avoiding reprocessing on subsequent runs
- **Interactive Query Mode**: Ask questions interactively or run predefined test queries
- **Comprehensive Logging**: Detailed logging for debugging and observability
- **Memory Efficient**: Designed to work with consumer hardware (tested on MacBook with 36GB RAM)

## Original Source

This project is based on the tutorial by **Aayush Mishra**:
- Blog Post: [Setting up RAG locally with Ollama - A Beginner-Friendly Guide](https://dev.to/the_aayush_mishra/setting-up-rag-locally-with-ollama-a-beginner-friendly-guide-428m)

## Changes from Original Code

The following changes have been made to the original tutorial code:

1. **Enhanced Logging for Observability**: Added comprehensive logging throughout the pipeline to track document loading, embedding generation, query processing, and responses because the embedding creation was taking so long time
2. **Dependency Version Pinning**: Pinned all dependencies to compatible versions to avoid API conflicts (specifically resolved `llama-index-workflows` 2.x breaking changes). Now using latest llama-index 0.14.x with Python 3.14.
3. **Interactive Query Mode**: Added CLI flag for interactive mode allowing users to ask questions in a loop
4. **ChromaDB Persistence**: Implemented persistent storage of embeddings to `./chroma_db`, preventing regeneration on every run
5. **CLI Arguments**: Added command-line flags for `--reindex` (force reindexing), `--interactive` (interactive mode), and `--reset` (reset data)
6. **Refactored Architecture**: Extracted core logic into `RAGEngine` class with lazy initialization - no side effects on import
7. **Test Suite**: Comprehensive unit and integration tests (46 tests) using pytest
8. **Streamlit Web UI**: Full-featured web interface (`app.py`) with sidebar for Ollama management, file upload, indexing controls, and chat interface
9. **Privacy-First**: Telemetry disabled via `.streamlit/config.toml` - no usage stats sent to external servers
10. **Safety Features**: Delete confirmation dialogs to prevent accidental data loss

## Dependencies

| Package | Purpose |
|---------|---------|
| `llama-index-core>=0.14.0` | Core LlamaIndex framework for indexing and querying |
| `llama-index-embeddings-ollama>=0.8.0` | Ollama embedding model integration |
| `llama-index-llms-ollama>=0.10.0` | Ollama LLM integration for text generation |
| `llama-index-vector-stores-chroma>=0.5.0` | ChromaDB vector store connector |
| `chromadb>=0.5.0` | Chroma vector database for embedding storage |
| `pypdf>=4.0` | PDF text extraction library |
| `streamlit>=1.30.0` | Web UI framework for interactive interface |

## Supported Document Types

The RAG system supports the following document formats:
- PDF (`.pdf`)
- Microsoft Word (`.doc`, `.docx`)
- Plain text (`.txt`)

## Installation

### Prerequisites

- Python 3.14 (or 3.10+)
- [Ollama](https://ollama.com/) installed and running locally
- The `llama3.1:8b` model pulled in Ollama (`ollama pull llama3.1:8b`)

> **Note:** Python 3.14+ is required for the latest llama-index packages (0.14.x). The project was tested with Python 3.14.

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mismasilfver/custom-rag.git
   cd custom-rag
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your documents**:
   Place PDF, Word, or text files in the `data/` directory

## Usage

### Streamlit Web UI (Recommended)

The easiest way to use the RAG system is through the Streamlit web interface:

```bash
./venv/bin/streamlit run app.py
```

This opens a web UI at `http://localhost:8501` with:
- **Sidebar**: Ollama controls (start/stop, model selection), file upload, document list with delete, indexing buttons
- **Main area**: Chat interface with conversation history

Features in the UI:
- Upload documents via drag-and-drop or file picker
- Delete documents with confirmation dialog
- Index or reindex documents with visual feedback
- Chat with your documents in a conversational interface
- Clear chat history anytime

### CLI Mode

For command-line usage:

#### Basic Test (Predefined Questions)

```bash
python custom-rag.py
```

#### Interactive Mode

```bash
python custom-rag.py --interactive
```

### Force Reindexing (Ignore Cached Embeddings)

```bash
python custom-rag.py --reindex
```

### Reset and Start Fresh

Delete the ChromaDB vector database and clear all document files from the data folder. This is useful when you want to start with a completely new set of documents.

```bash
python custom-rag.py --reset
```

After resetting, add your new documents to the `data/` folder and run normally to create a fresh vector database.

### Combined (Interactive with Reindex)

```bash
python custom-rag.py --interactive --reindex
```

## Performance Notes

- **Model**: `llama3.1:8b` works well and responds quickly after embeddings have been created
- **Hardware**: Tested on a MacBook with 36GB RAM - the system operates near memory limits but remains stable
- **Embeddings**: First run takes longer as embeddings are generated; subsequent runs are fast since embeddings are cached in ChromaDB
- **Memory Usage**: Initial indexing can spike memory usage; ensure no other heavy applications are running during "questioning" the document(s)

## Project Structure

```
custom-rag/
├── app.py                 # Streamlit web interface
├── rag_engine.py          # Core RAGEngine class (lazy init, all RAG logic)
├── custom-rag.py          # CLI wrapper around RAGEngine
├── requirements.txt       # Python dependencies
├── requirements-dev.txt   # Development dependencies (pytest, pytest-mock)
├── .streamlit/
│   └── config.toml        # Streamlit config (telemetry disabled)
├── tests/                 # Test suite
│   ├── conftest.py        # Pytest fixtures
│   ├── unit/
│   │   └── test_rag_engine.py
│   └── integration/
│       ├── test_app_ollama.py
│       ├── test_chat_flow.py
│       ├── test_indexing_flow.py
│       ├── test_ollama_lifecycle.py
│       └── test_upload_index.py
├── data/                  # Place PDF, Word, and text documents here
├── chroma_db/             # Persistent ChromaDB storage (auto-created)
└── README.md             # This file
```

## Todo and Experiment

Potential improvements and experiments to explore:

- ✅ **Create an interactive UI for the RAG**: ~~Build a web-based or desktop GUI interface to make the RAG system more accessible to non-technical users~~ **DONE** - Streamlit UI implemented with file management, indexing, and chat
- Test the UI with different document types and sizes
- Improve the UI with better error handling and user feedback
- Fix bugs in the UI
- **Chunk size optimization experiments**: Test different chunk sizes and overlap settings to find optimal balance between context preservation and retrieval precision
- **Retrieval tuning differences with similarity_top_k and response_mode**: Experiment with different `similarity_top_k` values (e.g., 3, 5, 10) and response modes (`compact`, `tree_summarize`, `accumulate`) to optimize answer quality
- **Source citations in chat**: Show which document chunks were used to generate each answer
- **Conversation memory**: Enable multi-turn context-aware conversations using chat history
- **Multiple collection support**: Allow organizing documents into separate collections/projects
- **Export chat history**: Save conversations to JSON or Markdown files

Based on the tutorial by Aayush Mishra. Modifications and changes by the repository owner.
