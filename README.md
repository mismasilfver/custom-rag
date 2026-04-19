# Custom RAG with Ollama

A local Retrieval-Augmented Generation (RAG) system using LlamaIndex, Ollama, and ChromaDB. This repository enables you to query your documents locally using open-source LLMs without sending data to external APIs.

Notice, you're still stopped by ollama's internal guardrails when trying to query a PDF say about art of psychological manipulation and protecting yourself from manipulation. Something I found out. 😂

## Features

- **Source Citations**: Perplexity-style numbered references showing which documents were used to generate each answer, with expandable source references panel
- **Web-based UI**: Streamlit interface for document management, indexing, and chat
- **Ollama Integration**: Start/stop Ollama server directly from the UI, with model selection
- **File Management**: Upload documents via drag-and-drop, delete with confirmation dialogs
- **Conversation Memory**: Multi-turn context-aware chat powered by LlamaIndex `ContextChatEngine` with `ChatMemoryBuffer`. Chat history is persisted per-project to `chat_history.json` and survives browser reloads. Follow-up questions like *"tell me more"* work naturally — the LLM always sees prior turns
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
5. **CLI Arguments**: Added command-line flags for `--reindex` (force reindexing), `--interactive` (interactive mode), `--reset` (reset data), `--project` (target specific project), and `--list` (show available projects)
6. **Refactored Architecture**: Extracted core logic into `RAGEngine` class with lazy initialization - no side effects on import
7. **Test Suite**: Comprehensive unit and integration tests (125 tests) using pytest
8. **Streamlit Web UI**: Full-featured web interface (`app.py`) with sidebar for Ollama management, file upload, indexing controls, and chat interface
9. **Privacy-First**: Telemetry disabled via `.streamlit/config.toml` - no usage stats sent to external servers
11. **Source Citations**: Added `query_with_sources()` method with Perplexity-style numbered references [1], [2], etc. LLM is prompted to cite sources, and UI displays expandable "Source references" panel with filename, page number (PDFs), and cleaned text snippets
12. **Text Sanitization**: Added `_clean_text_for_display()` helper to remove binary/garbage characters from PDF content extraction, ensuring readable source snippets
13. **Conversation Memory**: Replaced stateless `query_with_sources()` with a `ContextChatEngine` (`chat_mode="context"`) backed by `ChatMemoryBuffer` (token limit 3000) and a `SimpleChatStore` persisted to `<project>/chat_history.json`. Each message is written to disk after the LLM responds, so chat history survives page reloads. `load_chat_messages()` rehydrates the Streamlit display list on startup. History is scoped per-project and cleared on project switch or explicit clear

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
- Markdown (`.md`)

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

## Development Dependencies

For development, install additional dependencies:

```bash
pip install -r requirements-dev.txt
```

This includes:
- `pytest>=8.0` - Testing framework
- `pytest-mock>=3.12` - Mocking support for tests
- `pytest-cov>=4.0` - Code coverage reporting
- `black>=24.0` - Code formatting
- `flake8>=7.0` - Linting
- `isort>=5.12` - Import sorting
- `pre-commit>=3.5` - Pre-commit hooks

### Running Tests with Coverage

The project is configured with pytest-cov for code coverage reporting. Coverage reports are generated automatically when running tests:

```bash
# Run tests with coverage (configured in pyproject.toml)
pytest

# Run tests with specific coverage options
pytest --cov=rag_engine --cov=app --cov-report=term-missing

# Generate HTML coverage report
pytest --cov-report=html:htmlcov

# Open HTML coverage report in browser
open htmlcov/index.html
```

**Coverage Configuration:**
- **Minimum Coverage**: 70% (configured in `pyproject.toml`)
- **Coverage Reports**: Terminal with missing lines, HTML report, XML report
- **Covered Modules**: `rag_engine`, `app`, `project_manager`, `constants`
- **Reports Location**:
  - Terminal: Shows during test run
  - HTML: `htmlcov/index.html`
  - XML: `coverage.xml`

**Current Coverage**: 99.4% across all modules
- `constants.py`: 100% coverage
- `project_manager.py`: 100% coverage
- `rag_engine.py`: 99% coverage

## Usage

### Streamlit Web UI (Recommended)

The easiest way to use the RAG system is through the Streamlit web interface:

```bash
./venv/bin/streamlit run app.py
```

This opens a web UI at `http://localhost:8501` with:
- **Sidebar**: Ollama controls (start/stop, model selection), file upload, document list with delete, reindex button, new project creation
- **Main area**: Chat interface with conversation history

Features in the UI:
- Upload documents via drag-and-drop or file picker
- Delete documents with confirmation dialog
- Reindex documents (auto-indexing on upload, manual reindex available)
- Chat with your documents in a conversational interface
- **View source citations**: Expand the "📚 Source references" panel to see which documents were used for each answer
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

### List Available Projects

View all existing projects:

```bash
python custom-rag.py --list
```

### Target a Specific Project

Run commands against a specific project (creates/ converts the folder if it doesn't exist):

```bash
python custom-rag.py --project myproject --reindex
python custom-rag.py --project laugh --interactive
```

When converting an existing folder with document files, they are automatically moved to the `data/` subdirectory.

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
├── app.py                  # Streamlit web interface
├── custom-rag.py           # Command-line interface
├── rag_engine.py           # Core RAG logic and Ollama integration
├── project_manager.py      # Manages isolated project directories
├── projects/               # Base directory for all projects
│   ├── default/            # The default project
│   │   ├── data/           # PDF/TXT files for this project
│   │   ├── chroma_db/      # Vector database for this project
│   │   └── chat_history.json  # Persisted chat memory (auto-created)
│   └── ...                 # Additional projects
├── tests/                  # Pytest suite (unit & integration)
├── start-rag.sh            # Setup & run script
└── requirements.txt        # Python dependencies (pytest, pytest-mock)
├── .streamlit/
│   └── config.toml        # Streamlit config (telemetry disabled)
├── tests/                 # Test suite
│   ├── conftest.py        # Pytest fixtures
│   ├── unit/
│   │   ├── test_rag_engine.py        # Core RAGEngine tests
│   │   ├── test_rag_sources.py     # Source citations tests
│   │   ├── test_app_upload.py        # Upload handling regression tests
│   │   ├── test_upload_filename.py   # Filename handling tests
│   │   ├── test_project_manager.py   # Project management tests
│   │   └── test_garbled_detection.py # Text sanitization tests
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

- Test the UI with different document types and sizes
- Improve the UI with better error handling and user feedback
- **Chunk size optimization experiments**: Test different chunk sizes and overlap settings to find optimal balance between context preservation and retrieval precision
- **Retrieval tuning differences with similarity_top_k and response_mode**: Experiment with different `similarity_top_k` values (e.g., 3, 5, 10) and response modes (`default`, `compact`, `tree_summarize`, `accumulate`) to optimize answer quality
- **Response mode selector**: User might want to select response mode depending on what they want at that moment from RAG
- ~~**Conversation memory**: Enable multi-turn context-aware conversations using chat history.~~ ✅ Done — uses `ContextChatEngine` with per-project persisted `SimpleChatStore`
- **Export chat history**: Save conversations to JSON or Markdown files
- **Create eval tests** for the RAG system to evaluate answer quality and citation accuracy between models, chunk sizes, and retrieval strategies
- **Fix citation engine** to properly handle citations and sources, maybe experiment with custom citation engine instead of llmaindex built in

Based on the tutorial by Aayush Mishra. Modifications and changes by the repository owner.
