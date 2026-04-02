# Custom RAG with Ollama

A local Retrieval-Augmented Generation (RAG) system using LlamaIndex, Ollama, and ChromaDB. This repository enables you to query your documents locally using open-source LLMs without sending data to external APIs.

Notice, you're still stopped by ollama's internal guardrails when trying to query a PDF say about art of psychological manipulation and protecting yourself from manipulation. Something I found out. 😂

## Features

- **Local Document Processing**: Load and index PDF documents from a local directory
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
5. **CLI Arguments**: Added command-line flags for `--reindex` (force reindexing) and `--interactive` (interactive mode)
6. **Ollama Server Check**: Added connection validation to ensure Ollama is running before processing

## Dependencies

| Package | Purpose |
|---------|---------|
| `llama-index-core>=0.14.0` | Core LlamaIndex framework for indexing and querying |
| `llama-index-embeddings-ollama>=0.8.0` | Ollama embedding model integration |
| `llama-index-llms-ollama>=0.10.0` | Ollama LLM integration for text generation |
| `llama-index-vector-stores-chroma>=0.5.0` | ChromaDB vector store connector |
| `chromadb>=0.5.0` | Chroma vector database for embedding storage |
| `pypdf>=4.0` | PDF text extraction library |

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
   Place PDF files in the `data/` directory

## Usage

### Basic Test (Predefined Questions)

```bash
python custom-rag.py
```

### Interactive Mode

```bash
python custom-rag.py --interactive
```

### Force Reindexing (Ignore Cached Embeddings)

```bash
python custom-rag.py --reindex
```

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
├── custom-rag.py          # Main RAG script
├── requirements.txt       # Python dependencies
├── data/                  # Place PDF documents here
├── chroma_db/             # Persistent ChromaDB storage (auto-created)
└── README.md             # This file
```

## License

Based on the tutorial by Aayush Mishra. Modifications and changes by the repository owner.
