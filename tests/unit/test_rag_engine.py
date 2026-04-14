import json
from unittest.mock import MagicMock, patch

import pytest


class TestRAGEngineCheckOllama:
    """Tests for checking Ollama server connectivity."""

    def test_check_ollama_returns_true_when_server_is_reachable(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            assert engine.check_ollama() is True

    def test_check_ollama_returns_false_when_server_is_unreachable(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            assert engine.check_ollama() is False


class TestRAGEngineOllamaLifecycle:
    """Tests for starting and stopping Ollama."""

    def test_start_ollama_launches_subprocess(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch("rag_engine.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            # First call returns False (not running), second returns True (started)
            with patch.object(engine, "check_ollama", side_effect=[False, True]):
                with patch("rag_engine.time.sleep"):
                    result = engine.start_ollama()

            assert result is True
            mock_popen.assert_called_once()

    def test_start_ollama_returns_true_if_already_running(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch.object(engine, "check_ollama", return_value=True):
            result = engine.start_ollama()

        assert result is True

    def test_stop_ollama_terminates_process(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        mock_process = MagicMock()
        engine._ollama_process = mock_process

        engine.stop_ollama()

        mock_process.terminate.assert_called_once()

    def test_stop_ollama_does_nothing_when_no_process(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine._ollama_process = None

        engine.stop_ollama()  # should not raise


class TestRAGEngineListModels:
    """Tests for listing available Ollama models."""

    def test_list_models_returns_model_names(
        self, tmp_data_dir, tmp_chroma_dir, ollama_models_response
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(ollama_models_response).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            models = engine.list_models()

        assert models == ["llama3.1:8b", "nomic-embed-text:latest", "mistral:7b"]

    def test_list_models_returns_empty_list_when_ollama_unreachable(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            models = engine.list_models()

        assert models == []


class TestRAGEngineSetModel:
    """Tests for switching the LLM model."""

    def test_set_model_updates_model_name(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine.set_model("mistral:7b")

        assert engine.model_name == "mistral:7b"

    def test_set_model_clears_cached_llm(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine._llm = MagicMock()  # simulate a cached LLM

        engine.set_model("mistral:7b")

        assert engine._llm is None


class TestRAGEngineFileManagement:
    """Tests for uploading and listing files in the data directory."""

    def test_upload_files_copies_to_data_dir(
        self, tmp_data_dir, tmp_chroma_dir, sample_txt_file
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine.upload_files([str(sample_txt_file)])

        uploaded = list(tmp_data_dir.iterdir())
        assert len(uploaded) == 1
        assert uploaded[0].name == "sample.txt"

    def test_upload_files_rejects_unsupported_extension(
        self, tmp_data_dir, tmp_chroma_dir, tmp_path
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        bad_file = tmp_path / "malware.exe"
        bad_file.write_text("not a document")

        engine.upload_files([str(bad_file)])

        assert list(tmp_data_dir.iterdir()) == []

    def test_upload_multiple_files(
        self, tmp_data_dir, tmp_chroma_dir, sample_txt_file, sample_pdf_file
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine.upload_files([str(sample_txt_file), str(sample_pdf_file)])

        uploaded_names = sorted(f.name for f in tmp_data_dir.iterdir())
        assert uploaded_names == ["sample.pdf", "sample.txt"]

    def test_list_data_files_returns_file_names(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        (tmp_data_dir / "doc1.pdf").write_bytes(b"fake")
        (tmp_data_dir / "doc2.txt").write_text("hello")

        files = engine.list_data_files()

        assert sorted(files) == ["doc1.pdf", "doc2.txt"]

    def test_list_data_files_returns_empty_when_no_files(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        files = engine.list_data_files()

        assert files == []


class TestRAGEngineBuildIndexEmptyDataDir:
    """Tests for _build_index when data directory has no files."""

    def test_ensure_index_raises_with_helpful_message_when_data_dir_is_empty(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from unittest.mock import MagicMock, patch

        import pytest

        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_chroma_client = MagicMock()
        mock_chroma_client.list_collections.return_value = []

        with patch("chromadb.PersistentClient", return_value=mock_chroma_client):
            with patch(
                "rag_engine.RAGEngine._initialize_embed_model",
                return_value=MagicMock(),
            ):
                with patch(
                    "rag_engine.RAGEngine._initialize_llm",
                    return_value=MagicMock(),
                ):
                    with pytest.raises(ValueError, match="--project"):
                        engine.ensure_index()


class TestRAGEngineReset:
    """Tests for resetting chroma_db and data folder."""

    def test_reset_deletes_chroma_dir(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        tmp_chroma_dir.mkdir()
        (tmp_chroma_dir / "some_file").write_text("data")

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine.reset()

        assert not tmp_chroma_dir.exists()

    def test_reset_deletes_document_files_from_data_dir(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        (tmp_data_dir / "doc.pdf").write_bytes(b"fake")
        (tmp_data_dir / "notes.txt").write_text("hello")
        (tmp_data_dir / "keep.json").write_text("{}")  # non-document file

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine.reset()

        remaining = [f.name for f in tmp_data_dir.iterdir()]
        assert remaining == ["keep.json"]

    def test_reset_succeeds_when_nothing_exists(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        result = engine.reset()

        assert result is True

    def test_reset_clears_cached_index_and_query_engine(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine._index = MagicMock()
        engine._query_engine = MagicMock()

        engine.reset()

        assert engine._index is None
        assert engine._query_engine is None


class TestRAGEngineStartOllamaTimeout:
    """Tests for start_ollama when Ollama never becomes reachable."""

    def test_start_ollama_returns_false_when_ollama_never_starts(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch("rag_engine.subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 99999
            mock_popen.return_value = mock_process

            with patch.object(engine, "check_ollama", return_value=False):
                with patch("rag_engine.time.sleep"):
                    result = engine.start_ollama()

        assert result is False
        mock_popen.assert_called_once()


class TestRAGEngineListDataFilesMissingDir:
    """Tests for list_data_files when the data directory does not exist."""

    def test_list_data_files_returns_empty_when_dir_does_not_exist(
        self, tmp_chroma_dir, tmp_path
    ):
        from rag_engine import RAGEngine

        nonexistent_dir = str(tmp_path / "no_such_dir")
        engine = RAGEngine(data_dir=nonexistent_dir, chroma_dir=str(tmp_chroma_dir))

        assert engine.list_data_files() == []


class TestRAGEngineBuildIndex:
    """Tests for _build_index internal branching."""

    def _make_mock_chroma_client(self, collection_names):
        mock_client = MagicMock()
        mock_collections = [MagicMock(name=n) for n in collection_names]
        for mc, n in zip(mock_collections, collection_names):
            mc.name = n
        mock_client.list_collections.return_value = mock_collections
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_client.get_or_create_collection.return_value = mock_collection
        return mock_client, mock_collection

    def test_build_index_raises_file_not_found_when_data_dir_missing(
        self, tmp_chroma_dir, tmp_path
    ):
        from rag_engine import RAGEngine

        nonexistent_data = str(tmp_path / "missing_data")
        engine = RAGEngine(data_dir=nonexistent_data, chroma_dir=str(tmp_chroma_dir))

        mock_client, _ = self._make_mock_chroma_client([])

        with patch("chromadb.PersistentClient", return_value=mock_client):
            with pytest.raises(FileNotFoundError):
                engine._build_index(force=False)

    def test_build_index_raises_value_error_when_no_supported_files(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        (tmp_data_dir / "not_a_doc.exe").write_text("binary")
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_client, _ = self._make_mock_chroma_client([])

        with patch("chromadb.PersistentClient", return_value=mock_client):
            with pytest.raises(ValueError, match="No supported documents"):
                engine._build_index(force=False)

    def test_build_index_loads_existing_collection_when_not_forced(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_client, mock_collection = self._make_mock_chroma_client(["documents"])
        mock_vector_store = MagicMock()
        mock_index = MagicMock()

        with patch("chromadb.PersistentClient", return_value=mock_client):
            with patch(
                "llama_index.vector_stores.chroma.ChromaVectorStore",
                return_value=mock_vector_store,
            ):
                with patch("llama_index.core.StorageContext"):
                    with patch("llama_index.core.VectorStoreIndex") as mock_vi:
                        mock_vi.from_vector_store.return_value = mock_index
                        with patch.object(
                            engine,
                            "_initialize_embed_model",
                            return_value=MagicMock(),
                        ):
                            with patch.object(
                                engine,
                                "_initialize_llm",
                                return_value=MagicMock(),
                            ):
                                with patch("llama_index.core.Settings"):
                                    result = engine._build_index(force=False)

        assert result is True
        mock_client.get_collection.assert_called_once_with("documents")
        mock_vi.from_vector_store.assert_called_once()
        assert engine._index is mock_index

    def test_build_index_creates_new_index_when_no_collection(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        (tmp_data_dir / "doc.txt").write_text("hello")
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_client, mock_collection = self._make_mock_chroma_client([])
        mock_vector_store = MagicMock()
        mock_index = MagicMock()
        mock_docs = [MagicMock()]

        with patch("chromadb.PersistentClient", return_value=mock_client):
            with patch(
                "llama_index.vector_stores.chroma.ChromaVectorStore",
                return_value=mock_vector_store,
            ):
                with patch("llama_index.core.StorageContext"):
                    with patch("llama_index.core.VectorStoreIndex") as mock_vi:
                        mock_vi.from_documents.return_value = mock_index
                        with patch(
                            "llama_index.core.SimpleDirectoryReader"
                        ) as mock_sdr:
                            mock_sdr.return_value.load_data.return_value = mock_docs
                            with patch.object(
                                engine,
                                "_initialize_embed_model",
                                return_value=MagicMock(),
                            ):
                                with patch.object(
                                    engine,
                                    "_initialize_llm",
                                    return_value=MagicMock(),
                                ):
                                    with patch("llama_index.core.Settings"):
                                        result = engine._build_index(force=False)

        assert result is True
        mock_vi.from_documents.assert_called_once()
        assert engine._index is mock_index

    def test_build_index_force_true_rebuilds_even_when_collection_exists(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        (tmp_data_dir / "doc.txt").write_text("hello")
        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_client, mock_collection = self._make_mock_chroma_client(["documents"])
        mock_vector_store = MagicMock()
        mock_index = MagicMock()
        mock_docs = [MagicMock()]

        with patch("chromadb.PersistentClient", return_value=mock_client):
            with patch(
                "llama_index.vector_stores.chroma.ChromaVectorStore",
                return_value=mock_vector_store,
            ):
                with patch("llama_index.core.StorageContext"):
                    with patch("llama_index.core.VectorStoreIndex") as mock_vi:
                        mock_vi.from_documents.return_value = mock_index
                        with patch(
                            "llama_index.core.SimpleDirectoryReader"
                        ) as mock_sdr:
                            mock_sdr.return_value.load_data.return_value = mock_docs
                            with patch.object(
                                engine,
                                "_initialize_embed_model",
                                return_value=MagicMock(),
                            ):
                                with patch.object(
                                    engine,
                                    "_initialize_llm",
                                    return_value=MagicMock(),
                                ):
                                    with patch("llama_index.core.Settings"):
                                        result = engine._build_index(force=True)

        assert result is True
        mock_vi.from_documents.assert_called_once()
        mock_client.get_collection.assert_not_called()

    def test_build_index_raises_value_error_when_directory_reader_returns_empty(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        (tmp_data_dir / "doc.txt").write_text("hello")
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        mock_client, _ = self._make_mock_chroma_client([])

        with patch("chromadb.PersistentClient", return_value=mock_client):
            with patch("llama_index.core.SimpleDirectoryReader") as mock_sdr:
                mock_sdr.return_value.load_data.return_value = []
                with patch.object(
                    engine, "_initialize_embed_model", return_value=MagicMock()
                ):
                    with patch.object(
                        engine, "_initialize_llm", return_value=MagicMock()
                    ):
                        with patch("llama_index.core.Settings"):
                            with patch(
                                "llama_index.vector_stores.chroma.ChromaVectorStore"
                            ):
                                with patch("llama_index.core.StorageContext"):
                                    with patch("llama_index.core.VectorStoreIndex"):
                                        with pytest.raises(
                                            ValueError,
                                            match="No documents found",
                                        ):
                                            engine._build_index(force=False)


class TestRAGEngineGetCitationPromptTemplate:
    """Tests for _get_citation_prompt_template."""

    def test_returns_prompt_template_wrapping_citation_constant(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from constants import CITATION_PROMPT_TEMPLATE
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch("llama_index.core.PromptTemplate") as mock_pt_class:
            mock_pt_instance = MagicMock()
            mock_pt_class.return_value = mock_pt_instance

            result = engine._get_citation_prompt_template()

        mock_pt_class.assert_called_once_with(CITATION_PROMPT_TEMPLATE)
        assert result is mock_pt_instance
