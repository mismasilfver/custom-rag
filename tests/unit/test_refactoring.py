"""Tests for Priority 2 refactoring - renamed methods and structural improvements."""

from unittest.mock import MagicMock, patch


class TestRAGEngineEnsureIndex:
    """Tests for ensure_index() method (renamed from index())."""

    def test_ensure_index_calls_build_index_with_force_false(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch.object(engine, "_build_index") as mock_build:
            engine.ensure_index()
            mock_build.assert_called_once_with(force=False)

    def test_ensure_index_is_idempotent(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch.object(engine, "_build_index", return_value=True) as mock_build:
            engine.ensure_index()
            engine.ensure_index()
            # Always calls, _build_index handles caching logic internally
            assert mock_build.call_count == 2


class TestRAGEngineRebuildIndex:
    """Tests for rebuild_index() method (renamed from reindex())."""

    def test_rebuild_index_calls_build_index_with_force_true(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        with patch.object(engine, "_build_index") as mock_build:
            engine.rebuild_index()
            mock_build.assert_called_once_with(force=True)


class TestRAGEngineInitializeEmbedModel:
    """Tests for _initialize_embed_model() method (renamed from _get_embed_model())."""

    def test_initialize_embed_model_caches_instance(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine._embed_model = None

        # Patch where it's imported in the method (lazy import)
        with patch("llama_index.embeddings.ollama.OllamaEmbedding") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result1 = engine._initialize_embed_model()
            result2 = engine._initialize_embed_model()

            assert result1 is mock_instance
            assert result1 is result2  # Same cached instance
            mock_class.assert_called_once()  # Only initialized once


class TestRAGEngineInitializeLLM:
    """Tests for _initialize_llm() method (renamed from _get_llm())."""

    def test_initialize_llm_caches_instance(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine._llm = None

        # Patch where it's imported in the method (lazy import)
        with patch("llama_index.llms.ollama.Ollama") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result1 = engine._initialize_llm()
            result2 = engine._initialize_llm()

            assert result1 is mock_instance
            assert result1 is result2  # Same cached instance
            mock_class.assert_called_once()  # Only initialized once

    def test_initialize_llm_uses_current_model_name(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(
            data_dir=str(tmp_data_dir),
            chroma_dir=str(tmp_chroma_dir),
            model_name="test-model:latest",
        )
        engine._llm = None

        # Patch where it's imported in the method (lazy import)
        with patch("llama_index.llms.ollama.Ollama") as mock_class:
            engine._initialize_llm()

            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args.kwargs
            assert call_kwargs["model"] == "test-model:latest"
