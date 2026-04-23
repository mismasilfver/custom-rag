from unittest.mock import MagicMock, patch


class TestRAGEngineChatEngine:
    """Tests for get_chat_engine, chat, and clear_chat_history."""

    def test_get_chat_engine_returns_chat_engine(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine._index = MagicMock()

        mock_chat_engine = MagicMock()
        mock_memory = MagicMock()

        with patch("rag_engine.ChatMemoryBuffer") as mock_buf_cls:
            mock_buf_cls.from_defaults.return_value = mock_memory
            with patch("rag_engine.SimpleChatStore") as mock_store_cls:
                mock_store_cls.from_persist_path.return_value = MagicMock()
                engine._index.as_chat_engine.return_value = mock_chat_engine

                result = engine.get_chat_engine("path/to/chat.json")

        assert result is mock_chat_engine
        engine._index.as_chat_engine.assert_called_once()

    def test_get_chat_engine_caches_instance(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine._index = MagicMock()
        mock_chat_engine = MagicMock()
        engine._index.as_chat_engine.return_value = mock_chat_engine

        with patch("rag_engine.ChatMemoryBuffer"):
            with patch("rag_engine.SimpleChatStore"):
                engine.get_chat_engine("path/to/chat.json")
                engine.get_chat_engine("path/to/chat.json")

        engine._index.as_chat_engine.assert_called_once()

    def test_get_chat_engine_uses_context_mode(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine._index = MagicMock()
        engine._index.as_chat_engine.return_value = MagicMock()

        with patch("rag_engine.ChatMemoryBuffer"):
            with patch("rag_engine.SimpleChatStore"):
                engine.get_chat_engine("path/to/chat.json")

        call_kwargs = engine._index.as_chat_engine.call_args[1]
        assert call_kwargs.get("chat_mode") == "context"

    def test_chat_returns_answer_and_sources(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine._index = MagicMock()

        mock_source_node = MagicMock()
        mock_source_node.node.metadata = {"file_name": "doc.pdf", "page_label": "1"}
        mock_source_node.node.get_content.return_value = "relevant content"
        mock_source_node.score = 0.9

        mock_response = MagicMock()
        mock_response.__str__ = lambda s: "The answer is 42."
        mock_response.source_nodes = [mock_source_node]

        mock_chat_engine = MagicMock()
        mock_chat_engine.chat.return_value = mock_response

        with patch.object(engine, "get_chat_engine", return_value=mock_chat_engine):
            result = engine.chat("What is the answer?", "path/to/chat.json")

        assert result["answer"] == "The answer is 42."
        assert len(result["sources"]) == 1
        assert result["sources"][0]["file_name"] == "doc.pdf"

    def test_chat_returns_empty_sources_when_none(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine._index = MagicMock()

        mock_response = MagicMock()
        mock_response.__str__ = lambda s: "No sources available."
        mock_response.source_nodes = []

        mock_chat_engine = MagicMock()
        mock_chat_engine.chat.return_value = mock_response

        with patch.object(engine, "get_chat_engine", return_value=mock_chat_engine):
            result = engine.chat("Any question?", "path/to/chat.json")

        assert result["sources"] == []

    def test_clear_chat_history_resets_chat_engine(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        mock_chat_engine = MagicMock()
        engine._chat_engine = mock_chat_engine

        engine.clear_chat_history("path/to/chat.json")

        mock_chat_engine.reset.assert_called_once()
        assert engine._chat_engine is None

    def test_clear_chat_history_deletes_json_file(
        self, tmp_data_dir, tmp_chroma_dir, tmp_path
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        chat_file = tmp_path / "chat_history.json"
        chat_file.write_text('{"messages": []}')

        engine.clear_chat_history(str(chat_file))

        assert not chat_file.exists()

    def test_clear_chat_history_is_safe_when_no_file_exists(
        self, tmp_data_dir, tmp_chroma_dir, tmp_path
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        nonexistent = str(tmp_path / "no_such_file.json")

        engine.clear_chat_history(nonexistent)

    def test_clear_chat_history_is_safe_when_no_chat_engine(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        assert engine._chat_engine is None

        engine.clear_chat_history("path/to/chat.json")

    def test_set_model_clears_cached_chat_engine(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine._chat_engine = MagicMock()

        engine.set_model("mistral:7b")

        assert engine._chat_engine is None

    def test_reset_clears_cached_chat_engine(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine._chat_engine = MagicMock()

        engine.reset()

        assert engine._chat_engine is None

    def test_chat_persists_store_after_each_call(self, tmp_data_dir, tmp_chroma_dir):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine._index = MagicMock()

        mock_response = MagicMock()
        mock_response.__str__ = lambda s: "Answer."
        mock_response.source_nodes = []

        mock_chat_engine = MagicMock()
        mock_chat_engine.chat.return_value = mock_response

        mock_chat_store = MagicMock()

        with patch.object(engine, "get_chat_engine", return_value=mock_chat_engine):
            engine._chat_store = mock_chat_store
            engine.chat("Hello?", "path/to/chat.json")

        mock_chat_store.persist.assert_called_once_with("path/to/chat.json")

    def test_load_chat_messages_returns_messages_from_store(
        self, tmp_data_dir, tmp_chroma_dir, tmp_path
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))

        chat_file = tmp_path / "chat_history.json"
        chat_file.write_text("{}")

        mock_user_msg = MagicMock()
        mock_user_msg.role = "user"
        mock_user_msg.content = "Hello"

        mock_assistant_msg = MagicMock()
        mock_assistant_msg.role = "assistant"
        mock_assistant_msg.content = "Hi there"

        mock_store = MagicMock()
        mock_store.get_messages.return_value = [mock_user_msg, mock_assistant_msg]

        with patch("rag_engine.SimpleChatStore") as mock_store_cls:
            mock_store_cls.from_persist_path.return_value = mock_store
            messages = engine.load_chat_messages(str(chat_file))

        assert messages == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

    def test_load_chat_messages_returns_empty_when_no_file(
        self, tmp_data_dir, tmp_chroma_dir, tmp_path
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        nonexistent = str(tmp_path / "no_history.json")

        messages = engine.load_chat_messages(nonexistent)

        assert messages == []

    def test_clear_chat_history_also_clears_chat_store(
        self, tmp_data_dir, tmp_chroma_dir
    ):
        from rag_engine import RAGEngine

        engine = RAGEngine(data_dir=str(tmp_data_dir), chroma_dir=str(tmp_chroma_dir))
        engine._chat_store = MagicMock()
        engine._chat_engine = MagicMock()

        engine.clear_chat_history("path/to/chat.json")

        assert engine._chat_store is None
