import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.rag_engine import RAGManager


class TestRAGManager:
    """Test class for the RAGManager core orchestration class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the RAGManager singleton state before each test."""
        RAGManager._instance = None
        yield

    @patch("app.core.rag_engine.GroqRAGAssistance")
    @patch("app.core.rag_engine.SentenceTransformerStrategy")
    @patch("app.core.rag_engine.VectorStoreManager")
    @patch("app.core.rag_engine.Path.glob", return_value=[])
    def test_initialize_skips_loading_when_store_has_documents(self, mock_glob, mock_vector_store_manager, mock_strategy, mock_assistance):
        """Test that initialization skips document loading when vector store is already populated."""
        mock_store = MagicMock()
        mock_store.is_empty.return_value = False
        mock_vector_store_manager.return_value = mock_store
        mock_strategy.return_value = MagicMock()
        mock_assistance.return_value = MagicMock()

        manager = RAGManager()
        manager.initialize()

        mock_store.is_empty.assert_called_once()
        mock_glob.assert_not_called()
        mock_assistance.assert_called_once_with(
            vector_store=mock_store,
            embedding_strategy=mock_strategy.return_value,
            db_connection=None
        )
        assert manager._initialized is True

    @patch("app.core.rag_engine.GroqRAGAssistance")
    @patch("app.core.rag_engine.SentenceTransformerStrategy")
    @patch("app.core.rag_engine.VectorStoreManager")
    @patch("app.core.rag_engine.Path.glob", return_value=[])
    def test_initialize_loads_documents_when_empty_but_no_pdfs(self, mock_glob, mock_vector_store_manager, mock_strategy, mock_assistance):
        """Test initialization when the vector store is empty and no PDF files exist."""
        mock_store = MagicMock()
        mock_store.is_empty.return_value = True
        mock_vector_store_manager.return_value = mock_store
        mock_strategy.return_value = MagicMock()
        mock_assistance.return_value = MagicMock()

        manager = RAGManager()
        manager.initialize()

        mock_store.is_empty.assert_called_once()
        mock_glob.assert_called_once()
        mock_store.save_documents.assert_not_called()
        mock_assistance.assert_called_once()

    def test_get_response_raises_when_not_initialized(self):
        """Test that get_response raises when RAGManager has not been initialized."""
        manager = RAGManager()

        with pytest.raises(Exception, match="RAGManager is not initialized"):
            asyncio.run(manager.get_response("query", "session_1"))

    def test_get_response_awaits_assistance_ask(self):
        """Test that get_response invokes the assistance ask method asynchronously."""
        mock_assistance = MagicMock()
        mock_assistance.ask = AsyncMock(return_value="answer")

        manager = RAGManager()
        manager._initialized = True
        manager.assistance = mock_assistance

        result = asyncio.run(manager.get_response("query", "session_1"))

        mock_assistance.ask.assert_awaited_once_with(query="query", session_id="session_1")
        assert result == "answer"
