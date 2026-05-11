from unittest.mock import MagicMock, patch

import pytest

from app.service.vector_store.vectore_store_manager import VectorStoreManager


def make_mock_document(document_id: str, content: str, embedding: list[float]) -> MagicMock:
    """Create a document-like mock object with required metadata for storage."""
    mock_doc = MagicMock()
    mock_doc.page_content = content
    mock_doc.metadata = {
        "document_id": document_id,
        "embedding": embedding,
    }
    return mock_doc


class TestVectorStoreManager:

    @patch("app.service.vector_store.vectore_store_manager.chromadb.PersistentClient")
    def test_init_creates_client_and_collection(self, mock_persistent_client, tmp_path):
        """Test that constructor initializes chromadb client and collection."""
        client_instance = MagicMock()
        collection_instance = MagicMock()
        client_instance.get_or_create_collection.return_value = collection_instance
        mock_persistent_client.return_value = client_instance

        manager = VectorStoreManager(path=tmp_path / "store", collection_name="test_collection")

        mock_persistent_client.assert_called_once_with(path=tmp_path / "store")
        client_instance.get_or_create_collection.assert_called_once_with(name="test_collection")
        assert manager.client is client_instance
        assert manager.collection is collection_instance

    @patch("app.service.vector_store.vectore_store_manager.chromadb.PersistentClient")
    def test_save_documents_upserts_documents(self, mock_persistent_client):
        """Test save_documents calls collection.upsert with expected ids, embeddings, metadatas, and documents."""
        client_instance = MagicMock()
        collection_instance = MagicMock()
        client_instance.get_or_create_collection.return_value = collection_instance
        mock_persistent_client.return_value = client_instance

        manager = VectorStoreManager()
        documents = [
            make_mock_document("doc1", "content 1", [0.1, 0.2]),
            make_mock_document("doc2", "content 2", [0.3, 0.4]),
        ]

        expected_metadata = [
            {"document_id": "doc1"},
            {"document_id": "doc2"}
        ]

        manager.save_documents(documents)

        collection_instance.upsert.assert_called_once_with(
            ids=["doc1", "doc2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=expected_metadata,
            documents=["content 1", "content 2"],
        )

    @patch("app.service.vector_store.vectore_store_manager.chromadb.PersistentClient")
    def test_save_documents_handles_empty_document_list(self, mock_persistent_client):
        """Test save_documents supports an empty list by calling upsert with empty collections."""
        client_instance = MagicMock()
        collection_instance = MagicMock()
        client_instance.get_or_create_collection.return_value = collection_instance
        mock_persistent_client.return_value = client_instance

        manager = VectorStoreManager()

        manager.save_documents([])

        collection_instance.upsert.assert_called_once_with(
            ids=[],
            embeddings=[],
            metadatas=[],
            documents=[],
        )

    @patch("app.service.vector_store.vectore_store_manager.chromadb.PersistentClient")
    def test_search_relevant_chunks_queries_collection(self, mock_persistent_client):
        """Test search_relevant_chunks calls collection.query and returns the results."""
        client_instance = MagicMock()
        collection_instance = MagicMock()
        mock_response = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"id": 1}, {"id": 2}]],
            "distances": [[0.1, 0.2]]
        }
        collection_instance.query.return_value = mock_response
        
        client_instance.get_or_create_collection.return_value = collection_instance
        mock_persistent_client.return_value = client_instance

        manager = VectorStoreManager()

        docs, metas = manager.search_relevant_chunks([0.5, 0.6], n_results=3)

        collection_instance.query.assert_called_once_with(
            query_embeddings=[[0.5, 0.6]],
            n_results=3,
        )

        assert docs == ["doc1", "doc2"]
        assert metas == [{"id": 1}, {"id": 2}]

    @patch("app.service.vector_store.vectore_store_manager.chromadb.PersistentClient")
    def test_search_relevant_chunks_default_n_results(self, mock_persistent_client):
        """Test search_relevant_chunks uses the default n_results parameter when not specified."""
        client_instance = MagicMock()
        collection_instance = MagicMock()
        mock_response = {
            "documents": [["doc_default"]],
            "metadatas": [[{"info": "meta_default"}]]
        }
        collection_instance.query.return_value = mock_response
        
        client_instance.get_or_create_collection.return_value = collection_instance
        mock_persistent_client.return_value = client_instance

        manager = VectorStoreManager()

        docs, metas = manager.search_relevant_chunks([1.0, 2.0])

        collection_instance.query.assert_called_once_with(
            query_embeddings=[[1.0, 2.0]],
            n_results=5,
        )
        
        assert docs == ["doc_default"]
        assert metas == [{"info": "meta_default"}]

    @patch("app.service.vector_store.vectore_store_manager.chromadb.PersistentClient")
    def test_is_empty_returns_true_when_no_documents(self, mock_persistent_client):
        """
        Test that is_empty returns True when the collection count is zero.
        """
        client_instance = MagicMock()
        collection_instance = MagicMock()
        collection_instance.count.return_value = 0
        client_instance.get_or_create_collection.return_value = collection_instance
        mock_persistent_client.return_value = client_instance

        manager = VectorStoreManager()
        
        assert manager.is_empty() is True
        collection_instance.count.assert_called_once()

    @patch("app.service.vector_store.vectore_store_manager.chromadb.PersistentClient")
    def test_is_empty_returns_false_when_documents_exist(self, mock_persistent_client):
        """
        Test that is_empty returns False when the collection contains one or more documents.
        """
        client_instance = MagicMock()
        collection_instance = MagicMock()
        collection_instance.count.return_value = 10
        client_instance.get_or_create_collection.return_value = collection_instance
        mock_persistent_client.return_value = client_instance

        manager = VectorStoreManager()
        
        assert manager.is_empty() is False
        collection_instance.count.assert_called_once()