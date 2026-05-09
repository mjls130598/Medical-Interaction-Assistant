import pytest
from pathlib import Path

from langchain_core.documents import Document
from app.rag.embedding.document_embedder import DocumentEmbedder
from app.rag.embedding.embedding_strategy.sentence_transformer_strategy import SentenceTransformerStrategy
from app.service.vector_store.vectore_store_manager import VectorStoreManager


class TestEmbeddingAndVectorStore:
    """Integration tests for embedding generation and vector store persistence."""

    def test_generate_embeddings_for_documents(self, embedder, sample_documents):
        """Test full embedding generation for a real batch of documents."""
        embedded_documents = embedder.generate_embeddings(sample_documents)

        assert len(embedded_documents) == len(sample_documents)
        for document in embedded_documents:
            assert "embedding" in document.metadata
            assert isinstance(document.metadata["embedding"], list)
            assert len(document.metadata["embedding"]) > 0
            assert isinstance(document.metadata["embedding"][0], float)
            assert document.metadata["document_id"].startswith("doc_")

    def test_save_documents_to_vector_store(self, embedder, sample_documents, vector_store):
        """Test saving embedded documents to a persistent Chroma vector store."""
        embedded_documents = embedder.generate_embeddings(sample_documents)

        vector_store.save_documents(embedded_documents)

        assert vector_store.collection.count() == len(embedded_documents)

        retrieved = vector_store.collection.get(ids=[doc.metadata["document_id"] for doc in embedded_documents])
        assert len(retrieved["documents"]) == len(embedded_documents)
        assert all(metadata.get("document_id") for metadata in retrieved["metadatas"])

    def test_vector_store_persistence_across_instances(self, embedder, sample_documents, tmp_path):
        """Test that the Chroma store persists data across manager instances."""
        storage_dir = tmp_path / "vector_store"
        first_manager = VectorStoreManager(path=str(storage_dir), collection_name="prospectos_integration")
        embedded_documents = embedder.generate_embeddings(sample_documents)
        first_manager.save_documents(embedded_documents)

        second_manager = VectorStoreManager(path=str(storage_dir), collection_name="prospectos_integration")
        assert second_manager.collection.count() == len(sample_documents)

        retrieved = second_manager.collection.get(ids=[doc.metadata["document_id"] for doc in sample_documents])
        assert len(retrieved["documents"]) == len(sample_documents)

    def test_empty_document_list_embedding(self, embedder):
        """Test embedding generation handles an empty document list."""
        embedded_documents = embedder.generate_embeddings([])
        assert embedded_documents == []

    def test_save_documents_without_embedding_raises(self, vector_store, sample_documents):
        """Test that saving documents without embeddings fails as expected."""
        with pytest.raises(KeyError):
            vector_store.save_documents(sample_documents)
