import pytest
from unittest.mock import MagicMock

from app.rag.embedding.document_embedder import DocumentEmbedder


def make_mock_document(content: str, metadata: dict | None = None) -> MagicMock:
    """Create a simple document-like mock with content and metadata."""
    if metadata is None:
        metadata = {}

    mock_doc = MagicMock()
    mock_doc.page_content = content
    mock_doc.metadata = metadata
    return mock_doc


class TestDocumentEmbedder:

    def test_generate_embeddings_assigns_embeddings(self):
        """Test that generate_embeddings stores vectors in each document metadata."""
        strategy = MagicMock()
        strategy.embed_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]

        embedder = DocumentEmbedder(strategy=strategy)
        sections = [make_mock_document("first section"), make_mock_document("second section")]

        result = embedder.generate_embeddings(sections)

        strategy.embed_batch.assert_called_once_with(["first section", "second section"])
        assert result is sections
        assert sections[0].metadata["embedding"] == [0.1, 0.2]
        assert sections[1].metadata["embedding"] == [0.3, 0.4]

    def test_generate_embeddings_returns_empty_list(self):
        """Test that generate_embeddings handles an empty list without failing."""
        strategy = MagicMock()
        strategy.embed_batch.return_value = []

        embedder = DocumentEmbedder(strategy=strategy)
        sections = []

        result = embedder.generate_embeddings(sections)

        strategy.embed_batch.assert_called_once_with([])
        assert result == []

    def test_generate_embeddings_preserves_existing_metadata(self):
        """Test that existing metadata remains intact after embedding generation."""
        strategy = MagicMock()
        strategy.embed_batch.return_value = [[9.9]]

        existing_metadata = {"title": "Section 1"}
        section = make_mock_document("one section", metadata=existing_metadata)
        embedder = DocumentEmbedder(strategy=strategy)

        result = embedder.generate_embeddings([section])

        assert result[0].metadata["title"] == "Section 1"
        assert result[0].metadata["embedding"] == [9.9]
