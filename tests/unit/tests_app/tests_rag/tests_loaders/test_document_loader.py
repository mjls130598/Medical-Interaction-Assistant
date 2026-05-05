import pytest
from unittest.mock import Mock, patch

from app.rag.loaders.document_loader import DocumentLoader


class MockDocumentLoader(DocumentLoader):
    """Mock implementation of DocumentLoader for testing."""
    def _get_metadata(self):
        return {"test": "metadata"}

    def create_document(self):
        return []


class TestDocumentLoader:
    """Test class for DocumentLoader abstract class."""

    def test_init(self):
        """Test DocumentLoader initialization."""
        reader = Mock()
        cleaner = Mock()
        source = "test.pdf"
        loader = MockDocumentLoader(reader, cleaner, source)
        assert loader.reader == reader
        assert loader.cleaner == cleaner
        assert loader.source == source

    def test_get_sections(self):
        """Test get_sections calls reader and cleaner correctly."""
        reader = Mock()
        cleaner = Mock()
        source = "test.pdf"
        loader = MockDocumentLoader(reader, cleaner, source)

        raw_text = "Raw text content"
        sections = [{"content": "cleaned", "section_id": "1", "section_title": "Title"}]

        reader.read.return_value = raw_text
        cleaner.create_sections.return_value = sections

        result = loader.get_sections()

        reader.read.assert_called_once_with(source)
        cleaner.create_sections.assert_called_once_with(raw_text)
        assert result == sections