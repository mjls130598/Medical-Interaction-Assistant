import pytest
from abc import ABC

from app.rag.readers.document_reader import DocumentReader


class ConcreteDocumentReader(DocumentReader):
    """Concrete implementation of DocumentReader for testing."""
    
    def read(self, source: str) -> str:
        """Simple mock implementation."""
        if source == "valid_source":
            return "Sample document content"
        else:
            raise ValueError(f"Invalid source: {source}")


class TestDocumentReader:
    """Test class for DocumentReader abstract class."""

    def test_document_reader_is_abstract(self):
        """Test that DocumentReader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DocumentReader()

    def test_concrete_implementation_read(self):
        """Test concrete implementation of read method."""
        reader = ConcreteDocumentReader()
        result = reader.read("valid_source")
        assert result == "Sample document content"

    def test_concrete_implementation_read_invalid_source(self):
        """Test concrete implementation raises error for invalid source."""
        reader = ConcreteDocumentReader()
        with pytest.raises(ValueError, match="Invalid source"):
            reader.read("invalid_source")

    def test_read_returns_string(self):
        """Test read method returns a string."""
        reader = ConcreteDocumentReader()
        result = reader.read("valid_source")
        assert isinstance(result, str)
