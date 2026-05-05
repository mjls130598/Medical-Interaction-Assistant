import pytest
import os
from app.rag.readers.pdf_reader import PDFReader


class TestPDFReading:
    """Integration tests for PDF reading functionality."""

    @pytest.fixture
    def reader(self):
        """Fixture providing a PDFReader instance."""
        return PDFReader()

    @pytest.fixture
    def sample_pdf_path(self):
        """Fixture providing the path to a sample PDF."""
        return "tests/integration/data/test_prospecto.pdf"

    def test_successful_pdf_reading(self, reader, sample_pdf_path):
        """Test successful reading of a valid PDF file."""
        content = reader.read(sample_pdf_path)

        assert isinstance(content, str)
        assert len(content) > 0
        # Check for some expected content, but since it's real PDF, just check not empty

    def test_invalid_file_path(self, reader):
        """Test reading with an invalid file path."""
        with pytest.raises(ValueError, match="isn't a valid PDF file"):
            reader.read("/nonexistent/file.pdf")

    def test_non_pdf_file(self, reader, tmp_path):
        """Test reading a file that is not a PDF."""
        # Create a temporary text file
        text_file = tmp_path / "test.txt"
        text_file.write_text("This is not a PDF")

        with pytest.raises(ValueError, match="isn't a valid PDF file"):
            reader.read(str(text_file))