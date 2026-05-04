import pytest
from unittest.mock import Mock, patch, mock_open
from pyfakefs.fake_filesystem import FakeFilesystem

from app.rag.readers.pdf_reader import PDFReader


class TestPDFReader:
    """Test class for PDFReader methods."""

    def test_is_valid_pdf_valid(self, fs):
        """Test _is_valid_pdf returns True for valid PDF file."""
        fs.create_file('/test.pdf', contents=b'%PDF-1.4')
        reader = PDFReader()
        assert reader._is_valid_pdf('/test.pdf') is True

    def test_is_valid_pdf_invalid_extension(self, fs):
        """Test _is_valid_pdf returns False for non-PDF extension."""
        fs.create_file('/test.txt')
        reader = PDFReader()
        assert reader._is_valid_pdf('/test.txt') is False

    def test_is_valid_pdf_nonexistent(self, fs):
        """Test _is_valid_pdf returns False for nonexistent file."""
        reader = PDFReader()
        assert reader._is_valid_pdf('/nonexistent.pdf') is False

    def test_read_valid_pdf(self):
        """Test read extracts text from valid PDF."""
        reader = PDFReader()

        # Mock fitz
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = [
            [0, 0, 0, 0, "Block1", 0, 0],
            [0, 0, 0, 0, "", 0, 1],  # Non-text block
            [0, 0, 0, 0, "Block2", 0, 0]
        ]
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))

        with patch('app.rag.readers.pdf_reader.fitz.open', return_value=mock_doc):
            result = reader.read('/test.pdf')

            assert result == "Block1\nBlock2"

    def test_read_invalid_pdf_raises_value_error(self):
        """Test read raises ValueError for invalid PDF."""
        reader = PDFReader()
        with pytest.raises(ValueError, match="isn't a valid PDF file"):
            reader.read('/test.txt')

    def test_read_empty_pdf_raises_runtime_error(self):
        """Test read raises RuntimeError for empty PDF."""
        reader = PDFReader()

        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = []  # No blocks
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))

        with patch('app.rag.readers.pdf_reader.fitz.open', return_value=mock_doc):
            with patch('app.rag.readers.pdf_reader.Path') as mock_path:
                mock_path.return_value.is_file.return_value = True
                mock_path.return_value.suffix.lower.return_value = '.pdf'
                with pytest.raises(RuntimeError, match="empty or requires OCR"):
                    reader.read('/empty.pdf')

    def test_read_fitz_exception_raises_runtime_error(self):
        """Test read raises RuntimeError on fitz exception."""
        reader = PDFReader()

        with patch('app.rag.readers.pdf_reader.fitz.open', side_effect=Exception("Fitz error")):
            with patch('app.rag.readers.pdf_reader.Path') as mock_path:
                mock_path.return_value.is_file.return_value = True
                mock_path.return_value.suffix.lower.return_value = '.pdf'
                with pytest.raises(RuntimeError, match="Error processing"):
                    reader.read('/test.pdf')