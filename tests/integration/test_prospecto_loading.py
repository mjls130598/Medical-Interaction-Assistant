import pytest
from langchain_core.documents import Document
from app.rag.loaders.prospecto_loader import ProspectoLoader
from app.rag.readers.pdf_reader import PDFReader
from app.rag.cleaners.prospecto_cleaner import ProspectoCleaner


class TestProspectoLoading:
    """Integration tests for the prospecto loading functionality."""

    @pytest.fixture
    def sample_pdf_path(self):
        """Fixture providing the path to a sample PDF for testing."""
        return "tests/integration/data/test_prospecto.pdf"

    def test_successful_prospecto_loading(self, sample_pdf_path):
        """Test successful loading of a prospecto with valid PDF and CIMA ID."""
        reader = PDFReader()
        cleaner = ProspectoCleaner()
        loader = ProspectoLoader(reader=reader, cleaner=cleaner, source=sample_pdf_path, cima_id="67763")

        documents = loader.create_document()

        assert len(documents) > 0
        assert all('med_id' in doc.metadata for doc in documents)
        assert all('section_title' in doc.metadata for doc in documents)

    def test_invalid_pdf_file(self):
        """Test loading with an invalid PDF file path."""
        reader = PDFReader()
        cleaner = ProspectoCleaner()
        loader = ProspectoLoader(reader=reader, cleaner=cleaner, source="/invalid/path.pdf", cima_id="67763")

        with pytest.raises(ValueError, match="isn't a valid PDF file"):
            loader.create_document()

    def test_api_failure_invalid_cima_id(self, sample_pdf_path):
        """Test handling of API failure with invalid CIMA ID."""
        reader = PDFReader()
        cleaner = ProspectoCleaner()
        loader = ProspectoLoader(reader=reader, cleaner=cleaner, source=sample_pdf_path, cima_id="invalid")

        with pytest.raises(RuntimeError, match="Error fetching metadata"):
            loader.create_document()