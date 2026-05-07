import pytest
from unittest.mock import Mock, patch

from app.rag.loaders.prospecto_loader import ProspectoLoader


class TestProspectoLoader:
    """Test class for ProspectoLoader methods."""

    def test_init(self):
        """Test ProspectoLoader initialization."""
        reader = Mock()
        cleaner = Mock()
        source = "test.pdf"
        cima_id = "12345"
        loader = ProspectoLoader(reader, cleaner, source, cima_id)
        assert loader.cima_id == cima_id
        assert loader.source == source

    @patch('app.rag.loaders.prospecto_loader.requests.get')
    def test_get_metadata_success(self, mock_get):
        """Test _get_metadata fetches and parses metadata successfully."""
        reader = Mock()
        cleaner = Mock()
        source = "test.pdf"
        cima_id = "12345"
        loader = ProspectoLoader(reader, cleaner, source, cima_id)

        mock_response = Mock()
        mock_response.json.return_value = {
            "nombre": "Test Med",
            "principiosActivos": [{"nombre": "Active1"}],
            "docs": [{"fecha": "2023-01-01"}],
            "atcs": [{"nombre": "ATC1"}],
            "excipientes": [{"nombre": "Excipient1"}],
            "viasAdministracion": [{"nombre": "Oral"}],
            "dosis": "10mg"
        }
        mock_get.return_value = mock_response

        result = loader._get_metadata()

        # Verify the structure of the returned metadata
        assert result["med_id"] == cima_id
        assert result["source"] == f"https://cima.aemps.es/cima/dochtml/p/{cima_id}/"
        assert result["last_updated"] == "2023-01-01"
        assert result["n_principles"] == 1
        # Fields are cleaned (lowercase, accents removed)
        assert isinstance(result["med_name"], str)
        assert isinstance(result["active_principle"], str)
        mock_get.assert_called_once()

    @patch('app.rag.loaders.prospecto_loader.requests.get')
    def test_get_metadata_request_exception(self, mock_get):
        """Test _get_metadata raises RuntimeError on request failure."""
        reader = Mock()
        cleaner = Mock()
        source = "test.pdf"
        cima_id = "12345"
        loader = ProspectoLoader(reader, cleaner, source, cima_id)

        mock_get.side_effect = Exception("Network error")

        with pytest.raises(RuntimeError, match="Error fetching metadata"):
            loader._get_metadata()

    def test_get_context(self):
        """Test _get_context creates proper context string."""
        reader = Mock()
        cleaner = Mock()
        source = "test.pdf"
        cima_id = "12345"
        loader = ProspectoLoader(reader, cleaner, source, cima_id)

        result = loader._get_context("Aspirin", "Acetylsalicylic acid", "Dosage", "10mg")
        assert "Aspirin" in result
        assert "Acetylsalicylic acid" in result
        assert "Dosage" in result
        assert "10mg" in result

    def test_create_document(self):
        """Test create_document creates documents with metadata and sections."""
        reader = Mock()
        cleaner = Mock()
        source = "test.pdf"
        cima_id = "12345"
        loader = ProspectoLoader(reader, cleaner, source, cima_id)

        metadata = {"med_id": cima_id, "med_name": "Test", "active_principle": "Active1"}
        sections = [
            {"content": "Content1", "section_id": "1", "section_title": "Title1"},
            {"content": "Content2", "section_id": "2", "section_title": "Title2"}
        ]

        with patch.object(loader, '_get_metadata', return_value=metadata):
            with patch.object(loader, '_get_sections', return_value=sections):
                with patch.object(loader, '_get_context', return_value="Test context"):
                    result = loader.create_document()

                    assert len(result) == 2
                    assert result[0].page_content == "Test context"
                    assert 'section_id' in result[0].metadata
                    assert result[0].metadata['section_id'] == '1'