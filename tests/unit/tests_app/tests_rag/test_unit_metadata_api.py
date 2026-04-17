from unittest.mock import MagicMock, patch

import pytest
import requests

from app.rag.metadata_api import MetadataAPI


class TestMetadataAPI:

    @patch("requests.get")
    def test_fetch_metadata_success(self, mock_get):
        """Verify that the API data is correctly mapped to the internal dictionary."""
        client = MetadataAPI()
        cima_id = "12345"

        # 1. Simular la respuesta JSON de la API CIMA
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "nombre": "Ibuprofeno Fake",
            "principiosActivos": [{"nombre": "Ibuprofeno"}],
            "docs": [{"fecha": "2023-01-01"}, {"fecha": "2024-01-01"}],
            "atcs": [{"nombre": "M01AE01"}],
            "excipientes": "Lactosa",
            "viasAdministracion": [{"nombre": "Oral"}],
            "dosis": "600mg"
        }
        mock_get.return_value = mock_response

        # 2. Ejecutar
        result = client.fetch_metadata(cima_id)

        # 3. Aserciones
        assert result["med_id"] == cima_id
        assert result["med_name"] == "Ibuprofeno Fake"
        assert result["active_principle"] == ["Ibuprofeno"]
        assert result["last_updated"] == "2024-01-01"  # Debe coger el último de la lista
        assert result["atcs"] == "M01AE01"
        assert result["administrations"] == "Oral"
        
        # Verificar que la URL de requests fue la correcta
        mock_get.assert_called_once_with(
            url="https://cima.aemps.es/cima/rest/medicamento", 
            params={"nregistro": cima_id}
        )

    @patch("requests.get")
    def test_fetch_metadata_http_error(self, mock_get):
        """Verify that a 404 or 500 error raises a RuntimeError."""
        client = MetadataAPI()
        
        # Simulate an HTTP error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error")
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError, match="Error fetching metadata"):
            client.fetch_metadata("99999")

    @patch("requests.get")
    def test_fetch_metadata_missing_keys(self, mock_get):
        """Verify that the function handles correctly the absence of expected keys in the JSON response."""
        client = MetadataAPI()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "nombre": "Medicamento Vacío",
            "docs": [{"fecha": "hoy"}],
            "atcs": [{}],  
            "viasAdministracion": [{}] 
        }
        mock_get.return_value = mock_response

        result = client.fetch_metadata("00000")
        assert result["med_name"] == "Medicamento Vacío"
        assert result["atcs"] == "" 