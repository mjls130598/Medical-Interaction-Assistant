import logging

import requests


class MetadataAPI:
    
    def __init__(self):
        """
        Creates a new Metadata API client to fetch metadata from the AEMPS CIMA database
        """
        self.base_url = "https://cima.aemps.es/cima/rest/medicamento"

    def fetch_metadata(self, cima_id: str) -> dict:
        """
        Fetch metadata from the AEMPS CIMA database using the cima_id

        Arguments:
            **cima_id**: CIMA ID of the medication to fetch metadata for

        Returns:
            **metadata**: Fetched metadata
        """

        try:
            response = requests.get(url=self.base_url, params={"nregistro": cima_id}, timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                "med_id": cima_id,
                "source": f"https://cima.aemps.es/cima/dochtml/p/{cima_id}/",
                "med_name": data.get("nombre"),
                "active_principle": [princ.get("nombre", "") for princ in data.get("principiosActivos", [{}])],
                "last_updated": data.get("docs")[-1].get("fecha"),
                "atcs": data.get("atcs", [])[0].get("nombre", ""),
                "excipients": data.get("excipientes", ""),
                "administrations": data.get("viasAdministracion", [])[0].get("nombre", ""),
                "dosis": data.get("dosis", "")
            }

        except Exception as e:
            logging.error(f"Error fetching metadata for CIMA ID {cima_id}: {e}")
            raise RuntimeError(f"Error fetching metadata for CIMA ID {cima_id}: {e}")