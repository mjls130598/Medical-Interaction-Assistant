import logging

import requests

from rag.document_loader import DocumentLoader
from rag.document_reader import DocumentReader


class ProspectoLoader(DocumentLoader):
    def __init__(self, reader: DocumentReader, source:str, cima_id: str):
        """
        Prospecto reader to read the prospecto PDF from CIMA website

        Arguments:
            **reader**: Document reader to read the prospecto PDF
            **source**: The source from which to read the document (e.g., file path, URL)
            **cima_id**: CIMA ID of the prospecto to read
        """

        self.reader = reader
        self.source = source
        self.cima_id = cima_id

    def _get_metadata(self) -> dict:
        """
        Fetch metadata from the AEMPS CIMA database using the cima_id

        Returns:
            **metadata**: Fetched metadata
        """

        base_url = "https://cima.aemps.es/cima/rest/medicamento"

        try:

            logging.info(f"Fetching metadata for CIMA ID {self.cima_id} from {base_url}")

            response = requests.get(
                url=base_url,
                params={"nregistro": self.cima_id},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            return {
                "med_id": self.cima_id,
                "source": f"https://cima.aemps.es/cima/dochtml/p/{self.cima_id}/",
                "med_name": data.get("nombre"),
                "active_principle": ", ".join([princ.get("nombre", "") for princ in data.get("principiosActivos", [{}])]),
                "last_updated": data.get("docs")[-1].get("fecha"),
                "atcs": data.get("atcs", [])[0].get("nombre", ""),
                "excipients": ", ".join([exc.get("nombre", "") for exc in data.get("excipientes", [{}])]),
                "administrations": data.get("viasAdministracion", [])[0].get("nombre", ""),
                "dosis": data.get("dosis", "")
            }

        except Exception as e:
            logging.error(f"Error fetching metadata for CIMA ID {self.cima_id}: {e}")
            raise RuntimeError(f"Error fetching metadata for CIMA ID {self.cima_id}: {e}")