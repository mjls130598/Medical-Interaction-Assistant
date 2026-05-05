import logging
from typing import List

import requests

from langchain_core.documents import Document

from .document_loader import DocumentLoader
from ..readers.document_reader import DocumentReader
from ..cleaners.text_cleaner import TextCleaner


class ProspectoLoader(DocumentLoader):
    def __init__(self, reader: DocumentReader, cleaner: TextCleaner, source:str, cima_id: str):
        """
        Prospecto reader to read the prospecto PDF from CIMA website

        Arguments:
            **reader**: Document reader to read the prospecto PDF
            **cleaner**: Text cleaner to clean the extracted text
            **source**: The source from which to read the document (e.g., file path, URL)
            **cima_id**: CIMA ID of the prospecto to read
        """

        super().__init__(reader, cleaner, source)
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
                "atcs": ", ".join([atc.get("nombre", "") for atc in data.get("atcs", [{}])]),
                "excipients": ", ".join([exc.get("nombre", "") for exc in data.get("excipientes", [{}])]),
                "administrations": ", ".join([admin.get("nombre", "") for admin in data.get("viasAdministracion", [{}])]),
                "dosis": data.get("dosis", "")
            }

        except Exception as e:
            logging.error(f"Error fetching metadata for CIMA ID {self.cima_id}: {e}")
            raise RuntimeError(f"Error fetching metadata for CIMA ID {self.cima_id}: {e}")
        
    def create_document(self) -> List[Document]:
        """
        Creates a list of Documents with the content and metadata of the prospecto to load

        Returns:
            **documents**: List of Documents with the content and metadata of the prospecto to load
        """
        
        logging.info("Getting metadata for the document")        
        metadata = self._get_metadata()

        logging.info("Creating documents with content and metadata")

        sections = self.get_sections()

        documents = [
            Document(
                page_content=section['content'],
                metadata={
                    **metadata,
                    'section_id': section['section_id'],
                    'section_title': section['section_title']
                }
            )
            for section in sections
        ]

        return documents