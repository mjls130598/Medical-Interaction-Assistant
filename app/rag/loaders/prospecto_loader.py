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
                "med_name": self._clean_string(data.get("nombre", "")),
                "active_principle": self._clean_string(
                    "| ".join([princ.get("nombre", "") for princ in data.get("principiosActivos", [{}])])),
                "n_principles": len(data.get("principiosActivos", [])),
                "last_updated": data.get("docs")[-1].get("fecha"),
                "atcs": self._clean_string("| ".join([atc.get("nombre", "") for atc in data.get("atcs", [{}])])),
                "excipients": self._clean_string("| ".join([exc.get("nombre", "") for exc in data.get("excipientes", [{}])])),
                "administrations": self._clean_string("| ".join([admin.get("nombre", "") for admin in data.get("viasAdministracion", [{}])])),
                "dosis": self._clean_string(data.get("dosis", "").lower())
            }

        except Exception as e:
            logging.error(f"Error fetching metadata for CIMA ID {self.cima_id}: {e}")
            raise RuntimeError(f"Error fetching metadata for CIMA ID {self.cima_id}: {e}")
        
    def _get_context(self, med_name: str, active_principle: str,
                     section_title: str, content: str) -> str:
        """
        Gets the context of the prospecto to load by creating a string with the medication name,
        active principle, section title and content

        Arguments:
            **med_name**: Name of the medication
            **active_principle**: Active principle of the medication
            **section_title**: Title of the section of the prospecto
            **content**: Content of the section of the prospecto

        Returns:
            **context**: Cleaned context of the prospecto to load
        """

        logging.info(f"Creating context for medication {med_name}, " + 
                     f"active principle {active_principle}, section {section_title}")

        return (
            f"DATOS DEL MEDICAMENTO: {med_name} ({active_principle})\n"
            f"SECCIÓN DEL PROSPECTO: {section_title}\n"
            f"CONTENIDO:\n{content}"
        )
        
    def create_document(self) -> List[Document]:
        """
        Creates a list of Documents with the content and metadata of the prospecto to load

        Returns:
            **documents**: List of Documents with the content and metadata of the prospecto to load
        """
        
        logging.info("Getting metadata for the document")        
        metadata = self._get_metadata()

        logging.info("Creating documents with content and metadata")

        sections = self._get_sections()

        documents = [
            Document(
                page_content=self._get_context(
                    med_name=metadata['med_name'],
                    active_principle=metadata['active_principle'],
                    section_title=section['section_title'],
                    content=section['content']
                ),
                metadata={
                    **metadata,
                    'document_id': f"ID-{metadata['med_id']}-{section['section_id']}-{section['chunk_id']}" \
                                    if 'chunk_id' in section else f"ID-{metadata['med_id']}-{section['section_id']}",
                    'section_id': section['section_id'],
                    'section_title': section['section_title']
                }
            )
            for section in sections
        ]

        return documents