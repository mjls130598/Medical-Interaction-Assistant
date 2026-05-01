from abc import ABC, abstractmethod
import logging
from typing import List

from rag.document_reader import DocumentReader
from langchain_core.documents import Document


class DocumentLoader(ABC):
    def __init__(self, reader: DocumentReader, source: str):
        """
        Creates a new Document loader to load documents using the provided reader

        Arguments:
            **reader**: Document reader to read the document
            **source**: The source from which to read the document (e.g., file path, URL)
        """

        self.reader = reader
        self.source = source


    @abstractmethod
    def _get_metadata(self) -> dict:
        """
        Fetch metadata for the document to load

        Returns:
            **metadata**: Fetched metadata
        """
        pass
        
    def create_document(self) -> List[Document]:
        """
        Creates a list of Documents with the content and metadata of the document to load

        Returns:
            **documents**: List of Documents with the content and metadata of the document to load
        """

        logging.info(f"1. READ {self.source} AND EXTRACT METADATA AND SECTIONS")
        sections = self.reader.read(self.source)
        
        if not sections:
            logging.error("There isn't information to save")
            raise ValueError("There isn't information to save")
        
        logging.info("Getting metadata for the document")        
        metadata = self._get_metadata()
        
        logging.info("2. SAVE SECTIONS IN DOCUMENT")
        
        documents = [
            Document(
                page_content=content['content'],
                metadata={
                    **metadata,
                    'section_id': content['section_id'],
                    'section_title': content['section_title']
                }
            )
            for content in sections
        ]

        logging.info(f"Created {len(documents)} documents with metadata: {metadata}")

        return documents