from abc import ABC, abstractmethod
import logging
from typing import List

from ..readers.document_reader import DocumentReader
from langchain_core.documents import Document

from ..cleaners.text_cleaner import TextCleaner


class DocumentLoader(ABC):
    def __init__(self, reader: DocumentReader, cleaner: TextCleaner, source: str):
        """
        Creates a new Document loader to load documents using the provided reader

        Arguments:
            **reader**: Document reader to read the document
            **cleaner**: Text cleaner to clean the document content
            **source**: The source from which to read the document (e.g., file path, URL)
        """

        self.reader = reader
        self.cleaner = cleaner
        self.source = source


    @abstractmethod
    def _get_metadata(self) -> dict:
        """
        Fetch metadata for the document to load

        Returns:
            **metadata**: Fetched metadata
        """
        pass
        
    @abstractmethod
    def create_document(self) -> List[Document]:
        """
        Creates a list of Documents with the content and metadata of the document to load

        Returns:
            **documents**: List of Documents with the content and metadata of the document to load
        """
        
        pass
    
    def get_sections(self) -> List[dict]:

        """
        Gets the sections of the document to load by reading the document and creating sections with the cleaner
        Returns:
            **sections**: List of sections with content, section id and section title
        """

        logging.info(f"Getting text from {self.source}")
        raw_text = self.reader.read(self.source)

        logging.info("Creating sections from the document")
        sections = self.cleaner.create_paragraphs(raw_text)
        
        return sections