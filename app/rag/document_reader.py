from abc import ABC, abstractmethod


class DocumentReader(ABC):
    @abstractmethod
    def read(self, source:str) -> str:
        """
        Reads a document from the given source and returns its content as a string.

        Arguments:
            **source**: The source from which to read the document (e.g., file path, URL)
        
        Returns:
            **content**: The content of the document as a string
        """