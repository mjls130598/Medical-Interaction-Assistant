from abc import ABC, abstractmethod
from typing import List, Tuple


class DocumentReader(ABC):
    @abstractmethod
    def read(self, source:str) -> List[Tuple[int, str]]:
        """
        Reads a document from the given source and returns its content as a list of sections.

        Arguments:
            **source**: The source from which to read the document (e.g., file path, URL)
        
        Returns:
            **content**: The content of the document as a list of sorted sections, 
            where each section is a tuple of (section_id, section_title, content)
        """
        pass