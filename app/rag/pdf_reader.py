from rag.document_reader import DocumentReader


class PDFReader(DocumentReader):
    def read(self, source: str) -> str:
        """
        Reads a PDF document from the given source and returns its content as a string.

        Arguments:
            **source**: The file path of the PDF document to read

        Returns:
            **content**: The content of the PDF document as a string
        """

        pass