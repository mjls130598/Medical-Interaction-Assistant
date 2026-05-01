import logging
from pathlib import Path

import fitz

from rag.document_reader import DocumentReader


class PDFReader(DocumentReader):

    def _is_valid_pdf(self, file_path: str):
        """
        Checks if the given file path points to a valid PDF file.

        Arguments:
            **file_path**: The file path to check

        Returns:
            **is_valid**: True if the file is a valid PDF, False otherwise
        """

        path = Path(file_path)

        return path.is_file() and path.suffix.lower() == ".pdf"

    def read(self, source: str) -> str:
        """
        Reads a PDF document from the given source and returns its content as a string.

        Arguments:
            **source**: The file path of the PDF document to read

        Returns:
            **content**: The content of the PDF document as a string
        """

        if not self._is_valid_pdf(source):
            raise ValueError(f"Error: {source} isn't a valid PDF file")
        
        try:
            with fitz.open(source) as doc:
                
                logging.info(f"Reading {source}")

                # 1. Extracting text blocks from each page that are not empty
                # block[4] is the text, block[6] is the type (0 = text)
                blocks = [
                    block[4].strip() 
                    for page in doc 
                    for block in page.get_text("blocks") 
                    if block[6] == 0 and block[4].strip()
                ]

                # 2. Joining the text blocks into a single string with newlines
                content = "\n".join(blocks)

                # 3. Quick check for the log
                if not content:
                    logging.warning(f"The document {source} seems to be empty or scanned (OCR needed)")
                    raise RuntimeError(f"Document {source} is empty or requires OCR")
                else:
                    logging.info(f"Successful read: {len(content)} characters extracted from {source}")

                return content

        except Exception as e:
            logging.error(f"Error processing {source}: {e}")
            raise RuntimeError(f"Error processing {source}: {e}")