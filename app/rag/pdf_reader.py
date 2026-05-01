import logging
from pathlib import Path

import fitz

from rag.document_reader import DocumentReader
from rag.text_cleaner import TextCleaner


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

                logging.info("Extracting blocks from document")
                all_blocks = (
                    (num_page, block) for num_page, page in enumerate(doc)
                    for block in page.get_text("blocks") # read paragraphs instead of lines
                )
                
                logging.info("Extracting paragraphs from blocks")
                paragraphs = TextCleaner.create_paragraphs(all_blocks)

                logging.info(f"Finish reading {source}. Extracted {len(paragraphs)} paragraphs")

                sections = {}

                for paragraph in paragraphs:
                    page_num = paragraph['page_num']
                    text = paragraph['content']
                    section_id = paragraph['section_id']
                    section_title = paragraph['section_title']

                    if section_id in sections:
                        sections[section_id]['content'] += f"\n {text}"

                    else:
                        sections[section_id] = {
                            'section_id': section_id,
                            'section_title': section_title,
                            'page_num': page_num + 1,
                            'content': text
                        }

                sorted_sections = [v for k, v in sorted(sections.items(), key=lambda item: item[0])]

                return sorted_sections, len(doc)

        except Exception as e:
            logging.error(f"Error processing {source}: {e}")