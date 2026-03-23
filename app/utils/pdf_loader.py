from pathlib import Path
import re
from typing import List, Tuple
import fitz
from langchain_core.documents import Document
import logging

class MedicalPDFLoader:
    def __init__(self, file_path: str):
        """
        Creates a new Medical PDF loader

        Arguments:
            **file_path**: Medical PDF file path
        """

        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"Error: {file_path} doesn't exist")
        
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Error: {file_path} isn't a PDF file")

        self.file_path = file_path  

    def _clean_block(self, block:str) -> str:
        """
        Clean text block:
        1. Removing new lines inside words (f.e., ace-\ntil to acetil)
        2. Joining lines which doesn't start with -, number section or capital letter
        3. Concatenating multiple spaces in only one

        Returns a string with the cleaned text block

        Arguments:
            **block**: Text block to clean
        """

        logging.info("Cleaning block")

        # 1. Remove new lines inside words
        # For example: "ace- \n tilcisteína" -> "acetilcisteína"
        complete_words = re.sub(r'-\s*\n\s*', '', block)

        # 2. Join lines which doesn't start with -, number or capital letter 
        pattern = r'\n(?!\s*(?:[-•]|(?:\d+[.)\s])|[A-ZÁÉÍÓÚ]))'
        complete_sentences = re.sub(pattern, ' ', complete_words)
        
        # 3. Concatenate multiple spaces in only one
        return re.sub(r' +', " ", complete_sentences).strip()
    
    def _create_paragraphs(self, blocks) -> List[Tuple[int, str]]:
        """
        Create cleaned paragraphs from fitx text blocks

        Returns a list of tuples (page number -> int, text blocks -> str)

        Arguments:
            **blocks**: Array of cleaned text blocks
        """

        paragraphs = []

        for idx, (page_num, block) in enumerate(blocks):
            
            logging.info(f"Extracting block nº {idx + 1}")

            # If it's not a text
            if block[6] != 0:
                logging.info("Not text")
                continue

            page_pattern = r'\b\d+\s+de\s+\d+\b'

            if re.search(page_pattern, block[4]):
                logging.info("Removing page number")
                continue

            text = self._clean_block(block[4])
            
            # If there isn't text
            if not text:
                logging.info("There isn't text")
                continue
            
            if paragraphs:
                logging.info("Checking if the last paragraph is not complete and the current text is from the last paragraph")
                last_paragraph = paragraphs[-1][1]
                is_incomplete = not last_paragraph.endswith(('.', ':', '?', '!'))
                starts_with_low = text[0].islower()

                if is_incomplete and starts_with_low:
                    logging.info("Concatenate last paragraph with current text")
                    paragraphs[-1] = (paragraphs[-1][0], f'{last_paragraph} {text}')
                    continue

            logging.info("Creating new paragraph")                        
            paragraphs.append((page_num, text))

        return paragraphs


    def _read_pdf(self) -> Tuple[List[Tuple[int, str]], int]:
        """
        Read and extract pages from Medical PDF file

        Returns a list of tuples (page number -> int, text block -> str) and
        total page numbers of the PDF
        """

        try:
            with fitz.open(self.file_path) as doc:
                
                logging.info(f"Reading {self.file_path}")

                logging.info("Extracting blocks from document")
                all_blocks = (
                    (num_page, block) for num_page, page in enumerate(doc)
                    for block in page.get_text("blocks") # read paragraphs instead of lines
                )
                
                logging.info("Extracting paragraphs from blocks")
                paragraphs = self._create_paragraphs(all_blocks)

                logging.info(f"Finish reading {self.file_path}. Extracted {len(paragraphs)} paragraphs")

                pages = {}

                for paragraph in paragraphs:
                    page_num = paragraph[0]
                    text = paragraph[1]

                    if page_num in pages:
                        pages[page_num] += f"\n {text}"

                    else:
                        pages[page_num] = text 

                return sorted(pages.items()), len(doc)

        except Exception as e:
            logging.error(f"Error processing {self.file_path}: {e}")

    def read_load_document(self) -> List[Document]:
        """
        Read the Medical PDF and save the extracted information into a langchain Document

        Returns a list of langchain documents
        """

        logging.info(f"1. READ {self.file_path} AND EXTRACT PARAGRAPHS")
        pages, total_pages = self._read_pdf()

        logging.info("2. SAVE PARAGRAPHS IN DOCUMENT")

        if not pages:
            raise ValueError("There isn't information to save")
        
        documents = [
            Document(
                page_content=content,
                metadata={
                    "source": self.file_path,
                    "page": page_num + 1,
                    "total_pages": total_pages
                }
            )
            for page_num, content in pages
        ]

        return documents
