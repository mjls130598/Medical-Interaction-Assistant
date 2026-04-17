from pathlib import Path
import re
from typing import List, Tuple
import fitz
from langchain_core.documents import Document
import logging

from .metadata_api import MetadataAPI

# Patterns to extract metadata from the PDF
PATTERNS = {
    "cima_id": r"https://cima\.aemps\.es/cima/dochtml/p/(\d+)/"
}

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

        Returns:
            Cleaned text block
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
    
    def _extract_section(self, text: str) -> str:
        """
        Extract a section of the text using a regex pattern

        Arguments:
            **text**: Text to extract the section from

        Returns:
            Extracted section
        """

        logging.info("Extracting section")

        pattern = r'^(\d+(?:\.\d+)*)\.?\s+([A-ZÁÉÍÓÚ][^.\n]+)'
        match = re.search(pattern, text)

        if match:
            return match.group(1), match.group(2).strip()
        return None, None
    
    def _create_paragraphs(self, blocks) -> List[dict]:
        """
        Create cleaned paragraphs from fit text blocks.

        Arguments:
            **blocks**: Array of cleaned text blocks

        Returns:
            **paragraphs**: List of paragraphs with page number, content, section id and section title
        """

        paragraphs = []
        current_section_id = "0"
        current_section_title = "Introduction"

        for idx, (page_num, block) in enumerate(blocks):
            
            logging.info(f"Extracting block nº {idx + 1}")

            # If it's not a text
            if block[6] != 0:
                logging.info("Not text")
                continue

            text = block[4]

            # Extract section from the text
            sec_id, sec_title = self._extract_section(text)

            if sec_id:
                logging.info(f"New section found: {sec_id} {sec_title}")
                current_section_id = sec_id
                current_section_title = sec_title

            page_pattern = r'\b\d+\s+de\s+\d+\b'

            if re.search(page_pattern, text):
                logging.info("Removing page number")
                continue

            text = self._clean_block(text)
            
            # If there isn't text
            if not text:
                logging.info("There isn't text")
                continue
            
            if paragraphs:
                logging.info("Checking if the last paragraph is not complete and the current text is from the last paragraph")
                last_paragraph = paragraphs[-1]['content']
                is_incomplete = not last_paragraph.endswith(('.', ':', '?', '!'))
                starts_with_low = text[0].islower()

                if is_incomplete and starts_with_low:
                    logging.info("Concatenate last paragraph with current text")
                    paragraphs[-1]['content'] += f" {text}"
                    continue

            logging.info("Creating new paragraph")                        
            paragraphs.append({
                "page_num": page_num, 
                "content": text,
                "section_id": current_section_id,
                "section_title": current_section_title
            })

        return paragraphs


    def _read_pdf(self) -> Tuple[List[Tuple[int, str]], int]:
        """
        Read and extract pages from Medical PDF file

        Returns
            **sections**: section ID -> str, section data -> dict with section title, page number and content
            **total_pages**: total page numbers of the PDF
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
            logging.error(f"Error processing {self.file_path}: {e}")


    def read_load_document(self) -> List[Document]:
        """
        Read the Medical PDF and save the extracted information into a langchain Document

        Returns
            **documents**: a list of langchain documents
        """

        logging.info(f"1. READ {self.file_path} AND EXTRACT METADATA AND PARAGRAPHS")
        sections, total_pages = self._read_pdf()
        full_text = "\n".join([content['content'] for content in sections])
        
        cima_id_match = re.search(PATTERNS["cima_id"], full_text)
        cima_id = cima_id_match.group(1) if cima_id_match else None

        cima_metadata = {}
        if cima_id:
            logging.info(f"Enriqueciendo datos con ID de CIMA: {cima_id}")
            cima_client = MetadataAPI()
            cima_metadata = cima_client.fetch_metadata(cima_id)
        else:
            logging.warning("Didn't find CIMA ID in PDF text")

        logging.info("2. SAVE PARAGRAPHS IN DOCUMENT")

        if not sections:
            raise ValueError("There isn't information to save")
        
        documents = [
            Document(
                page_content=content['content'],
                metadata={
                    **cima_metadata,
                    'page': content['page_num'],
                    'section_id': content['section_id'],
                    'section_title': content['section_title'],
                    'total_pages': total_pages,
                }
            )
            for content in sections
        ]

        return documents
