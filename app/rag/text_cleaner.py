from abc import ABC, abstractmethod
import logging
import re
from typing import List


class TextCleaner(ABC):

    @abstractmethod
    def _extract_section(self, text: str) -> str:
        """
        Extract a section of the text using a regex pattern

        Arguments:
            **text**: Text to extract the section from

        Returns:
            Extracted section
        """
        pass
    
    def _clean_line(self, line:str) -> str:
        """
        Clean text line by applying the following transformations:
        1. Removing new lines inside words (f.e., ace-\ntil to acetil)
        2. Joining lines which doesn't start with -, number section or capital letter
        3. Concatenating multiple spaces in only one

        Returns a string with the cleaned text line

        Arguments:
            **line**: Text line to clean

        Returns:
            Cleaned text line
        """

        logging.info("Cleaning line")

        # 1. Remove new lines inside words
        # For example: "ace- \n tilcisteína" -> "acetilcisteína"
        complete_words = re.sub(r'-\s*\n\s*', '', line)

        # 2. Join lines which doesn't start with -, number or capital letter 
        pattern = r'\n(?!\s*(?:[-•]|(?:\d+[.)\s])|[A-ZÁÉÍÓÚ]))'
        complete_sentences = re.sub(pattern, ' ', complete_words)
        
        # 3. Concatenate multiple spaces in only one
        return re.sub(r' +', " ", complete_sentences).strip()
    

    def create_paragraphs(self, text: str) -> List[dict]:
        """
        Create cleaned paragraphs from fit text blocks.

        Arguments:
            **text**: Text extracted from the document as string

        Returns:
            **paragraphs**: List of paragraphs with content, section id and section title
        """

        lines = text.splitlines()

        paragraphs = []
        current_section_id = "0"
        current_section_title = "Introduction"

        for idx, text in enumerate(lines):
            
            logging.info(f"Extracting line nº {idx + 1}")

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

            text = self._clean_line(text)
            
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
                "content": text,
                "section_id": current_section_id,
                "section_title": current_section_title
            })

        return paragraphs