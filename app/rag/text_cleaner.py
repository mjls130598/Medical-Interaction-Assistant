import logging
import re
from typing import List


class TextCleaner:
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
    
    @staticmethod
    def create_paragraphs(self, blocks) -> List[dict]:
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