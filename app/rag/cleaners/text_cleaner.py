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
    
    def _is_page_number(self, text: str) -> bool:
        """
        Check if the text is a page number

        Arguments:
            **text**: Text to check
        Returns:
            True if the text is a page number, False otherwise
        """
        logging.info("Checking if the text is a page number")
        page_pattern = r'\b\d+\s+de\s+\d+\b'
        return bool(re.search(page_pattern, text))
    
    def _append_to_buffer(self, buffer_text: str, new_text: str) -> str:
        """
        Append new text to the buffer text, adding a space if the buffer text doesn't end with a punctuation mark and the new text doesn't start with a capital letter.

        Arguments:
            **buffer_text**: Text buffer to append the new text to
            **new_text**: New text to append to the buffer

        Returns:
            Updated buffer text
        """
        is_incomplete = not buffer_text.endswith(('.', ':', '?', '!'))
        starts_with_low = new_text[0].islower()

        if is_incomplete and starts_with_low:
            return buffer_text + f" {new_text}"
        else:
            return buffer_text + f"\n{new_text}"

    def create_sections(self, text: str) -> List[dict]:
        """
        Create cleaned sections from fit text blocks.

        Arguments:
            **text**: Text extracted from the document as string

        Returns:
            **sections**: List of sections with content, 
                          section id and section title
        """

        if not text or text == "":
            logging.warning("Empty text received, returning empty sections list")
            return []

        lines = text.splitlines()
        sections = []

        # Initialize context for section tracking
        context = {
            "section_id": "0",
            "section_title": "Introducción",
            "content": ""
        }

        for idx, text in enumerate(lines):
            
            logging.info(f"Extracting line nº {idx + 1}")

            text = text.strip()

            if not text or self._is_page_number(text):
                logging.info("Empty line or page number detected, skipping")
                continue

            # Extract section from the text
            sec_id, sec_title = self._extract_section(text)

            if sec_id: 
                logging.info(f"Section found: {sec_id} - {sec_title}")

                # If we have a section in the buffer, we need to save it before updating the context
                if context["content"]:
                    context["content"] = self._clean_line(context["content"])
                    sections.append(context.copy())
                    logging.info("Content added as a new section")

                # Update context with the new section
                context["section_id"] = sec_id
                context["section_title"] = sec_title
                context["content"] = ""
                
                continue

            logging.info("Appending line to buffer")
            context["content"] = self._append_to_buffer(context["content"], text)

            # If we are at the last line, we need to concatenate
            # the buffer text with the current line 
            if idx == len(lines) - 1:
                logging.info("Last line reached without " \
                "finding a new section, adding remaining " \
                "buffer text to the last section")
                context["content"] = self._clean_line(context["content"])
                sections.append(context.copy())
                continue

        return sections