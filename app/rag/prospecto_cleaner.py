import logging
import re

from rag.text_cleaner import TextCleaner


class ProspectoCleaner(TextCleaner):
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