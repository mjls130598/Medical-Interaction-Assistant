import logging
import re

from .chunking_strategy import ChunkingStrategy
import tiktoken


class GroqStrategy(ChunkingStrategy):
    def __init__(self, model_name: str = "gpt-4"):
        logging.info("Setting the encoder for GroQ")

        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except:
            logging.exception("Encoding gaven isn't available. Getting cl100K_base")
            self.encoder = tiktoken.get_encoding('cl100k_base')

    def length(self, text) -> int:
        return len(self.encoder.encode(text))
    
    def get_split_index(self, text: str, max_tokens: int) -> int:

        logging.info("Check the correct split index")

        # Get all tokens from the text
        tokens = self.encoder.encode(text)

        # Decode the max block to look at a natural cut
        decoded_fragment = self.encoder.decode(tokens[:max_tokens])

        logging.info("First try: Get the first double space for new paragraph")
        pos = decoded_fragment.rfind("\n\n")

        if pos == -1:
            logging.info("Second try: Get the first full stop")
            matches = list(re.finditer(r'\. (?=[A-ZÁÉÍÓÚ])', decoded_fragment))
            pos = matches[-1].end() if matches else -1

        if pos == -1:
            logging.info("Third try: Get the proper space")
            pos = decoded_fragment.rfind(" ")

        return pos if pos != -1 else len(decoded_fragment)
    
    def get_overlap_text(self, text: str, num_tokens: str):
        tokens = self.encoder.encode(text)
        return self.encoder.decode(tokens[-num_tokens:])