import logging
import re

from .chunking_strategy import ChunkingStrategy
import tiktoken


class GroqStrategy(ChunkingStrategy):
    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the GroQ strategy with the appropriate encoder for the given model.
        Arguments:
            **model_name**: Name of the model to get the encoder for (default: "gpt-4")
        """
        logging.info("Setting the encoder for GroQ")

        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except:
            logging.exception("Encoding gaven isn't available. Getting cl100K_base")
            self.encoder = tiktoken.get_encoding('cl100k_base')

    def length(self, text) -> int:
        """
        Calculate the length of the text in terms of tokens using the encoder.
        Arguments:
            **text**: Text to calculate the length of
        Returns:
            Length of the text in terms of tokens
        """
        return len(self.encoder.encode(text))
    
    def get_split_index(self, text: str, max_tokens: int) -> int:
        """
        Get the index where the text should be split.
        Arguments:
            **text**: Text to split
            **max_tokens**: Maximum number of tokens for each chunk
        Returns:
            Index where the text should be split
        """
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
        """
        Get the overlapping text for the next chunk.
        Arguments:
            **text**: Text to overlap
            **num_tokens**: Number of tokens to overlap
        Returns:
            Overlapping text for the next chunk
        """
        tokens = self.encoder.encode(text)
        return self.encoder.decode(tokens[-num_tokens:])