import logging
from typing import List

from .chunking_strategy import ChunkingStrategy
from .groq_strategy import GroqStrategy


class ContentChunker:
    def __init__(self, strategy: ChunkingStrategy = GroqStrategy(),
                 max_unit: int = 512, overlap_percentage: float = 0.2):
        
        logging.info("Configure content chunker")
        
        self.strategy = strategy
        self.max_unit = max_unit
        self.overlap_unit = int(max_unit * overlap_percentage)

    def split (self, text: str) -> List[str]:
        
        logging.info("Split the given text")
        
        chunks = []
        new_content = text.strip()
        overlap_text = ""

        while self.strategy.length(new_content) > 0:

            logging.info("Create new block with overlap text (if it exists)")
            current_block = (overlap_text + " " + new_content).strip() if overlap_text else new_content

            if self.strategy.length(current_block) <= self.max_unit:
                logging.info("There isn't such big text to split into pieces")
                chunks.append(current_block.strip())
                break

            logging.info("Get the perfect split index")
            split_pos = self.strategy.get_split_index(current_block, self.max_unit)

            logging.info("Add the current chunk")
            current_chuck = current_block[:split_pos].strip()
            chunks.append(current_chuck)

            logging.info("Prepare for the next block with the overlap")
            overlap_text = self.strategy.get_overlap_text(current_chuck, self.overlap_unit)
            new_content = current_block[split_pos:]

            if not new_content:
                break

        return chunks