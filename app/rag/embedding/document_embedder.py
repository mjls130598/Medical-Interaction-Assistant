import logging
from typing import Dict, List

from langchain_core.documents import Document

from .embedding_strategy.embedding_strategy import EmbeddingStrategy


class DocumentEmbedder:
    def __init__(self, strategy: EmbeddingStrategy):
        """
        Initialize the DocumentEmbedder with a specific embedding strategy.
        Arguments:
            strategy (EmbeddingStrategy): The embedding strategy to use for embedding documents.
        """
        self.strategy = strategy

    def generate_embeddings(self, sections: List[Document]) -> List[Document]:
        """
        Generate embeddings for a list of document sections.

        Arguments:
            sections (List[Document]): A list of document sections, each containing a "content" key.

        Returns:
            List[Document]: A list of document sections with added "embedding" keys.
        """

        logging.info("Starting embedding orchestration")

        texts = [section.page_content for section in sections]

        vectors = self.strategy.embed_batch(texts)

        for i, section in enumerate(sections):
            section.metadata["embedding"] = vectors[i]

        return sections