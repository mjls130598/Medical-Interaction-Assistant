import logging

from .embedding_strategy import EmbeddingStrategy
from sentence_transformers import SentenceTransformer


class SentenceTransformerStrategy(EmbeddingStrategy):

    def __init__(self, model_name: str="all-MiniLM-L6-v2"):
        logging.info(f"Initializing SentenceTransformerStrategy with model " +
                     f"{model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_batch(self, text: list[str]) -> list[list[float]]:
        logging.info(f"Embedding batch of {len(text)} documents using " +
                     f"SentenceTransformerStrategy")
        return self.model.encode(text).tolist()