from abc import ABC, abstractmethod


class EmbeddingStrategy(ABC):

    @abstractmethod
    def embed_batch(self, text: list[str]) -> list[list[float]]:
        pass