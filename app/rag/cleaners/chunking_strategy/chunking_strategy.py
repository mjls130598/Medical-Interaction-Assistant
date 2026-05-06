from abc import ABC, abstractmethod


class ChunkingStrategy(ABC):

    @abstractmethod
    def length(self, text: str) -> int:
        pass

    @abstractmethod
    def get_split_index(self, text: str, max_len: int) -> int:
        pass

    @abstractmethod
    def get_overlap_text(self, text: str, num_tokens: int) -> str:
        pass