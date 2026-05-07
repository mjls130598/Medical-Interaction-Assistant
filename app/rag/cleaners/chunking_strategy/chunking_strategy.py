from abc import ABC, abstractmethod


class ChunkingStrategy(ABC):
    """
    Abstract class for chunking strategy. It defines the interface for chunking text into smaller sections.
    """

    @abstractmethod
    def length(self, text: str) -> int:
        """
        Calculate the length of the text in terms of tokens.
        Arguments:
            **text**: Text to calculate the length of
        Returns:
            Length of the text in terms of tokens
        """
        pass

    @abstractmethod
    def get_split_index(self, text: str, max_len: int) -> int:
        """
        Get the index where the text should be split.
        Arguments:
            **text**: Text to split
            **max_len**: Maximum length of each chunk
        Returns:
            Index where the text should be split
        """
        pass

    @abstractmethod
    def get_overlap_text(self, text: str, num_tokens: int) -> str:
        """
        Get the overlapping text for the next chunk.
        Arguments:
            **text**: Text to overlap
            **num_tokens**: Number of tokens to overlap
        Returns:
            Overlapping text for the next chunk
        """
        pass