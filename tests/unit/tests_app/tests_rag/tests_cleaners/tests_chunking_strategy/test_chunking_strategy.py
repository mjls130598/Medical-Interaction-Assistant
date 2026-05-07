import pytest
from abc import ABC

from app.rag.cleaners.chunking_strategy.chunking_strategy import ChunkingStrategy


class TestChunkingStrategy:
    """Test class for ChunkingStrategy abstract class."""

    def test_is_abstract_class(self):
        """Test that ChunkingStrategy is an abstract class."""
        assert issubclass(ChunkingStrategy, ABC)

    def test_cannot_instantiate_abstract_class(self):
        """Test that ChunkingStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ChunkingStrategy()

    def test_abstract_methods_exist(self):
        """Test that abstract methods are defined."""
        assert hasattr(ChunkingStrategy, 'length')
        assert hasattr(ChunkingStrategy, 'get_split_index')
        assert hasattr(ChunkingStrategy, 'get_overlap_text')

        with pytest.raises(TypeError) as excinfo:
            ChunkingStrategy()
        assert "Can't instantiate abstract class ChunkingStrategy" in str(excinfo.value)

        class IncompleteStrategy(ChunkingStrategy):
            pass

        with pytest.raises(TypeError):
            IncompleteStrategy()