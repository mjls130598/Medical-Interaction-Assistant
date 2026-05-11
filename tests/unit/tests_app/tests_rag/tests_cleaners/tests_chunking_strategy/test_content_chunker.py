import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from app.rag.cleaners.chunking_strategy.content_chunker import ContentChunker


class TestContentChunker:
    """Test class for ContentChunker methods."""

    @pytest.fixture
    def mock_strategy(self):
        """Fixture to provide a mocked ChunkingStrategy."""
        strategy = MagicMock()
        # Mock length to return size based on words for predictable testing
        strategy.length.side_effect = lambda x: len(x.split())
        return strategy

    def test_init_custom_values(self, mock_strategy):
        """Test ContentChunker initialization with custom values."""
        chunker = ContentChunker(strategy=mock_strategy, max_unit=1000, overlap_percentage=0.1)
        assert chunker.strategy == mock_strategy
        assert chunker.max_unit == 1000
        assert chunker.overlap_unit == 100

    def test_split_text_shorter_than_max_unit(self, mock_strategy):
        """Test split when text is shorter than max unit."""
        mock_strategy.length.return_value = 100
        chunker = ContentChunker(strategy=mock_strategy, max_unit=200)
        
        result = chunker.split("short text")
        assert result == ["short text"]
        mock_strategy.length.assert_called_with("short text")

    def test_split_text_exactly_max_unit(self, mock_strategy):
        """Test split when text length equals max unit."""
        mock_strategy.length.return_value = 512
        chunker = ContentChunker(strategy=mock_strategy, max_unit=512)
        
        result = chunker.split("exact length text")
        assert result == ["exact length text"]

    def test_split_text_longer_than_max_unit(self, mock_strategy):
        """Test split when text is longer than max unit."""
        mock_strategy = mock_strategy.return_value
        
        mock_strategy.length.side_effect = [600, 600, 200, 200]
        mock_strategy.get_split_index.return_value = 10
        mock_strategy.get_overlap_text.return_value = "overlap"
        
        chunker = ContentChunker(strategy=mock_strategy, max_unit=512, overlap_percentage=0.1)
        
        result = chunker.split("texto largo que necesita ser dividido")
        
        assert len(result) == 2
        assert mock_strategy.get_split_index.called

    def test_split_with_overlap(self, mock_strategy):
        mock_strategy = mock_strategy.return_value
        mock_strategy.length.side_effect = [600, 600, 200, 200, 0]
        mock_strategy.get_split_index.return_value = 10
        mock_strategy.get_overlap_text.return_value = "SOLAPE"
        
        chunker = ContentChunker(strategy=mock_strategy, max_unit=500)
        result = chunker.split("texto para solapar")
        
        assert len(result) == 2
        assert "SOLAPE" in result[1]

    def test_split_empty_text(self, mock_strategy):
        """Test split with empty text."""
        mock_strategy.length.return_value = 0
        chunker = ContentChunker(strategy=mock_strategy)
        
        result = chunker.split("")
        assert result == []

    def test_split_whitespace_only(self, mock_strategy):
        """Test split with whitespace only text."""
        mock_strategy.length.return_value = 0
        chunker = ContentChunker(strategy=mock_strategy)
        
        result = chunker.split("   ")
        assert result == []