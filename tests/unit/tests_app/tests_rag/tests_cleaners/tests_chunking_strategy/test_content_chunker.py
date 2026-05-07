import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from app.rag.cleaners.chunking_strategy.content_chunker import ContentChunker


class TestContentChunker:
    """Test class for ContentChunker methods."""

    def test_init_default_values(self):
        """Test ContentChunker initialization with default values."""
        mock_strategy = MagicMock()
    
        path = 'app.rag.cleaners.chunking_strategy.content_chunker.GroqStrategy'
        
        with patch(path) as mock_class:
            mock_class.return_value = mock_strategy
            chunker = ContentChunker(strategy=mock_strategy)
            assert chunker.strategy == mock_strategy
            assert chunker.max_unit == 512
            assert chunker.overlap_unit == int(512 * 0.2)

    def test_init_custom_values(self):
        """Test ContentChunker initialization with custom values."""
        mock_strategy = MagicMock()
        chunker = ContentChunker(strategy=mock_strategy, max_unit=1000, overlap_percentage=0.1)
        assert chunker.strategy == mock_strategy
        assert chunker.max_unit == 1000
        assert chunker.overlap_unit == 100

    @patch('app.rag.cleaners.chunking_strategy.content_chunker.GroqStrategy')
    def test_split_text_shorter_than_max_unit(self, mock_groq_strategy):
        """Test split when text is shorter than max unit."""
        mock_strategy = MagicMock()
        mock_strategy.length.return_value = 100
        chunker = ContentChunker(strategy=mock_strategy, max_unit=200)
        
        result = chunker.split("short text")
        assert result == ["short text"]
        mock_strategy.length.assert_called_with("short text")

    @patch('app.rag.cleaners.chunking_strategy.content_chunker.GroqStrategy')
    def test_split_text_exactly_max_unit(self, mock_groq_strategy):
        """Test split when text length equals max unit."""
        mock_strategy = MagicMock()
        mock_strategy.length.return_value = 512
        chunker = ContentChunker(strategy=mock_strategy, max_unit=512)
        
        result = chunker.split("exact length text")
        assert result == ["exact length text"]

    @patch('app.rag.cleaners.chunking_strategy.content_chunker.GroqStrategy')
    def test_split_text_longer_than_max_unit(self, mock_strategy_class):
        """Test split when text is longer than max unit."""
        mock_strategy = mock_strategy_class.return_value
        
        mock_strategy.length.side_effect = [600, 600, 200, 200]
        mock_strategy.get_split_index.return_value = 10
        mock_strategy.get_overlap_text.return_value = "overlap"
        
        chunker = ContentChunker(strategy=mock_strategy, max_unit=512, overlap_percentage=0.1)
        
        result = chunker.split("texto largo que necesita ser dividido")
        
        assert len(result) == 2
        assert mock_strategy.get_split_index.called

    @patch('app.rag.cleaners.chunking_strategy.content_chunker.GroqStrategy')
    def test_split_with_overlap(self, mock_strategy_class):
        mock_strategy = mock_strategy_class.return_value
        mock_strategy.length.side_effect = [600, 600, 200, 200, 0]
        mock_strategy.get_split_index.return_value = 10
        mock_strategy.get_overlap_text.return_value = "SOLAPE"
        
        chunker = ContentChunker(strategy=mock_strategy, max_unit=500)
        result = chunker.split("texto para solapar")
        
        assert len(result) == 2
        assert "SOLAPE" in result[1]

    @patch('app.rag.cleaners.chunking_strategy.content_chunker.GroqStrategy')
    def test_split_empty_text(self, mock_groq_strategy):
        """Test split with empty text."""
        mock_strategy = MagicMock()
        mock_strategy.length.return_value = 0
        chunker = ContentChunker(strategy=mock_strategy)
        
        result = chunker.split("")
        assert result == []

    @patch('app.rag.cleaners.chunking_strategy.content_chunker.GroqStrategy')
    def test_split_whitespace_only(self, mock_groq_strategy):
        """Test split with whitespace only text."""
        mock_strategy = MagicMock()
        mock_strategy.length.return_value = 0
        chunker = ContentChunker(strategy=mock_strategy)
        
        result = chunker.split("   ")
        assert result == []