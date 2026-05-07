import pytest
from unittest.mock import Mock, patch, MagicMock

from app.rag.cleaners.chunking_strategy.groq_strategy import GroqStrategy


class TestGroqStrategy:
    """Test class for GroqStrategy methods."""

    @patch('app.rag.cleaners.chunking_strategy.groq_strategy.tiktoken.encoding_for_model')
    def test_init_with_available_encoder(self, mock_encoding_for_model):
        """Test GroqStrategy initialization with available encoder."""
        mock_encoder = MagicMock()
        mock_encoding_for_model.return_value = mock_encoder
        
        strategy = GroqStrategy(model_name="gpt-4")
        assert strategy.encoder == mock_encoder
        mock_encoding_for_model.assert_called_once_with("gpt-4")

    @patch('app.rag.cleaners.chunking_strategy.groq_strategy.tiktoken.get_encoding')
    @patch('app.rag.cleaners.chunking_strategy.groq_strategy.tiktoken.encoding_for_model')
    def test_init_with_unavailable_encoder_fallback(self, mock_encoding_for_model, mock_get_encoding):
        """Test GroqStrategy initialization falls back to default encoder."""
        mock_encoding_for_model.side_effect = Exception("Model not found")
        mock_encoder = MagicMock()
        mock_get_encoding.return_value = mock_encoder
        
        strategy = GroqStrategy(model_name="unknown-model")
        assert strategy.encoder == mock_encoder
        mock_get_encoding.assert_called_once_with('cl100k_base')

    @patch('app.rag.cleaners.chunking_strategy.groq_strategy.tiktoken.encoding_for_model')
    def test_length(self, mock_encoding_for_model):
        """Test length method calculates token count."""
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
        mock_encoding_for_model.return_value = mock_encoder
        
        strategy = GroqStrategy()
        result = strategy.length("test text")
        assert result == 5
        mock_encoder.encode.assert_called_with("test text")

    @patch('app.rag.cleaners.chunking_strategy.groq_strategy.tiktoken.encoding_for_model')
    def test_get_split_index_with_double_newline(self, mock_encoding_for_model):
        """Test get_split_index finds double newline as split point."""
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(50))  # 50 tokens
        decoded = "Some text here.\n\nNext paragraph starts."
        mock_encoder.decode.return_value = decoded
        mock_encoding_for_model.return_value = mock_encoder
        
        strategy = GroqStrategy()
        result = strategy.get_split_index("text", 50)
        expected_pos = decoded.rfind("\n\n")
        assert result == expected_pos

    @patch('app.rag.cleaners.chunking_strategy.groq_strategy.tiktoken.encoding_for_model')
    def test_get_split_index_with_period_and_capital(self, mock_encoding_for_model):
        """Test get_split_index uses period followed by capital letter."""
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(50))
        decoded = "First sentence. Second sentence here."
        mock_encoder.decode.return_value = decoded
        mock_encoding_for_model.return_value = mock_encoder
        
        strategy = GroqStrategy()
        result = strategy.get_split_index("text", 50)
        import re
        matches = list(re.finditer(r'\. (?=[A-ZÁÉÍÓÚ])', decoded))
        expected_pos = matches[-1].end() if matches else -1
        assert result == expected_pos

    @patch('app.rag.cleaners.chunking_strategy.groq_strategy.tiktoken.encoding_for_model')
    def test_get_split_index_fallback_to_space(self, mock_encoding_for_model):
        """Test get_split_index falls back to last space."""
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(50))
        decoded = "word1 word2 word3 word4"
        mock_encoder.decode.return_value = decoded
        mock_encoding_for_model.return_value = mock_encoder
        
        strategy = GroqStrategy()
        result = strategy.get_split_index("text", 50)
        expected_pos = decoded.rfind(" ")
        assert result == expected_pos

    @patch('app.rag.cleaners.chunking_strategy.groq_strategy.tiktoken.encoding_for_model')
    def test_get_split_index_no_split_found(self, mock_encoding_for_model):
        """Test get_split_index returns full length when no split found."""
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = list(range(50))
        decoded = "nospaceorperiods"
        mock_encoder.decode.return_value = decoded
        mock_encoding_for_model.return_value = mock_encoder
        
        strategy = GroqStrategy()
        result = strategy.get_split_index("text", 50)
        assert result == len(decoded)

    @patch('app.rag.cleaners.chunking_strategy.groq_strategy.tiktoken.encoding_for_model')
    def test_get_overlap_text(self, mock_encoding_for_model):
        """Test get_overlap_text returns decoded last tokens."""
        mock_encoder = MagicMock()
        tokens = [10, 20, 30, 40, 50]
        mock_encoder.encode.return_value = tokens
        overlap_text = "last two"
        mock_encoder.decode.return_value = overlap_text
        mock_encoding_for_model.return_value = mock_encoder
        
        strategy = GroqStrategy()
        result = strategy.get_overlap_text("original text", 2)
        assert result == overlap_text
        mock_encoder.decode.assert_called_with([40, 50])
