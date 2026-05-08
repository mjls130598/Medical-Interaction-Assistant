from unittest.mock import MagicMock, patch

import pytest

from app.rag.embedding.embedding_strategy.sentence_transformer_strategy import SentenceTransformerStrategy


class TestSentenceTransformerStrategy:

    @patch("app.rag.embedding.embedding_strategy.sentence_transformer_strategy.logging.info")
    @patch("app.rag.embedding.embedding_strategy.sentence_transformer_strategy.SentenceTransformer")
    def test_init_uses_default_model(self, mock_sentence_transformer, mock_logging):
        """Test default model name is passed to SentenceTransformer."""
        model_instance = MagicMock()
        mock_sentence_transformer.return_value = model_instance

        strategy = SentenceTransformerStrategy()

        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
        assert strategy.model is model_instance

    @patch("app.rag.embedding.embedding_strategy.sentence_transformer_strategy.logging.info")
    @patch("app.rag.embedding.embedding_strategy.sentence_transformer_strategy.SentenceTransformer")
    def test_init_uses_custom_model_name(self, mock_sentence_transformer, mock_logging):
        """Test a custom model_name is passed correctly to SentenceTransformer."""
        model_instance = MagicMock()
        mock_sentence_transformer.return_value = model_instance

        strategy = SentenceTransformerStrategy(model_name="test-model")

        mock_sentence_transformer.assert_called_once_with("test-model")
        assert strategy.model is model_instance

    @patch("app.rag.embedding.embedding_strategy.sentence_transformer_strategy.logging.info")
    def test_embed_batch_calls_model_encode_and_returns_vectors(self, mock_logging):
        """Test that embed_batch calls model.encode and returns the converted vectors."""
        strategy = SentenceTransformerStrategy.__new__(SentenceTransformerStrategy)
        strategy.model = MagicMock()

        encoded = MagicMock()
        encoded.tolist.return_value = [[0.2, 0.4]]
        strategy.model.encode.return_value = encoded

        output = strategy.embed_batch(["hello world"])

        strategy.model.encode.assert_called_once_with(["hello world"])
        assert output == [[0.2, 0.4]]

    @patch("app.rag.embedding.embedding_strategy.sentence_transformer_strategy.logging.info")
    def test_embed_batch_with_empty_text_list(self, mock_logging):
        """Test embed_batch handles an empty text list by encoding an empty list."""
        strategy = SentenceTransformerStrategy.__new__(SentenceTransformerStrategy)
        strategy.model = MagicMock()

        encoded = MagicMock()
        encoded.tolist.return_value = []
        strategy.model.encode.return_value = encoded

        output = strategy.embed_batch([])

        strategy.model.encode.assert_called_once_with([])
        assert output == []
