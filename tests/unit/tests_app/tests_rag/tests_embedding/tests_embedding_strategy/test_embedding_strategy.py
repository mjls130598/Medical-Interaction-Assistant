import pytest

from app.rag.embedding.embedding_strategy.embedding_strategy import EmbeddingStrategy


class DummyEmbeddingStrategy(EmbeddingStrategy):
    """Simple concrete implementation of EmbeddingStrategy for testing."""

    def embed_batch(self, text: list[str]) -> list[list[float]]:
        return [[float(len(item))] for item in text]


class TestEmbeddingStrategy:

    def test_abstract_base_class_cannot_be_instantiated(self):
        """Test that the abstract base class raises TypeError when instantiated."""
        with pytest.raises(TypeError):
            EmbeddingStrategy()

    def test_concrete_subclass_can_embed_batch(self):
        """Test that a concrete subclass correctly implements embed_batch."""
        strategy = DummyEmbeddingStrategy()
        output = strategy.embed_batch(["a", "bb"])

        assert output == [[1.0], [2.0]]
