import pytest
from abc import ABC
from app.service.history_store.interfaces.historic_repository import HistoryRepository


class TestHistoryRepository:
    """Test class for HistoryRepository abstract interface."""

    def test_is_abstract_class(self):
        """Test that HistoryRepository is an abstract base class."""
        assert issubclass(HistoryRepository, ABC)

    def test_abstract_methods(self):
        """Test that the required abstract methods are defined."""
        # Check that save_interaction is abstract
        assert hasattr(HistoryRepository, 'save_interaction')
        # Check that get_history is abstract
        assert hasattr(HistoryRepository, 'get_history')

    def test_cannot_instantiate(self):
        """Test that HistoryRepository cannot be instantiated directly."""
        with pytest.raises(TypeError):
            HistoryRepository()