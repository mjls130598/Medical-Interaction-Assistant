from abc import ABC
from unittest.mock import MagicMock

import pytest

from app.service.assistance import abstract_rag_assistance as assistance_module


class ConcreteRAGAssistance(assistance_module.AbstractRAGAssistance):
    """Concrete helper class for testing abstract assistance behavior."""

    def ask(self, query: str) -> str:
        return ""


def make_mock_embedding_strategy(return_vector: list[float]) -> MagicMock:
    """Create a mock embedding strategy returning a fixed vector."""
    strategy = MagicMock()
    strategy.embed_batch.return_value = [return_vector]
    return strategy


def make_mock_vector_store(results: dict) -> MagicMock:
    """Create a mock vector store returning a fixed query result."""
    store = MagicMock()
    store.search_relevant_chunks.return_value = results
    return store


class TestAbstractRAGAssistance:

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that AbstractRAGAssistance cannot be instantiated directly."""
        with pytest.raises(TypeError):
            assistance_module.AbstractRAGAssistance(
                vector_store=MagicMock(),
                embedding_strategy=MagicMock()
            )

    def test_get_relevant_context_returns_formatted_documents(self):
        """Test that _get_relevant_context returns properly formatted document context with metadata."""
        vector_store = make_mock_vector_store({
            "documents": [["doc A", "doc B"]],
            "metadatas": [[
                {"med_name": "Paracetamol", "section_title": "Indicaciones", "source": "https://example.com/1"},
                {"med_name": "Ibuprofeno", "section_title": "Posología", "source": "https://example.com/2"}
            ]]
        })
        embedding_strategy = make_mock_embedding_strategy([0.5, 0.6])

        assistance = ConcreteRAGAssistance(
            vector_store=vector_store,
            embedding_strategy=embedding_strategy
        )

        context = assistance._get_relevant_context("query text")

        embedding_strategy.embed_batch.assert_called_once_with(["query text"])
        vector_store.search_relevant_chunks.assert_called_once_with(
            query_embedding=[0.5, 0.6],
            n_results=5
        )

        expected_context = (
            "-- FUENTE [1] --\n"
            "Medicamento: Paracetamol\n"
            "Sección: Indicaciones\n"
            "URL: https://example.com/1\n"
            "Contenido: doc A\n\n"
            "-- FUENTE [2] --\n"
            "Medicamento: Ibuprofeno\n"
            "Sección: Posología\n"
            "URL: https://example.com/2\n"
            "Contenido: doc B"
        )

        assert context == expected_context

    def test_get_relevant_context_uses_custom_n_results(self):
        """Test that _get_relevant_context forwards the n_results parameter to the vector store."""
        vector_store = make_mock_vector_store({
            "documents": [["single document"]],
            "metadatas": [[{"med_name": "Desconocido", "section_title": "Desconocida", "source": "Desconocida"}]]
        })
        embedding_strategy = make_mock_embedding_strategy([0.9, 0.1])

        assistance = ConcreteRAGAssistance(
            vector_store=vector_store,
            embedding_strategy=embedding_strategy
        )

        context = assistance._get_relevant_context("another query", n_results=3)

        vector_store.search_relevant_chunks.assert_called_once_with(
            query_embedding=[0.9, 0.1],
            n_results=3
        )
        assert "Contenido: single document" in context
