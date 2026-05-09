import pytest
from unittest.mock import MagicMock, patch

from app.service.assistance.groq_rag_assistance import GroqRAGAssistance


class TestAssistantFlow:
    """
    Integration tests for the assistant flow, covering document processing, 
    embedding generation, and vector store interactions.
    """

    @pytest.fixture
    def assistant(self, vector_store, embedding_strategy):
        """Fixture providing a fully initialized GroqRAGAssistance instance for integration testing."""
        return GroqRAGAssistance(
            vector_store=vector_store, 
            embedding_strategy=embedding_strategy
        )

    def test_ask_flow_integration(self, assistant, embedder, sample_documents):
        """Check the complete flow: context retrieval and LLM response."""
        
        embedded_docs = embedder.generate_embeddings(sample_documents)
        assistant.vector_store.save_documents(embedded_docs)

        mock_text = "Según la FUENTE [1], la educación del paciente es clave para el éxito."
        
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=mock_text))
        ]

        with patch("app.service.assistance.groq_rag_assistance.Groq.chat.completions.create") as mock_create:
            mock_create.return_value = mock_response
            
            query = "¿Por qué es importante la educación del paciente?"
            response = assistant.ask(query)

            assert response == mock_text
            mock_create.assert_called_once()
            
            args, kwargs = mock_create.call_args
            prompt_sent = kwargs['messages'][1]['content']
            assert "Patient education" in prompt_sent
        