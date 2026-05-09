from unittest.mock import MagicMock, patch

from app.service.assistance.groq_rag_assistance import GroqRAGAssistance


def make_mock_embedding_strategy(return_vector: list[float]) -> MagicMock:
    """Create a mock embedding strategy returning a fixed vector."""
    strategy = MagicMock()
    strategy.embed_batch.return_value = [return_vector]
    return strategy


def make_mock_vector_store() -> MagicMock:
    """Create a mock vector store for initializing the assistance class."""
    return MagicMock()


class TestGroqRAGAssistance:

    @patch("app.service.assistance.groq_rag_assistance.Groq")
    @patch("app.service.assistance.groq_rag_assistance.os.getenv", return_value="test_api_key")
    def test_init_creates_groq_client_with_api_key(self, mock_getenv, mock_groq):
        """Test that GroqRAGAssistance initializes Groq client using environment API key."""
        client_instance = MagicMock()
        mock_groq.return_value = client_instance

        assistance = GroqRAGAssistance(
            vector_store=make_mock_vector_store(),
            embedding_strategy=make_mock_embedding_strategy([0.1, 0.2]),
            model="custom-model"
        )

        mock_getenv.assert_called_once_with("GROQ_API_KEY")
        mock_groq.assert_called_once_with(api_key="test_api_key")
        assert assistance.client is client_instance
        assert assistance.model == "custom-model"
        assert "asistente virtual sanitario" in assistance.system_prompt

    @patch("app.service.assistance.groq_rag_assistance.Groq")
    @patch("app.service.assistance.groq_rag_assistance.os.getenv", return_value="test_api_key")
    def test_ask_creates_messages_and_returns_stripped_content(self, mock_getenv, mock_groq):
        """Test that ask uses retrieved context and returns stripped chat completion content."""
        client_instance = MagicMock()
        mock_groq.return_value = client_instance

        content_mock = MagicMock()
        content_mock.content = "   final answer   "
        response_mock = MagicMock()
        response_mock.choices = [MagicMock(message=content_mock)]
        client_instance.chat.completions.create.return_value = response_mock

        assistance = GroqRAGAssistance(
            vector_store=make_mock_vector_store(),
            embedding_strategy=make_mock_embedding_strategy([0.2, 0.3]),
            model="custom-model"
        )

        with patch.object(GroqRAGAssistance, "_get_relevant_context", return_value="relevant context") as mock_context:
            answer = assistance.ask("How are you?")

        expected_messages = [
            {"role": "system", "content": assistance.system_prompt},
            {"role": "user", "content": "Contexto relevante:\nrelevant context\n\nPregunta: How are you?"}
        ]

        mock_context.assert_called_once_with("How are you?")
        client_instance.chat.completions.create.assert_called_once_with(
            messages=expected_messages,
            model="custom-model",
            temperature=0.2
        )
        assert answer == "final answer"

    @patch("app.service.assistance.groq_rag_assistance.Groq")
    @patch("app.service.assistance.groq_rag_assistance.os.getenv", return_value="test_api_key")
    def test_ask_forwards_model_and_temperature(self, mock_getenv, mock_groq):
        """Test that ask forwards the configured model and temperature to Groq completions."""
        client_instance = MagicMock()
        mock_groq.return_value = client_instance

        response_mock = MagicMock()
        message_mock = MagicMock()
        message_mock.content = "response"
        response_mock.choices = [MagicMock(message=message_mock)]
        client_instance.chat.completions.create.return_value = response_mock

        assistance = GroqRAGAssistance(
            vector_store=make_mock_vector_store(),
            embedding_strategy=make_mock_embedding_strategy([0.2, 0.3])
        )

        with patch.object(GroqRAGAssistance, "_get_relevant_context", return_value="context"):
            assistance.ask("query")

        client_instance.chat.completions.create.assert_called_once()
        args, kwargs = client_instance.chat.completions.create.call_args
        assert kwargs["model"] == "llama-3.3-70b-versatile"
        assert kwargs["temperature"] == 0.2
