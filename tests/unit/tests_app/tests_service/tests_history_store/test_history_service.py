from datetime import datetime, timezone
import pytest
from unittest.mock import MagicMock, patch
from app.service.history_store.history_service import HistoryService
from app.service.history_store.models.chat_session import MessageModel, MessageRole, MetadataModel, SourceModel


class TestHistoryService:
    """Test class for HistoryService."""

    def test_init(self):
        """Test initialization of HistoryService with a repository."""
        mock_repo = MagicMock()
        service = HistoryService(mock_repo)
        assert service.history_repository is mock_repo

    @patch('app.service.history_store.history_service.MessageModel')
    @patch('app.service.history_store.history_service.MetadataModel')
    @patch('app.service.history_store.history_service.SourceModel')
    def test_save_interaction_valid_messages(self, mock_source_model, mock_metadata_model, mock_message_model):
        """Test saving a valid human-AI interaction."""
        mock_repo = MagicMock()
        service = HistoryService(mock_repo)

        session_id = "test_session"
        human_message = {
            "content": "Hello",
            "timestamp": datetime.now(timezone.utc)
        }
        ai_response = {
            "content": "Hi there",
            "timestamp": datetime.now(timezone.utc),
            "metadata": {
                "sources": [
                    {
                        "url": "http://example.com",
                        "med_name": "Test Med",
                        "section_title": "Test Section",
                        "verified": True
                    }
                ],
                "total_sources_retrieved": 1
            }
        }

        # Mock the model instances
        mock_human_msg = MagicMock()
        mock_ai_msg = MagicMock()
        mock_message_model.side_effect = [mock_human_msg, mock_ai_msg]
        mock_human_msg.model_dump.return_value = {"role": "human", "content": "Hello"}
        mock_ai_msg.model_dump.return_value = {"role": "ai", "content": "Hi there"}

        mock_source = MagicMock()
        mock_source_model.return_value = mock_source

        mock_metadata = MagicMock()
        mock_metadata_model.return_value = mock_metadata

        service.save_interaction(session_id, human_message, ai_response)

        # Assert MessageModel called for human and AI
        assert mock_message_model.call_count == 2
        # Assert repository save called
        mock_repo.save_interaction.assert_called_once_with(
            session_id,
            mock_human_msg.model_dump(),
            mock_ai_msg.model_dump()
        )

    @patch('app.service.history_store.history_service.MessageModel')
    @patch('app.service.history_store.history_service.MetadataModel')
    @patch('app.service.history_store.history_service.SourceModel')
    def test_save_interaction_default_values(self, mock_source_model, mock_metadata_model, mock_message_model):
        """Test saving interaction with default values when fields are missing."""
        mock_repo = MagicMock()
        service = HistoryService(mock_repo)

        session_id = "test_session"
        human_message = {}  # Empty dict
        ai_response = {}  # Empty dict

        mock_human_msg = MagicMock()
        mock_ai_msg = MagicMock()
        mock_message_model.side_effect = [mock_human_msg, mock_ai_msg]
        mock_human_msg.model_dump.return_value = {"role": "human", "content": ""}
        mock_ai_msg.model_dump.return_value = {"role": "ai", "content": ""}

        mock_source = MagicMock()
        mock_source_model.return_value = mock_source

        mock_metadata = MagicMock()
        mock_metadata_model.return_value = mock_metadata

        with patch('app.service.history_store.history_service.datetime') as mock_datetime:
            mock_now = MagicMock()
            mock_datetime.now.return_value = mock_now
            mock_datetime.timezone.utc = timezone.utc

            service.save_interaction(session_id, human_message, ai_response)

            # Assert timestamps are set to now
            mock_message_model.assert_any_call(
                role=MessageRole.HUMAN,
                content="",
                timestamp=mock_now
            )
            mock_message_model.assert_any_call(
                role=MessageRole.AI,
                content="",
                metadata=mock_metadata,
                timestamp=mock_now
            )

    @patch('app.service.history_store.history_service.MessageModel')
    @patch('app.service.history_store.history_service.MetadataModel')
    @patch('app.service.history_store.history_service.SourceModel')
    def test_save_interaction_with_sources(self, mock_source_model, mock_metadata_model, mock_message_model):
        """Test saving interaction with multiple sources in metadata."""
        mock_repo = MagicMock()
        service = HistoryService(mock_repo)

        session_id = "test_session"
        human_message = {"content": "Test"}
        ai_response = {
            "content": "Response",
            "metadata": {
                "sources": [
                    {"url": "url1", "med_name": "med1", "section_title": "sec1", "verified": True},
                    {"url": "url2", "med_name": "med2", "section_title": "sec2", "verified": False}
                ],
                "total_sources_retrieved": 2
            }
        }

        mock_human_msg = MagicMock()
        mock_ai_msg = MagicMock()
        mock_message_model.side_effect = [mock_human_msg, mock_ai_msg]

        mock_source1 = MagicMock()
        mock_source2 = MagicMock()
        mock_source_model.side_effect = [mock_source1, mock_source2]

        mock_metadata = MagicMock()
        mock_metadata_model.return_value = mock_metadata

        service.save_interaction(session_id, human_message, ai_response)

        # Assert SourceModel called twice
        assert mock_source_model.call_count == 2
        mock_source_model.assert_any_call(
            index=1,
            url="url1",
            med_name="med1",
            section_title="sec1",
            verified=True
        )
        mock_source_model.assert_any_call(
            index=2,
            url="url2",
            med_name="med2",
            section_title="sec2",
            verified=False
        )

        # Assert MetadataModel called with sources
        mock_metadata_model.assert_called_once_with(
            sources=[mock_source1, mock_source2],
            total_sources_retrieved=2
        )