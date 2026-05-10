from datetime import datetime, timezone
import pytest
from unittest.mock import MagicMock, patch
from app.service.history_store.repositories.mongo_history_repo import MongoHistoryRepo


class TestMongoHistoryRepo:
    """Test class for MongoHistoryRepo."""

    def test_init(self):
        """Test initialization of MongoHistoryRepo with db_client."""
        mock_db_client = MagicMock()
        mock_collection = MagicMock()
        mock_db_client.sessions = mock_collection

        repo = MongoHistoryRepo(mock_db_client)

        assert repo.collection is mock_collection

    @patch('app.service.history_store.repositories.mongo_history_repo.datetime')
    def test_save_interaction(self, mock_datetime):
        """Test saving an interaction to MongoDB."""
        mock_db_client = MagicMock()
        mock_collection = MagicMock()
        mock_db_client.sessions = mock_collection

        repo = MongoHistoryRepo(mock_db_client)

        session_id = "test_session"
        human_message = {"role": "human", "content": "Hello"}
        ai_response = {"role": "ai", "content": "Hi"}

        mock_now = MagicMock()
        mock_datetime.now.return_value = mock_now
        mock_datetime.timezone.utc = timezone.utc

        repo.save_interaction(session_id, human_message, ai_response)

        # Assert update_one called with correct parameters
        mock_collection.update_one.assert_called_once_with(
            {"session_id": session_id},
            {
                "$push": {"messages": {"$each": [human_message, ai_response]}},
                "$set": {"updated_at": mock_now},
                "$setOnInsert": {"created_at": mock_now}
            },
            upsert=True
        )

    @patch('app.service.history_store.repositories.mongo_history_repo.datetime')
    def test_save_interaction_with_existing_session(self, mock_datetime):
        """Test saving interaction when session already exists (upsert)."""
        mock_db_client = MagicMock()
        mock_collection = MagicMock()
        mock_db_client.sessions = mock_collection

        repo = MongoHistoryRepo(mock_db_client)

        session_id = "existing_session"
        human_message = {"role": "human", "content": "Test"}
        ai_response = {"role": "ai", "content": "Response"}

        mock_now = MagicMock()
        mock_datetime.now.return_value = mock_now

        repo.save_interaction(session_id, human_message, ai_response)

        # The upsert=True ensures it works for both new and existing
        mock_collection.update_one.assert_called_once()

    def test_save_interaction_collection_access(self):
        """Test that collection is accessed correctly."""
        mock_db_client = MagicMock()
        mock_collection = MagicMock()
        mock_db_client.sessions = mock_collection

        repo = MongoHistoryRepo(mock_db_client)

        # Ensure collection is set
        assert repo.collection is mock_collection

    def test_get_history_existing_session(self):
        """Test retrieving history for an existing session with messages."""
        mock_db_client = MagicMock()
        mock_collection = MagicMock()
        mock_db_client.sessions = mock_collection

        repo = MongoHistoryRepo(mock_db_client)

        session_id = "existing_session"
        mock_messages = [
            {"role": "human", "content": "Hello"},
            {"role": "ai", "content": "Hi there"}
        ]
        mock_session = {"session_id": session_id, "messages": mock_messages}

        mock_collection.find_one.return_value = mock_session

        history = repo.get_history(session_id)

        mock_collection.find_one.assert_called_once_with({"session_id": session_id})
        assert history == mock_messages

    def test_get_history_non_existing_session(self):
        """Test retrieving history for a non-existing session."""
        mock_db_client = MagicMock()
        mock_collection = MagicMock()
        mock_db_client.sessions = mock_collection

        repo = MongoHistoryRepo(mock_db_client)

        session_id = "non_existing_session"

        mock_collection.find_one.return_value = None

        history = repo.get_history(session_id)

        mock_collection.find_one.assert_called_once_with({"session_id": session_id})
        assert history == []

    def test_get_history_session_without_messages(self):
        """Test retrieving history for a session that exists but has no messages."""
        mock_db_client = MagicMock()
        mock_collection = MagicMock()
        mock_db_client.sessions = mock_collection

        repo = MongoHistoryRepo(mock_db_client)

        session_id = "empty_session"
        mock_session = {"session_id": session_id}  # No messages key

        mock_collection.find_one.return_value = mock_session

        history = repo.get_history(session_id)

        mock_collection.find_one.assert_called_once_with({"session_id": session_id})
        assert history == []

    def test_get_history_empty_messages(self):
        """Test retrieving history when messages list is empty."""
        mock_db_client = MagicMock()
        mock_collection = MagicMock()
        mock_db_client.sessions = mock_collection

        repo = MongoHistoryRepo(mock_db_client)

        session_id = "empty_messages_session"
        mock_session = {"session_id": session_id, "messages": []}

        mock_collection.find_one.return_value = mock_session

        history = repo.get_history(session_id)

        mock_collection.find_one.assert_called_once_with({"session_id": session_id})
        assert history == []