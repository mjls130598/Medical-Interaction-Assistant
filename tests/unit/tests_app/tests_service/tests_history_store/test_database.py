import os
import pytest
from unittest.mock import patch, MagicMock
from app.service.history_store.database import MongoDBClient


class TestMongoDBClient:
    """Test class for MongoDBClient singleton."""

    @patch('app.service.history_store.database.MongoClient')
    @patch.dict(os.environ, {'MONGO_URL': 'test_url', 'MONGO_DB_NAME': 'test_db'})
    def test_singleton_instance_creation(self, mock_mongo_client):
        """Test that MongoDBClient creates a singleton instance."""
        # Reset singleton for test
        MongoDBClient._instance = None

        mock_client = MagicMock()
        mock_mongo_client.return_value = mock_client
        mock_db = MagicMock()
        mock_client.__getitem__.return_value = mock_db

        instance1 = MongoDBClient()
        instance2 = MongoDBClient()

        assert instance1 is instance2
        mock_mongo_client.assert_called_once_with('test_url')
        mock_client.__getitem__.assert_called_once_with('test_db')

    @patch('app.service.history_store.database.MongoClient')
    @patch.dict(os.environ, {'MONGO_URL': 'test_url', 'MONGO_DB_NAME': 'test_db'})
    def test_medical_db_property(self, mock_mongo_client):
        """Test the medical_db property returns the database instance."""
        MongoDBClient._instance = None

        mock_client = MagicMock()
        mock_mongo_client.return_value = mock_client
        mock_db = MagicMock()
        mock_client.__getitem__.return_value = mock_db

        instance = MongoDBClient()
        db = instance.medical_db

        assert db is mock_db

    @patch('app.service.history_store.database.MongoClient')
    @patch.dict(os.environ, {'MONGO_URL': 'test_url', 'MONGO_DB_NAME': 'test_db'})
    def test_close_connection(self, mock_mongo_client):
        """Test closing the MongoDB connection."""
        MongoDBClient._instance = None

        mock_client = MagicMock()
        mock_mongo_client.return_value = mock_client

        instance = MongoDBClient()
        instance.close_connection()

        mock_client.close.assert_called_once()

    @patch('app.service.history_store.database.MongoClient')
    @patch.dict(os.environ, {'MONGO_URL': 'test_url', 'MONGO_DB_NAME': 'test_db'})
    def test_close_connection_no_instance(self, mock_mongo_client):
        """Test closing connection when no instance exists."""
        MongoDBClient._instance = None

        # Call close without creating instance
        MongoDBClient().close_connection()

        # Should not raise error, but since no instance, nothing to close
        # In code, it checks if _instance, so safe