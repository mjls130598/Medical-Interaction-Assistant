import logging
import os

from pymongo import MongoClient


class MongoDBClient:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBClient, cls).__new__(cls)
            cls._instance._initialize_client()

            uri = os.getenv("MONGO_URL")
            db_name = os.getenv("MONGO_DB_NAME")

            cls._instance.client = MongoClient(uri)
            cls._instance.db = cls._instance.client[db_name]

            logging.info("MongoDB client initialized successfully.")
            
        return cls._instance
    
    @property
    def medical_db(self):
        return self._instance.db
    
    def close_connection(self):
        if self._instance and self._instance.client:
            self._instance.client.close()
            logging.info("MongoDB connection closed.")