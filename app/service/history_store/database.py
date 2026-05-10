import logging
import os

from pymongo import MongoClient


class MongoDBClient:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBClient, cls).__new__(cls)

            uri = os.getenv("MONGO_URL")
            db_name = os.getenv("MONGO_DB_NAME")

            if not uri or not db_name:
                raise ValueError("MONGO_URL and MONGO_DB_NAME environment variables must be set.")

            cls._instance.client = MongoClient(uri)
            cls._instance.db = cls._instance.client[db_name]

            logging.info("MongoDB client initialized successfully.")
            
        return cls._instance
    
    @property
    def medical_db(self):
        return self.db
    
    def close_connection(self):
        if self._instance and self._instance.client:
            self._instance.client.close()
            logging.info("MongoDB connection closed.")