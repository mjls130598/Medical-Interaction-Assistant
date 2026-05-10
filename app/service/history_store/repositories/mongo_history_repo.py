from datetime import datetime, timezone
import logging

from ..interfaces.historic_repository import HistoryRepository


class MongoHistoryRepo(HistoryRepository):
    def __init__(self, db_client):
        self.collection = db_client.sessions

    def save_interaction(self, session_id: str, human_message: dict, ai_response: dict):
        """
        Save a human-AI interaction to the MongoDB collection for the given session ID.
        
        Arguments:
            session_id (str): The unique identifier for the chat session.
            human_message (dict): A dictionary containing the human message details (role, content, timestamp).
            ai_response (dict): A dictionary containing the AI response details (role, content, timestamp, metadata).
        """
        
        date = datetime.now(timezone.utc)
        
        new_messages = [human_message, ai_response]

        self.collection.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": {"$each": new_messages}},
                "$set": {"updated_at": date},
                "$setOnInsert": {"created_at": date}
            },
            upsert=True
        )

        logging.info(f"Interaction for session {session_id} saved to MongoDB.")