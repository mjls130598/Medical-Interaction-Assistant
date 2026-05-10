from datetime import datetime, timezone
import logging

from .models.chat_session import MessageModel, MessageRole, MetadataModel, SourceModel
from .interfaces.historic_repository import HistoryRepository


class HistoryService:
    """
    Service responsible for managing chat history, including saving and retrieving chat sessions.
    """

    def __init__(self, history_repository: HistoryRepository):
        self.history_repository = history_repository

    def save_interaction(self, session_id: str, human_message: dict, ai_response: dict):
        """
        Save a human-AI interaction to the history repository for the given session ID.
        
        Arguments:
            session_id (str): The unique identifier for the chat session.
            human_message (dict): A dictionary containing the human message details (role, content, timestamp).
            ai_response (dict): A dictionary containing the AI response details (role, content, timestamp, metadata).
        """
        
        logging.info(f"Checking if human message and AI response are valid for session {session_id}.")
        human_valid = MessageModel(
            role=MessageRole.HUMAN,
            content=human_message.get("content", ""),
            timestamp=human_message.get("timestamp", datetime.now(timezone.utc))
        )
        ai_valid = MessageModel(
            role=MessageRole.AI,
            content=ai_response.get("content", ""),
            metadata=MetadataModel(
                sources = [SourceModel(
                    index=i+1,
                    url=source.get("url", "Desconocida"),
                    med_name=source.get("med_name", "Desconocida"),
                    section_title=source.get("section_title", "Desconocida"),
                    verified=source.get("verified", False)
                ) for i, source in enumerate(ai_response.get("metadata", {}).get("sources", []))],
                total_sources_retrieved=ai_response.get("metadata", {}).get("total_sources_retrieved", 0)
            ),
            timestamp=ai_response.get("timestamp", datetime.now(timezone.utc))
        )

        logging.info(f"Saving interaction for session {session_id} to the history repository.")
        self.history_repository.save_interaction(session_id, human_valid.model_dump(), ai_valid.model_dump())