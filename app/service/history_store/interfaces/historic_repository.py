from abc import ABC, abstractmethod


class HistoryRepository(ABC):

    @abstractmethod
    def save_interaction(self, session_id: str, human_message: dict, ai_response: dict):
        """
        Save a chat interaction to the history repository.

        Arguments:
            session_id (str): The unique identifier for the chat session.
            human_message (dict): A dictionary representing the user's message, typically containing 'role' and 'content'.
            ai_response (dict): The AI's response to the user's message.
        """
        pass

    @abstractmethod
    def get_history(self, session_id: str) -> list[dict]:
        """
        Retrieve the chat history for a given session ID.

        Arguments:
            session_id (str): The unique identifier for the chat session.

        Returns:
            list[dict]: A list of dictionaries representing the chat history, where each dictionary contains 
                        'role' and 'content' keys for both human messages and AI responses.

        """
        pass