from datetime import datetime, timezone
import logging
import os

from groq import Groq

from app.rag.embedding.embedding_strategy.embedding_strategy import EmbeddingStrategy
from .abstract_rag_assistance import AbstractRAGAssistance
from ..vector_store.vectore_store_manager import VectorStoreManager


class GroqRAGAssistance(AbstractRAGAssistance):
    def __init__(self, vector_store: VectorStoreManager,
                 embedding_strategy: EmbeddingStrategy,
                 db_connection = None,
                 model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the GroqRAGAssistance with a vector store manager and an embedding strategy.
        Arguments:
            vector_store (VectorStoreManager): An instance of VectorStoreManager for handling vector storage.
            embedding_strategy (EmbeddingStrategy): An instance of EmbeddingStrategy for generating embeddings.
            db_connection: Database connection for storing chat history and interactions.
            model (str): The Groq model to use for generating responses. Default is "ll
        """
        logging.info("Initializing GroqRAGAssistance with vector store and embedding strategy.")
        
        super().__init__(vector_store=vector_store,
                         embedding_strategy=embedding_strategy,
                         db_connection=db_connection)
        
        self.client = Groq(api_key = os.getenv("GROQ_API_KEY"))
        self.model = model
        
    def ask(self, query: str, session_id: str) -> str:
        """
        Process a user query and return an answer based on the RAG approach using Groq for generation.

        Arguments:
            query (str): The user's question or input for which an answer is sought.
            session_id (str): The unique identifier for the chat session.

        Returns:
            str: The generated answer based on the retrieved context and the query.
        """

        query_timestamp = datetime.now(timezone.utc)

        logging.info("Getting relevant context for query: " + query)
        context, source_metadata = self._get_relevant_context(query)

        logging.info("Generating answer using Groq with retrieved context.")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Contexto relevante:\n{context}\n\nPregunta: {query}"}
        ]

        logging.debug(f"Messages sent to Groq: {messages}")
        response = self.client.chat.completions.create(
            messages = messages,
            model = self.model,
            temperature = 0.2
        )

        logging.info("Received response from Groq.")
        ai_response = response.choices[0].message.content.strip()

        final_response = self._format_ai_response(ai_response, source_metadata, response.created)
        logging.debug(f"Final formatted response: {final_response}")

        if self.db:
            logging.info("Saving interaction to the database.")
            human_msg = {"role": "human", "content": query, "timestamp": query_timestamp}
            self.db.save_interaction(session_id, human_msg, final_response)

        return final_response