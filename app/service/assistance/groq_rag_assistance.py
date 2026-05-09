import logging
import os

from groq import Groq

from rag.embedding.embedding_strategy.embedding_strategy import EmbeddingStrategy
from .abstract_rag_assistance import AbstractRAGAssistance
from ..vector_store.vectore_store_manager import VectorStoreManager


class GroqRAGAssistance(AbstractRAGAssistance):
    def __init__(self, vector_store: VectorStoreManager,
                 embedding_strategy: EmbeddingStrategy, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the GroqRAGAssistance with a vector store manager and an embedding strategy.
        Arguments:
            vector_store (VectorStoreManager): An instance of VectorStoreManager for handling vector storage.
            embedding_strategy (EmbeddingStrategy): An instance of EmbeddingStrategy for generating embeddings.
        """
        logging.info("Initializing GroqRAGAssistance with vector store and embedding strategy.")
        
        super().__init__(vector_store=vector_store,
                         embedding_strategy=embedding_strategy)
        
        self.client = Groq(api_key = os.getenv("GROQ_API_KEY"))
        self.model = model
        
    def ask(self, query: str) -> str:
        """
        Process a user query and return an answer based on the RAG approach using Groq for generation.

        Arguments:
            query (str): The user's question or input for which an answer is sought.

        Returns:
            str: The generated answer based on the retrieved context and the query.
        """

        logging.info("Getting relevant context for query: " + query)
        context = self._get_relevant_context(query)

        logging.info("Generating answer using Groq with retrieved context.")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Contexto relevante:\n{context}\n\nPregunta: {query}"}
        ]

        response = self.client.chat.completions.create(
            messages = messages,
            model = self.model,
            temperature = 0.2
        )

        return response.choices[0].message.content.strip()