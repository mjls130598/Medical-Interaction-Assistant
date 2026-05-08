from abc import ABC, abstractmethod
import logging

from rag.embedding.embedding_strategy.embedding_strategy import EmbeddingStrategy
from ..vector_store.vectore_store_manager import VectorStoreManager


class AbstractRAGAssistance(ABC):
    def __init__(self, vector_store: VectorStoreManager,
                 embedding_strategy: EmbeddingStrategy):
        """
        Abstract class for RAG-based assistance services.
        Arguments:
            vector_store (VectorStoreManager): An instance of VectorStoreManager for handling vector storage.
            embedding_strategy (EmbeddingStrategy): An instance of EmbeddingStrategy for generating embeddings.
        """
        self.vector_store = vector_store
        self.embedding_strategy = embedding_strategy

    @abstractmethod
    def ask(self, query: str) -> str:
        """
        Process a user query and return an answer based on the RAG approach.
        This method should be implemented by subclasses to define specific retrieval and generation logic.
        
        Arguments:
            query (str): The user's question or input for which an answer is sought.
        """
        pass

    def _get_relevant_context(self, query: str, n_results: int = 5) -> str:
        
        logging.info("Looking context for query: " + query)

        query_vector = self.embedding_strategy.embed_batch([query])[0]
        results = self.vector_store.search_relevant_chunks(
            query_embedding=query_vector,
            n_results=n_results
        )

        return "\n\n".join(results['documents'][0])