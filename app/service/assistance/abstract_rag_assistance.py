from abc import ABC, abstractmethod
import logging

from app.rag.embedding.embedding_strategy.embedding_strategy import EmbeddingStrategy
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

        self.system_prompt = (            
            "Eres un asistente virtual sanitario estrictamente informativo. \n"
            "1. Analiza el contexto. 2. Identifica la respuesta. 3. Si no hay evidencia textual directa, declara ignorancia.\n"
            "NUNCA recomiendes cambiar un tratamiento médico.\n"
            "Siempre añade la cláusula: 'Esta información no sustituye el consejo médico profesional'.\n"
            "Si el usuario pregunta por un medicamento específico, responde solo si el texto lo menciona explícitamente.\n"
            "Si la respuesta no está en el contexto, usa exactamente esta frase: 'Lo siento, el prospecto proporcionado "
            "no contiene información sobre [tema]'.\n" 
            "REGLAS CRÍTICAS:\n"
            "1. CITAS: Cada vez que afirmes algo basado en el contexto, añade el número de fuente al final de la frase,"
            " por ejemplo: 'La dosis recomendada es de 500mg [1]'.\n"
            "2. BIBLIOGRAFÍA: Al final de tu respuesta, crea una sección llamada 'Fuentes consultadas'"
            " donde listes el nombre del medicamento, la sección y el enlace (URL) de cada fuente utilizada "
            "(separado entre viñetas cada fuente).\n Ejemplo:\n"
            "Fuentes consultadas:\n"
            "[1] Paracetamol (Prospecto) - Sección: Indicaciones. URL: https://cima.aemps.es/...\n"
            "[2] Paracetamol (Prospecto) - Sección: Posología." 
        )

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

        formatted_context = []

        for i, (doc_text, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            chunk_info = (
                f"-- FUENTE [{i+1}] --\n"
                f"Medicamento: {metadata.get('med_name', 'Desconocido')}\n"
                f"Sección: {metadata.get('section_title', 'Desconocida')}\n"
                f"URL: {metadata.get('source', 'Desconocida')}\n"
                f"Contenido: {doc_text}"
            )
            formatted_context.append(chunk_info)

        return "\n\n".join(formatted_context)