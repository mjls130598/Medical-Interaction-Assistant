from abc import ABC, abstractmethod
from datetime import datetime
import logging
import re

from app.rag.embedding.embedding_strategy.embedding_strategy import EmbeddingStrategy
from ..vector_store.vectore_store_manager import VectorStoreManager
from langchain_community.chat_message_histories import ChatMessageHistory


class AbstractRAGAssistance(ABC):
    def __init__(self, vector_store: VectorStoreManager,
                 embedding_strategy: EmbeddingStrategy, db_connection):
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

        self.db = db_connection

    @abstractmethod
    def ask(self, query: str, session_id: str) -> str:
        """
        Process a user query and return an answer based on the RAG approach.
        This method should be implemented by subclasses to define specific retrieval and generation logic.
        
        Arguments:
            query (str): The user's question or input for which an answer is sought.
            session_id (str): The unique identifier for the chat session.

        Returns:
            str: The generated answer based on the retrieved context and the query.
        """
        pass

    def _get_relevant_context(self, query: str, n_results: int = 5) -> tuple[str, list]:
        """
        Retrieve and format the most relevant context chunks for a given query.

        Arguments:
            query (str): The user's question or input for which relevant context is sought.
            n_results (int): The number of relevant chunks to retrieve from the vector store.

        Returns:
            tuple[str, list]: A formatted string containing the relevant context chunks with
                              source information, and a list of source metadata.
        """
        
        logging.info("Looking context for query: " + query)

        query_vector = self.embedding_strategy.embed_batch([query])[0]
        results = self.vector_store.search_relevant_chunks(
            query_embedding=query_vector,
            n_results=n_results
        )

        formatted_context = []
        source_metadata = []

        logging.info(f"Retrieved {len(results['documents'][0])} relevant chunks for the query.")

        for i, (doc_text, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            chunk_info = (
                f"-- FUENTE [{i+1}] --\n"
                f"Medicamento: {metadata.get('med_name', 'Desconocido')}\n"
                f"Sección: {metadata.get('section_title', 'Desconocida')}\n"
                f"URL: {metadata.get('source', 'Desconocida')}\n"
                f"Contenido: {doc_text}"
            )
            formatted_context.append(chunk_info)
            source_metadata.append({
                "med_name": metadata.get('med_name', 'Desconocido'),
                "section_title": metadata.get('section_title', 'Desconocida'),
                "source": metadata.get('source', 'Desconocida')
            })

        return "\n\n".join(formatted_context), source_metadata
    
    def _format_ai_response(self, ai_response: str, source_metadata: list, timestamp: datetime) -> dict:
        """
        Format the AI response to include source citations and a bibliography section based on the retrieved context.
        
        Arguments:
            ai_response (str): The raw response generated by the AI model.
            source_metadata (list): A list of metadata dictionaries for the sources retrieved as context.
        
        Returns:
            str: The formatted AI response with citations and bibliography.
        """
        found_indices = re.findall(r'\[(\d+)\]', ai_response)
        used_sources = []
        
        for idx_str in set(found_indices):
            idx = int(idx_str)
            if idx in source_metadata:
                used_sources.append({
                    "index": idx,
                    "url": source_metadata[idx]["source"],
                    "med_name": source_metadata[idx]["med_name"],
                    "section_title": source_metadata[idx]["section_title"],
                    "verified": True
                })

        return {
            "role": "ai",
            "content": ai_response,
            "metadata": {
                "sources": used_sources,
                "total_sources_retrieved": len(source_metadata)
            },
            "timestamp": timestamp
        }