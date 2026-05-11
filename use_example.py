import asyncio
import os
from dotenv import load_dotenv

from app.config.config_log import setup_logging
from app.rag.loaders.prospecto_loader import ProspectoLoader
from app.rag.cleaners.prospecto_cleaner import ProspectoCleaner
from app.rag.readers.pdf_reader import PDFReader
from app.rag.embedding.document_embedder import DocumentEmbedder
from app.rag.embedding.embedding_strategy.sentence_transformer_strategy import SentenceTransformerStrategy
from app.service.assistance.groq_rag_assistance import GroqRAGAssistance
from app.service.history_store.database import MongoDBClient
from app.service.history_store.history_service import HistoryService
from app.service.history_store.repositories.mongo_history_repo import MongoHistoryRepo
from app.service.vector_store.vectore_store_manager import VectorStoreManager


async def main():
    setup_logging()
    
    pdf_example = "data/prospectos_pdfs/acetilcisteina.pdf"
    load_dotenv(os.path.join(os.path.dirname(__file__), '.', '.env'))

    # Example usage
    prospecto_cleaner = ProspectoCleaner()
    pdf_reader = PDFReader()
    prospecto_loader = ProspectoLoader(reader=pdf_reader,
        cleaner=prospecto_cleaner, source=pdf_example, cima_id="67763")

    documents = prospecto_loader.create_document()

    strategy = SentenceTransformerStrategy(model_name="all-MiniLM-L6-v2")
    document_embedder = DocumentEmbedder(strategy=strategy)
    final_documents = document_embedder.generate_embeddings(documents)

    vector_store_manager = VectorStoreManager()
    vector_store_manager.save_documents(final_documents)

    db_instance = MongoDBClient().medical_db
    history_repository = MongoHistoryRepo(db_client=db_instance)
    history_service = HistoryService(history_repository=history_repository)

    assistent = GroqRAGAssistance(vector_store=vector_store_manager, embedding_strategy=strategy, db_connection=history_service)
    session_id = "test_session_123"
    response = await assistent.ask("¿Cuáles son las indicaciones de la acetilcisteina?", session_id)
    print("Respuesta del asistente:")
    print(response)

if __name__ == "__main__":
    # Run the async main loop
    asyncio.run(main())