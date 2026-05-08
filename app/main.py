from config.config_log import setup_logging
from rag.embedding.document_embedder import DocumentEmbedder
from rag.embedding.embedding_strategy.sentence_transformer_strategy import SentenceTransformerStrategy


if __name__ == "__main__":
    setup_logging()
    
    from rag.loaders.prospecto_loader import ProspectoLoader
    from rag.cleaners.prospecto_cleaner import ProspectoCleaner
    from rag.readers.pdf_reader import PDFReader
    
    pdf_example = "data/input_pdfs/acetilcisteina.pdf"

    # Example usage
    prospecto_cleaner = ProspectoCleaner()
    pdf_reader = PDFReader()
    prospecto_loader = ProspectoLoader(reader=pdf_reader,
        cleaner=prospecto_cleaner, source=pdf_example, cima_id="67763")

    documents = prospecto_loader.create_document()

    strategy = SentenceTransformerStrategy(model_name="all-MiniLM-L6-v2")
    embedder = DocumentEmbedder(strategy=strategy)
    final_documents = embedder.generate_embeddings(documents)

    print(len(final_documents))

    for i, doc in enumerate(final_documents): 
        print(f"--- DOCUMENTO {i} ---")
        print(f"METADATOS: {doc.metadata}")
        print(f"CONTENIDO:\n{doc.page_content}")
        print("-" * 30)