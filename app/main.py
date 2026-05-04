from config.config_log import setup_logging


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

    print(len(documents))

    for i, doc in enumerate(documents): 
        print(f"--- DOCUMENTO {i} ---")
        print(f"TÍTULO: {doc.metadata.get('section_title', 'Sin título')}")
        print(f"CONTENIDO:\n{doc.page_content}")
        print("-" * 30)