if __name__ == "__main__":
    from rag.loaders.prospecto_loader import ProspectoLoader
    from rag.cleaners.prospecto_cleaner import ProspectoCleaner
    from rag.readers.pdf_reader import PDFReader
    
    pdf_example = "/home/mjesus/Escritorio/Medical Interaction Assistant/data/input_pdfs/acetilcisteina.pdf"

    # Example usage
    prospecto_cleaner = ProspectoCleaner()
    pdf_reader = PDFReader()
    prospecto_loader = ProspectoLoader(reader=pdf_reader, cleaner=prospecto_cleaner, source=pdf_example, cima_id="67763")

    documents = prospecto_loader.create_document()

    print(len(documents))

    for i, doc in enumerate(documents[:3]): # Vemos los primeros 3
        print(f"--- DOCUMENTO {i} ---")
        print(f"CONTENIDO: {doc.page_content[:100]}...") # Primeros 100 caracteres
        print(f"METADATOS: {doc.metadata}")
        print("-" * 30)