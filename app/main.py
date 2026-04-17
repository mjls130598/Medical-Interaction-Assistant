from rag.pdf_loader import MedicalPDFLoader


if __name__ == "__main__":
    pdf_example = "/home/mjesus/Escritorio/Medical Interaction Assistant/data/input_pdfs/acetilcisteina.pdf" 

    pdf_loader = MedicalPDFLoader(pdf_example)
    documents = pdf_loader.read_load_document()
    for doc in documents:
        print(doc.metadata)
        print(doc.page_content)
        print("\n\n")