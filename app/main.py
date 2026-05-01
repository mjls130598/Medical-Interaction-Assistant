from rag.pdf_loader import MedicalPDFLoader
from services.medical_consultant import MedicalConsultant
from services.vector_db import MedicalVectorDB


if __name__ == "__main__":
    pdf_example = "/home/mjesus/Escritorio/Medical Interaction Assistant/data/input_pdfs/acetilcisteina.pdf" 

    pdf_loader = MedicalPDFLoader(pdf_example)
    documents = pdf_loader.read_load_document()
    vector_db = MedicalVectorDB()
    vector_db.add_documents(documents)

    consultant = MedicalConsultant(vector_db)
    question = "¿Cuáles son las indicaciones de la acetilcisteina?"
    answer = consultant.ask_question(question)
    print("Respuesta del asistente médico:")
    print(answer)