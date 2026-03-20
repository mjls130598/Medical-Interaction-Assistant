import pytest

from app.utils.pdf_loader import MedicalPDFLoader

class TestPdfLoader:

    class TestInit:

        def test_init_not_exists(self, fs):
            file_path = "Ejemplo.pdf"

            with pytest.raises(FileNotFoundError) as file_error:
                MedicalPDFLoader(file_path)

                assert "doesn't exist" in str(file_error.value)


        def test_init_not_file(self, fs):
            
            file_path = "/data/inputs_pdfs/"
            fs.create_file(file_path)


            with pytest.raises(ValueError) as value_error:
                MedicalPDFLoader(file_path)

                assert "isn't a PDF file" in str(value_error.value)

        def test_init_not_pdf(self, fs):
            
            file_path = "/data/inputs_pdfs/ejemplo.csv"
            fs.create_file(file_path)


            with pytest.raises(ValueError) as value_error:
                MedicalPDFLoader(file_path)

                assert "isn't a PDF file" in str(value_error.value)

        def test_init_pdf_file(self, fs):

            file_path = "/data/inputs_pdfs/Ejemplo.pdf"
            fs.create_file(file_path)

            pdfLoader = MedicalPDFLoader(file_path)
            assert hasattr(pdfLoader, "file_path")
            assert pdfLoader.file_path == file_path
