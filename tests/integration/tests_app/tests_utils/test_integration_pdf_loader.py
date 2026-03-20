import pytest

from app.utils.pdf_loader import MedicalPDFLoader

class TestPdfLoader:

    class TestInit:

        def test_init_not_exists(self, fs):
            file_path = "/home/mjesus/Escritorio/Medical Interaction Assistant/tests/data/Example.pdf"

            with pytest.raises(FileNotFoundError) as file_error:
                MedicalPDFLoader(file_path)

                assert "doesn't exist" in str(file_error.value)


        def test_init_not_file(self, fs):
            
            file_path = "/home/mjesus/Escritorio/Medical Interaction Assistant/tests/data"
            fs.create_file(file_path)


            with pytest.raises(ValueError) as value_error:
                MedicalPDFLoader(file_path)

                assert "isn't a PDF file" in str(value_error.value)

        def test_init_not_pdf(self, fs):
            
            file_path = "/home/mjesus/Escritorio/Medical Interaction Assistant/tests/data/example.txt"
            fs.create_file(file_path)


            with pytest.raises(ValueError) as value_error:
                MedicalPDFLoader(file_path)

                assert "isn't a PDF file" in str(value_error.value)

        def test_init_pdf_file(self):
            file_path = "/home/mjesus/Escritorio/Medical Interaction Assistant/tests/data/example.pdf"

            pdfLoader = MedicalPDFLoader(file_path)
            assert hasattr(pdfLoader, "file_path")
            assert pdfLoader.file_path == file_path