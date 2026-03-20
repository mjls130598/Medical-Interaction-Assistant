from pathlib import Path

import pytest

from app.utils.pdf_loader import MedicalPDFLoader

class TestPdfLoader:

    @pytest.fixture(scope="class", autouse=True)
    def setup_base_path(self, request):
        # Asignamos el atributo a la clase para que las subclases lo vean
        request.cls.base_path = Path(__file__).parent.parent.parent

    class TestInit:

        def test_init_not_exists(self):
            file_path = str(self.base_path / "data" / "Example.pdf")

            with pytest.raises(FileNotFoundError) as file_error:
                MedicalPDFLoader(file_path)

                assert "doesn't exist" in str(file_error.value)

        def test_init_not_file(self):            
            file_path = str(self.base_path / "data")

            with pytest.raises(FileNotFoundError) as file_error:
                MedicalPDFLoader(file_path)

                assert "doesn't exist" in str(file_error.value)

        def test_init_not_pdf(self):
            file_path = str(self.base_path / "data" / "example.txt")

            with pytest.raises(ValueError) as value_error:
                MedicalPDFLoader(file_path)

                assert "isn't a PDF file" in str(value_error.value)

        def test_init_pdf_file(self):
            file_path = str(self.base_path / "data" / "example.pdf")

            pdfLoader = MedicalPDFLoader(file_path)
            assert hasattr(pdfLoader, "file_path")
            assert pdfLoader.file_path == file_path

    class TestReadPdf:

        def test_no_read_pdf(self, caplog):
            pdf_path = str(self.base_path / "data" / "corrupt_example.pdf")

            pdf_loader = MedicalPDFLoader(pdf_path)
            paragraphs = pdf_loader._read_pdf()

            assert "Error processing" in caplog.text


        def test_read_pdf(self):
            pdf_path = str(self.base_path / "data" / "example.pdf")

            pdf_loader = MedicalPDFLoader(pdf_path)
            paragraphs = pdf_loader._read_pdf()

            assert len(paragraphs) == 1
            assert "This is a PDF example" in paragraphs