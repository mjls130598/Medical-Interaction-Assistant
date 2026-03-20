from pathlib import Path

import fitz
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
            fs.create_dir(file_path)

            with pytest.raises(FileNotFoundError) as file_error:
                MedicalPDFLoader(file_path)

                assert "doesn't exist" in str(file_error.value)

        def test_init_not_pdf(self, fs):
            
            file_path = "/data/inputs_pdfs/ejemplo.csv"
            fs.create_file(file_path)


            with pytest.raises(ValueError) as value_error:
                MedicalPDFLoader(file_path)

                assert "isn't a PDF file" in str(value_error.value)

        def test_init_pdf_file(self, fs):

            file_path = "/data/inputs_pdfs/Ejemplo.pdf"
            fs.create_file(file_path)

            pdf_loader = MedicalPDFLoader(file_path)
            assert hasattr(pdf_loader, "file_path")
            assert pdf_loader.file_path == file_path

    class TestReadPdf:

        def _create_fake_pdf(self, pdf_path, fs):  
            # Creation of PDF file
            doc_temp = fitz.open()
            page = doc_temp.new_page()
            page.insert_text((50, 50), "This is a PDF example")
            pdf_bytes = doc_temp.tobytes()
            doc_temp.save(pdf_path)
            doc_temp.close()

            # Creation of PDF path
            fs.create_file(pdf_path, contents=pdf_bytes)

        def _create_corrupt_pdf(self, pdf_path, fs):
            fs.create_file(pdf_path, contents="Corrupt PDF!")

        def test_no_read_pdf(self, fs, caplog):
            pdf_path = "data/input_pdfs/corrupt.pdf"
            self._create_corrupt_pdf(pdf_path, fs)

            pdf_loader = MedicalPDFLoader(pdf_path)
            paragraphs = pdf_loader._read_pdf()

            assert "Error processing" in caplog.text


        def test_read_pdf(self, fs):
            pdf_path = "data/input_pdfs/example.pdf"

            self._create_fake_pdf(pdf_path, fs)

            pdf_loader = MedicalPDFLoader(pdf_path)
            paragraphs = pdf_loader._read_pdf()

            assert len(paragraphs) == 1
            assert "This is a PDF example" in paragraphs          
            
