from pathlib import Path
from unittest.mock import patch

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

    class TestCleanBlock:

        def test_remove_space_between_sentence(self):

            pdf_path = str(self.base_path / "data" / "separate_sentences.pdf")

            pdf_loader = MedicalPDFLoader(pdf_path)

            with patch.object(MedicalPDFLoader, '_clean_block', wraps=pdf_loader._clean_block) as spy_clean:
                read_paragraph = pdf_loader._read_pdf()[0]
                real_paragraph = (
                    "Lea todo el prospecto detenidamente antes de empezar a tomar este medicamento, porque contiene información importante para usted."
                )
                
                assert read_paragraph == real_paragraph
                assert spy_clean.called
                assert spy_clean.call_count == 1

        def test_new_lines_between_capital_letters(self):

            pdf_path = str(self.base_path / "data" / "new_lines_capital.pdf")

            pdf_loader = MedicalPDFLoader(pdf_path)

            with patch.object(MedicalPDFLoader, '_clean_block', wraps=pdf_loader._clean_block) as spy_clean:
                read_paragraph = pdf_loader._read_pdf()[0]
            
                real_paragraph = (
                    "Lea todo el prospecto detenidamente antes de empezar a tomar este medicamento, porque contiene información importante para usted.\n"
                    "Conserve este prospecto, ya que puede tener que volver a leerlo."
                )

                assert read_paragraph == real_paragraph
                assert spy_clean.called
                assert spy_clean.call_count == 1

        def test_new_lines_between_guion(self):

            pdf_path = str(self.base_path / "data" / "new_lines_guion.pdf")

            pdf_loader = MedicalPDFLoader(pdf_path)

            with patch.object(MedicalPDFLoader, '_clean_block', wraps=pdf_loader._clean_block) as spy_clean:
                read_paragraph = pdf_loader._read_pdf()[0]
            
                real_paragraph = (
                    "Lea todo el prospecto detenidamente antes de empezar a tomar este medicamento, porque contiene información importante para usted.\n"
                    "- Conserve este prospecto, ya que puede tener que volver a leerlo.\n"
                    "- Si tiene alguna duda, consulte a su médico o farmacéutico.\n"
                    "- Este medicamento se le ha recetado solamente a usted, y no debe dárselo a otras personas aunque tengan los mismos síntomas que usted, ya que puede perjudicarles.\n"
                    "- Si experimenta efectos adversos, consulte a su médico o farmacéutico, incluso si se trata de efectos adversos que no aparecen en este prospecto. Ver sección 4."
                )

                assert read_paragraph == real_paragraph
                assert spy_clean.called
                assert spy_clean.call_count == 1

        def test_new_lines_between_number_sections(self):

            pdf_path = str(self.base_path / "data" / "new_lines_number_section.pdf")

            pdf_loader = MedicalPDFLoader(pdf_path)

            with patch.object(MedicalPDFLoader, '_clean_block', wraps=pdf_loader._clean_block) as spy_clean:
                read_paragraph = pdf_loader._read_pdf()[0]
            
                real_paragraph = (
                    "Lea todo el prospecto detenidamente antes de empezar a tomar este medicamento, porque contiene información importante para usted.\n"
                    "1. Conserve este prospecto, ya que puede tener que volver a leerlo.\n"
                    "2. Si tiene alguna duda, consulte a su médico o farmacéutico.\n"
                    "3. Este medicamento se le ha recetado solamente a usted, y no debe dárselo a otras personas aunque tengan los mismos síntomas que usted, ya que puede perjudicarles.\n"
                    "4. Si experimenta efectos adversos, consulte a su médico o farmacéutico, incluso si se trata de efectos adversos que no aparecen en este prospecto. Ver sección 4."
                )

                assert read_paragraph == real_paragraph
                assert spy_clean.called
                assert spy_clean.call_count == 1

        def test_complete_word(self):

            pdf_path = str(self.base_path / "data" / "complete_word.pdf")

            pdf_loader = MedicalPDFLoader(pdf_path)

            with patch.object(MedicalPDFLoader, '_clean_block', wraps=pdf_loader._clean_block) as spy_clean:
                read_paragraph = pdf_loader._read_pdf()[0]
            
                real_paragraph = "Lea todo el prospecto detenidamente"

                assert read_paragraph == real_paragraph
                assert spy_clean.called
                assert spy_clean.call_count == 1

    class TestCreateParagraphs:

        def test_remove_page_number(self):

            pdf_path = str(self.base_path / "data" / "number_page.pdf")

            pdf_loader = MedicalPDFLoader(pdf_path)

            with patch.object(MedicalPDFLoader, '_create_paragraphs', wraps=pdf_loader._create_paragraphs) as spy_create:
                read_paragraph = pdf_loader._read_pdf()
            
                assert read_paragraph == []
                assert spy_create.called
                assert spy_create.call_count == 1

        def test_remove_no_text_block(self):

            pdf_path = str(self.base_path / "data" / "no_text_blocks.pdf")

            pdf_loader = MedicalPDFLoader(pdf_path)

            with patch.object(MedicalPDFLoader, '_create_paragraphs', wraps=pdf_loader._create_paragraphs) as spy_create:
                read_paragraph = pdf_loader._read_pdf()
            
                assert read_paragraph == []
                assert spy_create.called
                assert spy_create.call_count == 1

        def test_remove_empty_block(self):

            pdf_path = str(self.base_path / "data" / "empty_blocks.pdf")

            pdf_loader = MedicalPDFLoader(pdf_path)

            with patch.object(MedicalPDFLoader, '_create_paragraphs', wraps=pdf_loader._create_paragraphs) as spy_create:
                read_paragraph = pdf_loader._read_pdf()
            
                assert read_paragraph == []
                assert spy_create.called
                assert spy_create.call_count == 1

        def test_concatenate_incomplete_paragraphs(self):
            pdf_path = str(self.base_path / "data" / "incomplete_paragraphs.pdf")

            pdf_loader = MedicalPDFLoader(pdf_path)

            with patch.object(MedicalPDFLoader, '_create_paragraphs', wraps=pdf_loader._create_paragraphs) as spy_create:
                with patch.object(MedicalPDFLoader, '_clean_block', wraps=pdf_loader._clean_block) as spy_clean:
                    read_paragraph = pdf_loader._read_pdf()
                    real_paragraphs = [
                        "Párrafo incompleto continuación del texto.",
                        "Nuevo párrafo independiente."
                    ]
                
                    assert read_paragraph == real_paragraphs
                    assert spy_create.called
                    assert spy_create.call_count == 1
                    assert spy_clean.called
                    assert spy_clean.call_count == 3