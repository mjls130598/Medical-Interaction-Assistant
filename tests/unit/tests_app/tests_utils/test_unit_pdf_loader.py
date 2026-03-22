from unittest.mock import MagicMock, patch

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

        @patch("app.utils.pdf_loader.Path.is_file")
        @patch("app.utils.pdf_loader.fitz.open")
        def test_no_read_pdf(self, mock_fitz_open, mock_path_file, caplog):

            mock_path_file.return_value = True
            mock_fitz_open.side_effect = Exception("RuntimeError: bad PDF")

            pdf_path = "data/input_pdfs/corrupt.pdf"

            pdf_loader = MedicalPDFLoader(pdf_path)
            pdf_loader._read_pdf()

            assert "Error processing" in caplog.text

        @patch("app.utils.pdf_loader.Path.is_file")
        @patch("app.utils.pdf_loader.fitz.open")
        def test_read_pdf(self, mock_fitz_open, mock_path_file):
            pdf_path = "data/input_pdfs/example.pdf"

            mock_path_file.return_value = True

            mock_doc = MagicMock()
            mock_page = MagicMock()

            mock_page.get_text.return_value = [(0, 0, 0, 0, "This is a PDF example", 0, 0)]
            mock_doc.__iter__.return_value = [mock_page]
            mock_fitz_open.return_value.__enter__.return_value = mock_doc

            pdf_loader = MedicalPDFLoader(pdf_path)

            paragraphs = pdf_loader._read_pdf()

            assert len(paragraphs) == 1
            assert "This is a PDF example" in paragraphs 
            mock_page.get_text.assert_called_once_with("blocks")         
            
    class TestCleanBlock:

        @pytest.fixture
        def loader(self):
            return MedicalPDFLoader.__new__(MedicalPDFLoader)

        def test_remove_space_between_sentence(self, loader):

            paragraph = (
                "Lea todo el prospecto detenidamente antes de empezar a tomar este medicamento, porque contiene \n"
                "información importante para usted."
            )

            read_paragraph = loader._clean_block(paragraph)
            real_paragraph = "Lea todo el prospecto detenidamente antes de empezar a tomar este medicamento, porque contiene información importante para usted."

            assert read_paragraph == real_paragraph

        def test_new_lines_between_capital_letters(self, loader):

            paragraph = (
                "Lea todo el prospecto detenidamente antes de empezar a tomar este medicamento, porque contiene \n"
                "información importante para usted.\n"
                "Conserve este prospecto, ya que puede tener que volver a leerlo.")

            read_paragraph = loader._clean_block(paragraph)
            real_paragraph = (
                "Lea todo el prospecto detenidamente antes de empezar a tomar este medicamento, porque contiene información importante para usted.\n"
                "Conserve este prospecto, ya que puede tener que volver a leerlo."
            )

            assert read_paragraph == real_paragraph

        def test_new_lines_between_guion(self, loader):

            paragraph = (
                "Lea todo el prospecto detenidamente antes de empezar a tomar este medicamento, porque contiene \n"
                "información importante para usted.\n"
                "- Conserve este prospecto, ya que puede tener que volver a leerlo.\n"
                "- Si tiene alguna duda, consulte a su médico o farmacéutico.\n"
                "- Este medicamento se le ha recetado solamente a usted, y no debe dárselo a otras personas aunque tengan \n"
                "los mismos síntomas que usted, ya que puede perjudicarles.\n"
                "- Si experimenta efectos adversos, consulte a su médico o farmacéutico, incluso si se trata de efectos \n"
                "adversos que no aparecen en este prospecto. Ver sección 4.")

            read_paragraph = loader._clean_block(paragraph)
            real_paragraph = (
                "Lea todo el prospecto detenidamente antes de empezar a tomar este medicamento, porque contiene información importante para usted.\n"
                "- Conserve este prospecto, ya que puede tener que volver a leerlo.\n"
                "- Si tiene alguna duda, consulte a su médico o farmacéutico.\n"
                "- Este medicamento se le ha recetado solamente a usted, y no debe dárselo a otras personas aunque tengan los mismos síntomas que usted, ya que puede perjudicarles.\n"
                "- Si experimenta efectos adversos, consulte a su médico o farmacéutico, incluso si se trata de efectos adversos que no aparecen en este prospecto. Ver sección 4."
            )

            assert read_paragraph == real_paragraph

        def test_new_lines_between_number_sections(self, loader):

            paragraph = (
                "Lea todo el prospecto detenidamente antes de empezar a tomar este medicamento, porque contiene \n"
                "información importante para usted.\n"
                "1. Conserve este prospecto, ya que puede tener que volver a leerlo.\n"
                "2. Si tiene alguna duda, consulte a su médico o farmacéutico.\n"
                "3. Este medicamento se le ha recetado solamente a usted, y no debe dárselo a otras personas aunque tengan \n"
                "los mismos síntomas que usted, ya que puede perjudicarles.\n"
                "4. Si experimenta efectos adversos, consulte a su médico o farmacéutico, incluso si se trata de efectos \n"
                "adversos que no aparecen en este prospecto. Ver sección 4.")

            read_paragraph = loader._clean_block(paragraph)
            real_paragraph = (
                "Lea todo el prospecto detenidamente antes de empezar a tomar este medicamento, porque contiene información importante para usted.\n"
                "1. Conserve este prospecto, ya que puede tener que volver a leerlo.\n"
                "2. Si tiene alguna duda, consulte a su médico o farmacéutico.\n"
                "3. Este medicamento se le ha recetado solamente a usted, y no debe dárselo a otras personas aunque tengan los mismos síntomas que usted, ya que puede perjudicarles.\n"
                "4. Si experimenta efectos adversos, consulte a su médico o farmacéutico, incluso si se trata de efectos adversos que no aparecen en este prospecto. Ver sección 4."
            )

            assert read_paragraph == real_paragraph

        def test_complete_word(self, loader):
            paragraph = "Lea todo el prospecto deteni-\ndamente"

            read_paragraph = loader._clean_block(paragraph)
            real_paragraph = "Lea todo el prospecto detenidamente"

            assert read_paragraph == real_paragraph

    class TestCreateParagraphs:

        @pytest.fixture
        def loader(self):
            return MedicalPDFLoader.__new__(MedicalPDFLoader)
        
        def test_remove_page_number(self, loader):

            blocks = [
                (0, 0, 0, 0, "1 de 5", 0, 0)
            ]
            cleaned_paragraph = loader._create_paragraphs(blocks)

            assert cleaned_paragraph == []

        def test_remove_no_text_block(self, loader):

            blocks = [
                (0, 0, 0, 0, "no text", 0, 1)
            ]
            cleaned_paragraph = loader._create_paragraphs(blocks)

            assert cleaned_paragraph == []

        def test_remove_empty_block(self, loader):

            blocks = [
                (0, 0, 0, 0, "", 0, 0)
            ]
            cleaned_paragraph = loader._create_paragraphs(blocks)

            assert cleaned_paragraph == []

        def test_concatenate_incomplete_paragraphs(self, loader):
            blocks = [
                [0, 0, 0, 0, "Párrafo incompleto", 0, 0],
                [0, 0, 0, 0, "continuación del texto.", 1, 0],
                [0, 0, 0, 0, "Nuevo párrafo independiente.", 2, 0]
            ]

            def side_effect_clean(text):
                return text
            
            patch.object(MedicalPDFLoader, '_clean_block', side_effect=side_effect_clean)

            cleaned_paragraphs = loader._create_paragraphs(blocks)
            real_paragraphs = [
                "Párrafo incompleto continuación del texto.",
                "Nuevo párrafo independiente."
            ]

            assert cleaned_paragraphs == real_paragraphs

