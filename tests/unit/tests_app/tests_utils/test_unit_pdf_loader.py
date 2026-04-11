from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

import pytest

from app.utils.pdf_loader import MedicalPDFLoader

class TestPdfLoader:

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------
    class TestInit:

        def test_init_not_exists(self, fs):
            """Check that FileNotFoundError is raised when the file path does not exist."""
            file_path = "NonExistent.pdf"
            with pytest.raises(FileNotFoundError) as file_error:
                MedicalPDFLoader(file_path)
            assert "doesn't exist" in str(file_error.value)

        def test_init_not_file(self, fs):
            """Check that FileNotFoundError is raised when the path provided is a directory, not a file."""
            dir_path = "/data/inputs_pdfs/"
            fs.create_dir(dir_path)
            with pytest.raises(FileNotFoundError) as file_error:
                MedicalPDFLoader(dir_path)
            assert "doesn't exist" in str(file_error.value)

        def test_init_not_pdf(self, fs):
            """Check that ValueError is raised when the file exists but does not have a .pdf extension."""
            file_path = "/data/inputs_pdfs/ejemplo.csv"
            fs.create_file(file_path)
            with pytest.raises(ValueError) as value_error:
                MedicalPDFLoader(file_path)
            assert "isn't a PDF file" in str(value_error.value)

        def test_init_pdf_file(self, fs):
            """Check that the class initializes correctly when a valid PDF file path is provided."""
            file_path = "/data/inputs_pdfs/Ejemplo.pdf"
            fs.create_file(file_path)
            pdf_loader = MedicalPDFLoader(file_path)
            assert pdf_loader.file_path == file_path

    # -------------------------------------------------------------------------
    # Auxiliary Method: _clean_block
    # -------------------------------------------------------------------------
    class TestCleanBlock:
        
        @pytest.fixture
        def loader(self, fs):
            fs.create_file("dummy.pdf")
            return MedicalPDFLoader("dummy.pdf")

        def test_remove_space_between_sentence(self, loader):
            """Check that a newline is replaced by a space when it occurs mid-sentence (lowercase follow-up)."""
            paragraph = "Contiene \ninformación importante."
            assert loader._clean_block(paragraph) == "Contiene información importante."

        def test_complete_word(self, loader):
            """Check that hyphenated words split by a newline are correctly joined back together."""
            paragraph = "Es un deteni-\ndamente."
            assert loader._clean_block(paragraph) == "Es un detenidamente."

        def test_clean_block_bullet_points(self, loader):
            """Check that lines starting with bullets are NOT joined to ensure list formatting is preserved."""
            block = "Síntomas:\n- Mareo\n- Náuseas"
            assert loader._clean_block(block) == "Síntomas:\n- Mareo\n- Náuseas"

        def test_clean_block_multiple_spaces(self, loader):
            """Check that multiple consecutive spaces are reduced to a single space."""
            block = "Texto    con  demasiados     espacios."
            assert loader._clean_block(block) == "Texto con demasiados espacios."

        def test_clean_block_capital_letters(self, loader):
            """Check that newlines before capital letters are preserved as they likely indicate new sentences."""
            block = "Fin de frase.\nNueva frase."
            assert "Fin de frase.\nNueva frase" in loader._clean_block(block)

    # -------------------------------------------------------------------------
    # Auxiliary Method: _extract_section
    # -------------------------------------------------------------------------
    class TestExtractSection:

        @pytest.fixture
        def loader(self, fs):
            fs.create_file("dummy.pdf")
            return MedicalPDFLoader("dummy.pdf")

        def test_extract_section_valid(self, loader):
            """Check if a standard section header (number + title) is correctly parsed."""
            text = "1.2. POSOLOGÍA Y ADMINISTRACIÓN"
            sec_id, sec_title = loader._extract_section(text)
            assert sec_id == "1.2"
            assert sec_title == "POSOLOGÍA Y ADMINISTRACIÓN"

        def test_extract_section_no_dot(self, loader):
            """Check if headers without a trailing dot after the number are correctly identified."""
            text = "3 MEDICAMENTOS"
            sec_id, sec_title = loader._extract_section(text)
            assert sec_id == "3"
            assert sec_title == "MEDICAMENTOS"

        def test_extract_section_invalid(self, loader):
            """Check that regular text without section numbering returns None for both values."""
            text = "Este es un párrafo de texto normal."
            sec_id, sec_title = loader._extract_section(text)
            assert sec_id is None
            assert sec_title is None

    # -------------------------------------------------------------------------
    # Auxiliary Method: _create_paragraphs
    # -------------------------------------------------------------------------
    class TestCreateParagraphs:

        @pytest.fixture
        def loader(self, fs):
            fs.create_file("dummy.pdf")
            return MedicalPDFLoader("dummy.pdf")

        def test_remove_page_number(self, loader):
            """Check that strings matching the page pattern (e.g., '1 de 5') are ignored."""
            blocks = [(0, (0, 0, 0, 0, "1 de 5", 0, 0))]
            assert loader._create_paragraphs(blocks) == []

        def test_remove_no_text_block(self, loader):
            """Check that non-text blocks (images/graphics with type != 0) are skipped."""
            blocks = [(0, (0, 0, 0, 0, "Image", 0, 1))]
            assert loader._create_paragraphs(blocks) == []

        def test_concatenate_incomplete_paragraphs(self, loader):
            """Check that a paragraph starting with a lowercase letter is merged with the previous one."""
            blocks = [
                (0, (0, 0, 0, 0, "La dosis es", 0, 0)),
                (0, (0, 0, 0, 0, "muy baja.", 0, 0))
            ]
            paragraphs = loader._create_paragraphs(blocks)
            assert len(paragraphs) == 1
            assert paragraphs[0]["content"] == "La dosis es muy baja."

        def test_section_context_update(self, loader):
            """Check that finding a new section updates the context for all following paragraphs."""
            blocks = [
                (0, (0, 0, 0, 0, "1. TITULO", 0, 0)),
                (0, (0, 0, 0, 0, "Contenido.", 1, 0))
            ]
            paragraphs = loader._create_paragraphs(blocks)
            assert paragraphs[0]["section_id"] == "1"
            assert paragraphs[0]["section_title"] == "TITULO"

    # -------------------------------------------------------------------------
    # Method: _read_pdf
    # -------------------------------------------------------------------------
    class TestReadPdf:

        @patch("app.utils.pdf_loader.fitz.open")
        def test_read_pdf_error(self, mock_fitz_open, fs, caplog):
            """Check that errors during PDF opening are logged and return None."""
            fs.create_file("corrupt.pdf")
            mock_fitz_open.side_effect = Exception("bad PDF")
            loader = MedicalPDFLoader("corrupt.pdf")
            result = loader._read_pdf()
            assert "Error processing" in caplog.text
            assert result is None

        @patch("app.utils.pdf_loader.fitz.open")
        def test_read_pdf_success(self, mock_fitz_open, fs):
            """Check that the PDF content is correctly grouped by section and page count is accurate."""
            fs.create_file("test.pdf")
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_page.get_text.return_value = [(0, 0, 0, 0, "1. Sección A", 0, 0), (0, 0, 0, 0, "Texto.", 0, 0)]
            mock_doc.__iter__.return_value = [mock_page]
            mock_doc.__len__.return_value = 1
            mock_fitz_open.return_value.__enter__.return_value = mock_doc

            loader = MedicalPDFLoader("test.pdf")
            sections, total = loader._read_pdf()
            assert total == 1
            assert sections[0]["section_id"] == "1"
            assert "Texto." in sections[0]["content"]

    # -------------------------------------------------------------------------
    # Method: read_load_document
    # -------------------------------------------------------------------------
    class TestReadLoadDocument:

        @patch("app.utils.pdf_loader.MetadataAPI")
        @patch("app.utils.pdf_loader.re.search")
        def test_read_load_document_success(self, mock_re_search, mock_metadata_api, fs):
            """Check that LangChain Document objects are created with correct content and combined metadata."""
            fs.create_file("test.pdf")
            loader = MedicalPDFLoader("test.pdf")
            
            mock_match = MagicMock()
            mock_match.group.return_value = "12345"
            mock_re_search.return_value = mock_match
            mock_metadata_api.return_value.fetch_metadata.return_value = {"medicine": "X"}

            fake_sections = [("1", {"page_num": 1, "section_id": "1", "section_title": "S1", "content": "URL: 12345"})]

            with patch.object(MedicalPDFLoader, '_read_pdf', return_value=(fake_sections, 1)):
                docs = loader.read_load_document()
                assert len(docs) == 1
                assert docs[0].metadata["medicine"] == "X"
                assert isinstance(docs[0], Document)

        def test_read_load_document_empty_error(self, fs):
            """Check that a ValueError is raised if no content is extracted from the PDF."""
            fs.create_file("empty.pdf")
            loader = MedicalPDFLoader("empty.pdf")
            with patch.object(MedicalPDFLoader, '_read_pdf', return_value=([], 0)):
                with pytest.raises(ValueError, match="There isn't information to save"):
                    loader.read_load_document()