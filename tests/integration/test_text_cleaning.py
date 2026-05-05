import pytest
from app.rag.cleaners.prospecto_cleaner import ProspectoCleaner


class TestTextCleaning:
    """Integration tests for the text cleaning functionality."""

    @pytest.fixture
    def cleaner(self):
        """Fixture providing a ProspectoCleaner instance."""
        return ProspectoCleaner()

    def test_successful_section_creation(self, cleaner):
        """Test successful creation of sections from well-formed text."""
        sample_text = """1. INDICACIONES
                        This is the indications section.

                        2. CONTRAINDICACIONES
                        This is contraindications.
                        Some more text here.

                        3. DOSIFICACIÓN
                        Dosage information."""

        sections = cleaner.create_sections(sample_text)

        assert len(sections) == 3
        assert sections[0]['section_id'] == '1'
        assert sections[0]['section_title'] == 'INDICACIONES'
        assert 'indications section' in sections[0]['content'].lower()

    def test_empty_text_handling(self, cleaner):
        """Test handling of empty or whitespace-only text."""
        sections = cleaner.create_sections("")
        assert sections == []

        sections = cleaner.create_sections("   \n   ")
        assert sections == []

    def test_no_sections_found(self, cleaner):
        """Test text without recognizable section headers."""
        sample_text = "This is some random text without sections."
        sections = cleaner.create_sections(sample_text)

        # Should create one section with default section
        assert len(sections) == 1
        assert sections[0]['section_id'] == '0'
        assert sections[0]['section_title'] == 'Introducción'

    def test_malformed_section_headers(self, cleaner):
        """Test handling of malformed section headers."""
        sample_text = """1. INDICACIONES
                        Content here.

                        2. CONTRAINDICACIONES
                        More content."""

        sections = cleaner.create_sections(sample_text)

        assert len(sections) == 2  

    def test_line_cleaning_integration(self, cleaner):
        """Test that line cleaning is applied during section creation."""
        sample_text = """1. INDICACIONES
                        This is a line with multiple   spaces.
                        Another line- 
                        broken."""

        sections = cleaner.create_sections(sample_text)

        content = sections[0]['content']
        assert 'multiple   spaces' not in content  # Should be cleaned to single space
        assert 'broken' in content  # Should be joined</content>
