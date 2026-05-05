import pytest
from unittest.mock import patch

from app.rag.cleaners.prospecto_cleaner import ProspectoCleaner


class TestProspectoCleaner:
    """Test class for ProspectoCleaner methods."""

    def test_extract_section_valid(self):
        """Test _extract_section extracts valid section."""
        cleaner = ProspectoCleaner()
        text = "1. Introduction\nContent"
        result = cleaner._extract_section(text)
        assert result == ("1", "Introduction")

    def test_extract_section_with_subsection(self):
        """Test _extract_section with subsection numbers."""
        cleaner = ProspectoCleaner()
        text = "1.2. Subsection\nContent"
        result = cleaner._extract_section(text)
        assert result == ("1.2", "Subsection")

    def test_extract_section_no_match(self):
        """Test _extract_section returns None for no match."""
        cleaner = ProspectoCleaner()
        text = "No section here"
        result = cleaner._extract_section(text)
        assert result == (None, None)

    def test_extract_section_multiline(self):
        """Test _extract_section with multiline text."""
        cleaner = ProspectoCleaner()
        text = "Some text\n1. Section\nMore text"
        result = cleaner._extract_section(text)
        assert result == ("1", "Section")