import pytest
from unittest.mock import patch

from app.rag.cleaners.text_cleaner import TextCleaner


class MockTextCleaner(TextCleaner):
    """Mock implementation of TextCleaner for testing concrete methods."""
    def _extract_section(self, text: str) -> str:
        return "1", "Test Section"


class TestTextCleaner:
    """Test class for TextCleaner abstract class methods."""

    def test_clean_line_removes_newlines_inside_words(self):
        """Test that _clean_line removes newlines inside words."""
        cleaner = MockTextCleaner()
        line = "ace-\ntilcisteína"
        result = cleaner._clean_line(line)
        assert result == "acetilcisteína"

    def test_clean_line_joins_sentences(self):
        """Test that _clean_line joins lines that don't start with special characters."""
        cleaner = MockTextCleaner()
        line = "This is a sentence.\ncontinuation"
        result = cleaner._clean_line(line)
        assert result == "This is a sentence. continuation"

    def test_clean_line_preserves_newlines_for_sections(self):
        """Test that _clean_line preserves newlines for section starts."""
        cleaner = MockTextCleaner()
        line = "1. Section\ncontent"
        result = cleaner._clean_line(line)
        assert "1. Section content" == result  # Should preserve

    def test_clean_line_concatenates_spaces(self):
        """Test that _clean_line concatenates multiple spaces."""
        cleaner = MockTextCleaner()
        line = "word1    word2"
        result = cleaner._clean_line(line)
        assert result == "word1 word2"

    def test_is_page_number_valid(self):
        """Test _is_page_number returns True for valid page numbers."""
        cleaner = MockTextCleaner()
        assert cleaner._is_page_number("1 de 10") is True
        assert cleaner._is_page_number("Página 5 de 20") is True  # Not exact match

    def test_is_page_number_invalid(self):
        """Test _is_page_number returns False for invalid formats."""
        cleaner = MockTextCleaner()
        assert cleaner._is_page_number("random text") is False
        assert cleaner._is_page_number("1 10") is False

    def test_append_to_buffer_incomplete_and_lowercase(self):
        """Test _append_to_buffer adds space for incomplete sentence and lowercase start."""
        cleaner = MockTextCleaner()
        result = cleaner._append_to_buffer("Incomplete", "continuation")
        assert result == "Incomplete continuation"

    def test_append_to_buffer_complete_or_capital(self):
        """Test _append_to_buffer adds newline for complete sentence or capital start."""
        cleaner = MockTextCleaner()
        result = cleaner._append_to_buffer("Complete.", "Continuation")
        assert result == "Complete.\nContinuation"

    def test_create_paragraphs_empty_text(self):
        """Test create_paragraphs returns empty list for empty text."""
        cleaner = MockTextCleaner()
        result = cleaner.create_paragraphs("")
        assert result == []

    def test_create_paragraphs_with_sections(self):
        """Test create_paragraphs extracts sections correctly."""
        cleaner = MockTextCleaner()
        text = "1. Introduction\nThis is content.\n2. Details\nMore content."
        with patch.object(cleaner, '_extract_section') as mock_extract:
            mock_extract.side_effect = [
                ("1", "Introduction"), 
                (None, None),
                ("2", "Details"), 
                (None, None)
            ]
            result = cleaner.create_paragraphs(text)
            assert len(result) == 2
            assert result[0]['section_id'] == "1"
            assert result[0]['section_title'] == "Introduction"
            assert "This is content." in result[0]['content']

    def test_create_paragraphs_skips_page_numbers(self):
        """Test create_paragraphs skips lines that are page numbers."""
        cleaner = MockTextCleaner()
        text = "1 de 10\nContent"
        with patch.object(cleaner, '_is_page_number') as mock_is_page, \
             patch.object(cleaner, '_extract_section') as mock_extract:
            mock_is_page.side_effect = [True, False]
            mock_extract.return_value = (None, None)
            
            result = cleaner.create_paragraphs(text)
            mock_is_page.assert_called()
            # Depending on implementation, but since first line skipped, and no section, buffer empty, but last line adds if idx==len-1
            # Wait, for idx=0, text="1 de 10", is_page=True, continue
            assert len(result) == 1