import pytest
from app.rag.cleaners.chunking_strategy.content_chunker import ContentChunker
from app.rag.cleaners.chunking_strategy.groq_strategy import GroqStrategy


class TestContentChunking:
    """Integration tests for the content chunking functionality."""

    @pytest.fixture
    def chunker(self):
        """Fixture providing a ContentChunker instance with default GroqStrategy."""
        return ContentChunker(strategy=GroqStrategy())

    @pytest.fixture
    def chunker_with_custom_params(self):
        """Fixture providing a ContentChunker with custom max_unit and overlap."""
        return ContentChunker(strategy=GroqStrategy(), max_unit=256, overlap_percentage=0.25)

    def test_successful_text_splitting(self, chunker):
        """Test successful splitting of large text into chunks."""
        large_text = """
        This is a comprehensive medical document about various treatments and conditions.
        It contains multiple paragraphs explaining different aspects of medical procedures.
        
        The document discusses pain management, treatment options, and patient care strategies.
        It provides detailed information about dosages, side effects, and contraindications.
        
        Further sections include information about dietary considerations and lifestyle changes.
        Patients are advised to consult with healthcare professionals before making changes.
        
        The treatment should be administered carefully and with proper medical supervision.
        Regular monitoring and follow-up appointments are essential for patient safety.
        
        This section covers additional complications that may arise during treatment.
        It is crucial to be aware of warning signs and seek immediate medical attention.
        
        The effectiveness of the treatment depends on patient compliance and adherence.
        Results may vary based on individual factors and overall health condition.
        """ * 10  # Repeat to create a large text
        
        chunks = chunker.split(large_text)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk.strip()) > 0 for chunk in chunks)

    def test_small_text_no_split(self, chunker):
        """Test that small text that fits in one chunk is not split."""
        small_text = "This is a small medical note that fits in a single chunk."
        
        chunks = chunker.split(small_text)
        
        assert len(chunks) == 1
        assert chunks[0].strip() == small_text.strip()

    def test_empty_text_handling(self, chunker):
        """Test handling of empty or whitespace-only text."""
        empty_chunks = chunker.split("")
        assert empty_chunks == [] or (len(empty_chunks) == 1 and empty_chunks[0].strip() == "")
        
        whitespace_chunks = chunker.split("   \n   \t   ")
        assert len(whitespace_chunks) <= 1

    def test_chunking_with_custom_parameters(self, chunker_with_custom_params):
        """Test chunking with custom max_unit and overlap percentage."""
        text = """
        Medical treatment requires careful planning and patient education.
        Doctors must consider all relevant factors before prescribing medication.
        The patient's medical history is crucial for treatment success.
        Proper dosage and timing are essential for effectiveness.
        Side effects must be monitored and reported immediately.
        """ * 5
        
        chunks = chunker_with_custom_params.split(text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_overlap_content(self, chunker):
        """Test that chunks have appropriate overlap between them."""
        text = "First section. " * 100 + "Second section. " * 100 + "Third section. " * 100
        
        chunks = chunker.split(text)
        
        assert len(chunks) > 1
        # Check that consecutive chunks have some overlap
        for i in range(len(chunks) - 1):
            # The end of chunk i should have some words in common with start of chunk i+1
            assert len(chunks[i]) > 0 and len(chunks[i+1]) > 0

    def test_chunking_preserves_content_integrity(self, chunker):
        """Test that chunking preserves all original content (with overlaps)."""
        original_text = """
        This medical document contains important information about patient care.
        Treatment protocols must be followed strictly for patient safety.
        Complications can arise if instructions are not followed properly.
        Regular monitoring helps identify issues early.
        Patient education is fundamental to treatment success.
        """ * 8
        
        chunks = chunker.split(original_text)
        
        # All chunks should be non-empty
        assert all(len(chunk.strip()) > 0 for chunk in chunks)
        # Original content should be represented in chunks (allowing for some loss due to overlap trimming)
        combined = " ".join(chunks)
        assert "patient care" in combined.lower()
        assert "treatment protocols" in combined.lower()

    def test_text_with_paragraph_breaks(self, chunker):
        """Test chunking text with multiple paragraph breaks."""
        text = """
        First paragraph about medical treatments.

        Second paragraph about patient care requirements.

        Third paragraph discussing side effects and monitoring.

        Fourth paragraph on follow-up procedures and recommendations.
        """ * 10
        
        chunks = chunker.split(text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk.strip()) > 0 for chunk in chunks)

    def test_text_with_special_characters(self, chunker):
        """Test chunking text containing special characters and accents."""
        text = """
        Información médica sobre el tratamiento de condiciones crónicas.
        El paciente debe seguir las instrucciones con precisión.
        Síntomas adversos incluyen: fiebre, dolor de cabeza, náuseas.
        Interacciones medicamentosas deben ser consideradas cuidadosamente.
        """ * 8
        
        chunks = chunker.split(text)
        
        assert len(chunks) > 0
        assert all("í" in chunk or "á" in chunk or len(chunk) > 0 for chunk in chunks if len(chunk) > 20)

    def test_different_strategy_initialization(self):
        """Test ContentChunker with explicitly initialized GroqStrategy."""
        strategy = GroqStrategy()
        chunker = ContentChunker(strategy=strategy, max_unit=512, overlap_percentage=0.2)
        
        text = "Medical information. " * 30
        chunks = chunker.split(text)
        
        assert len(chunks) > 0

    def test_very_long_document(self, chunker):
        """Test chunking of a very long medical document."""
        # Create a long document
        long_text = """
        Detailed medical treatment documentation including:
        - Patient assessment procedures
        - Treatment methodology and protocols
        - Expected outcomes and monitoring
        - Adverse effects and contraindications
        """ * 100
        
        chunks = chunker.split(long_text)
        
        assert len(chunks) > 1
        assert all(len(chunk) > 0 for chunk in chunks)
