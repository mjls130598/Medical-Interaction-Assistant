import pytest
import os
from pathlib import Path
from langchain_core.documents import Document

from app.rag.embedding.document_embedder import DocumentEmbedder
from app.rag.embedding.embedding_strategy.sentence_transformer_strategy import SentenceTransformerStrategy
from app.rag.readers.pdf_reader import PDFReader
from app.rag.cleaners.prospecto_cleaner import ProspectoCleaner
from app.rag.loaders.prospecto_loader import ProspectoLoader
from app.service.vector_store.vectore_store_manager import VectorStoreManager


@pytest.fixture
def embedding_strategy():
    """Fixture providing the sentence transformer embedding strategy."""
    return SentenceTransformerStrategy(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def embedder(embedding_strategy):
    """Fixture providing a DocumentEmbedder configured with strategy."""
    return DocumentEmbedder(strategy=embedding_strategy)


@pytest.fixture
def sample_documents():
    """Fixture providing a small batch of documents for embedding and storage."""
    texts = [
        "Patient education is essential for successful treatment.",
        "Medication must be administered under medical supervision.",
        "Follow-up care includes monitoring and reporting side effects."
    ]
    return [Document(page_content=text, metadata={"document_id": f"doc_{index}"})
            for index, text in enumerate(texts)]


@pytest.fixture
def vector_store(tmp_path):
    """Fixture providing a VectorStoreManager backed by a temporary directory."""
    storage_dir = tmp_path / "vector_store"
    storage_dir.mkdir()
    return VectorStoreManager(path=str(storage_dir), collection_name="prospectos_integration")


@pytest.fixture
def pdf_reader():
    """Fixture providing a PDFReader instance."""
    return PDFReader()


@pytest.fixture
def sample_pdf_path():
    """Fixture providing the path to a sample PDF."""
    return os.path.join(os.path.dirname(__file__), "data", "test_prospecto.pdf")


@pytest.fixture
def prospecto_cleaner():
    """Fixture providing a ProspectoCleaner instance."""
    return ProspectoCleaner()


@pytest.fixture
def prospecto_loader(pdf_reader, prospecto_cleaner, sample_pdf_path):
    """Fixture providing a ProspectoLoader instance with sample PDF."""
    return ProspectoLoader(reader=pdf_reader, cleaner=prospecto_cleaner, source=sample_pdf_path, cima_id="67763")