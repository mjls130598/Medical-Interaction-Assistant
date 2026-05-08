from typing import List

import chromadb
from langchain_core.documents import Document


class VectorStoreManager:

    def __init__(self, path: str = "./vector_store", collection_name: str = "prospectos"):
        """
        Initialize the VectorStoreManager with a specific path and collection name.
        
        Arguments:
            path (str): The directory where the vector store will be saved.
            collection_name (str): The name of the collection to store the vectors.
        """
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def save_documents(self, documents: List[Document]):
        """
        Save a list of documents with their embeddings to the vector store.

        Arguments:
            documents (List[Document]): A list of Document objects, 
                                        each containing metadata with an "embedding" key.
        """

        self.collection.upsert(
            ids=[doc.metadata["document_id"] for doc in documents],
            embeddings=[doc.metadata["embedding"] for doc in documents],
            metadatas=[doc.metadata for doc in documents],
            documents=[doc.page_content for doc in documents]
        )

    def search_relevant_chunks(self, query_embedding: List[float], n_results: int = 5) -> List[str]:
        """
        Search for the most relevant document chunks based on a query embedding.

        Arguments:
            query_embedding (List[float]): The embedding vector of the query.
            n_results (int): The number of top relevant results to return.
        
        Returns:
            List[str]: A list of the most relevant document contents.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results