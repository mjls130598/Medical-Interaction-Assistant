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

    def is_empty(self) -> bool:
        """
        Check if the vector store collection is empty.

        Returns:
            bool: True if the collection has no documents, False otherwise.
        """
        return self.collection.count() == 0

    def save_documents(self, documents: List[Document]):
        """
        Save a list of documents with their embeddings to the vector store.

        Arguments:
            documents (List[Document]): A list of Document objects, 
                                        each containing metadata with an "embedding" key.
        """

        ids = []
        embeddings = []
        clean_metadatas = []
        page_contents = []

        for doc in documents:
            ids.append(doc.metadata["document_id"])
            embeddings.append(doc.metadata["embedding"])
            
            # Creamos una copia de la metadata SIN el embedding
            meta = doc.metadata.copy()
            meta.pop("embedding", None) # Eliminamos la clave 'embedding' si existe
            clean_metadatas.append(meta)
            
            page_contents.append(doc.page_content)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=clean_metadatas, 
            documents=page_contents
        )

    def search_relevant_chunks(self, query_embedding: List[float], n_results: int = 5) -> List[tuple[str, dict]]:
        """
        Search for the most relevant document chunks based on a query embedding.

        Arguments:
            query_embedding (List[float]): The embedding vector of the query.
            n_results (int): The number of top relevant results to return.
        
        Returns:
            List[tuple[str, dict]]: A list of the most relevant document contents.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results['documents'][0], results['metadatas'][0]