import chromadb
from chromadb.config import Settings
import pandas as pd
class Memory:
    def __init__(self, lm, collection_name="escargot_memory"):
        # Initialize ChromaDB client and specify the collection for storing vectors
        self.client = chromadb.PersistentClient(path="./escargot_memory", settings=Settings(allow_reset=True))
        self.collection_name = collection_name
        self.lm = lm

        self.collection = self.client.get_or_create_collection(collection_name)

    def reset_collection(self):
        self.client.reset() 

    def store_memory(self, text, metadata={None:None},collection_name = None):
        # Embed the text using the lm's embed function
        vector = self.lm.get_embedding(text)

        # Add the embedded vector to the collection
        if collection_name is None:
            self.collection.add(ids=text, embeddings=[vector], metadatas=[metadata])
        else:
            collection = self.client.get_collection(collection_name)
            collection.add(ids=text, embeddings=[vector], metadatas=[metadata])

    def query_collection(self,query, max_results=10, collection_name = "escargot_memory", metadata = None):
        collection = self.client.get_collection(collection_name)
        query_embeddings = self.lm.get_embedding(query)
        if metadata is not None:
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=max_results,
                where=metadata,
                include=['distances']
            )
        else:
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=max_results,
                include=['distances']
            )
        return results

    def get_all_vectors(self):
        # Return all vectors stored in the collection
        return self.collection.get()

    def delete_vector(self, text):
        # Delete a vector by id (text)
        self.collection.delete(ids=text)