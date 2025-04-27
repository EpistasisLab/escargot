import chromadb
from chromadb.config import Settings
import pandas as pd
import networkx as nx
import uuid
import pickle
import os
from raphtory import Graph
from raphtory import algorithms as algo
import shutil
class Memory:
    def __init__(self, lm, collection_name="default"):
        # Initialize ChromaDB client and specify the collection for storing vectors
        #create the collection folder if it doesn't exist
        if not os.path.exists(f"./escargot_memory"):
            os.makedirs(f"./escargot_memory")
        self.client = chromadb.PersistentClient(path=f"./escargot_memory/{collection_name}", settings=Settings(allow_reset=True))
        self.collection_name = collection_name
        self.lm = lm
        self.collection = self.client.get_or_create_collection(collection_name)
        #if graph exists load it
        if os.path.exists(f"./escargot_memory/{collection_name}/graph"):
            self.graph = Graph.load_from_file(f"./escargot_memory/{collection_name}/graph")
        else:
            self.graph = Graph()
            self.save_graph()

        #history of when nodes were inserted
        self.graph_timestamp = 1
            

    def reset_collection(self, collection_name = None):
        if collection_name is None:
            collection_name = self.collection_name
        
        self.client.delete_collection(name=collection_name)

        self.client.reset()
        self.client.clear_system_cache() # very important
        self.client = None

        if os.path.exists(f"./escargot_memory/{collection_name}"):
            shutil.rmtree(f"./escargot_memory/{collection_name}")
        os.makedirs(f"./escargot_memory/{collection_name}")

       # delete you persist_directory and create persist_directory againt
        self.client = chromadb.PersistentClient(path=f"./escargot_memory/{collection_name}", settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(collection_name)
        self.graph = Graph()
        self.save_graph()

    def store_memory(self, text, metadata={}, data = None):
        # Embed the text using the lm's embed function
        vector = self.lm.get_embedding(text)
        memory_id = str(uuid.uuid4())
        metadata["memory_id"] = memory_id
        # If data is not None, create a uuid name and a pkl file. the metadata will contain the file name within the collection folder
        if data is not None:
            with open(f"./escargot_memory/{self.collection_name}/{memory_id}.pkl", "wb") as f:
                pickle.dump(data, f)
            metadata["file"] = memory_id

        # Add the embedded vector to the collection
        collection = self.client.get_collection(self.collection_name)
        collection.add(ids=text, embeddings=[vector], metadatas=metadata)

        self.graph.add_node(
            timestamp=self.graph_timestamp,
            id=memory_id,
            properties={"text":text, "vector":vector},
        )
        self.graph_timestamp += 1

    def query_collection(self,query, max_results=10, metadata = None):
        collection = self.client.get_collection(self.collection_name)
        query_embeddings = self.lm.get_embedding(query)
        if metadata is not None:
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=max_results,
                where=metadata
            )
        else:
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=max_results
            )
        return results

    def get_all_vectors(self):
        # Return all vectors stored in the collection
        return self.collection.get()

    def delete_vector(self, text):
        # Delete a vector by id (text)
        self.collection.delete(ids=text)

    def get_pkl_data(self, memory_id):
        """
        Retrieves the data stored in a pickle file associated with a given memory_id.

        Args:
            memory_id (str): The UUID string used as the filename (without extension) for the pickle file.

        Returns:
            The deserialized data from the pickle file, or None if the file does not exist.
        """
        file_path = f"./escargot_memory/{self.collection_name}/{memory_id}.pkl"
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                return data
            except Exception as e:
                print(f"Error loading pickle file {file_path}: {e}")
                return None
        else:
            print(f"Pickle file not found: {file_path}")
            return None
        

    def save_graph(self, collection_name = None):
        if collection_name is None:
            self.graph.save_to_file(f"./escargot_memory/{self.collection_name}/graph")
        else:
            if os.path.exists(f"./escargot_memory/{collection_name}/graph"):
                self.graph.save_to_file(f"./escargot_memory/{collection_name}/graph")

    def load_graph(self, collection_name = None):
        if collection_name is None:
            self.graph = Graph.load_from_file(f"./escargot_memory/{self.collection_name}/graph")
        else:
            if os.path.exists(f"./escargot_memory/{collection_name}/graph"):
                self.graph = Graph.load_from_file(f"./escargot_memory/{collection_name}/graph")
        