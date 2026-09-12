'''
long term memory module that uses a local sqlite database to store and retrieve information.
This module is designed to be pre-emptively injected into the model's context, allowing it to seamlessly access and utilize the stored information during its interactions. The memory is structured to support various types of data, including text, images, and other relevant content, enabling the model to provide more informed and contextually aware responses.
'''
import ollama
import chromadb
import os
import uuid
from datetime import datetime, timezone


EMBEDDING_MODEL = "bge-m3"



class LongTermMemory:
    def __init__(self):
        self.client = ollama.Client(host="http://localhost:11434")
        self.chroma_client = chromadb.PersistentClient(path=os.path.dirname(os.path.abspath(__file__)) + "/semantic_memory")
        self.collection = self.chroma_client.get_or_create_collection("amadeus_memory")


    def embed(self, text):
        """Return the embedding vector for a single piece of text, or None on failure."""

        try:
            embed_response = self.client.embed(model=EMBEDDING_MODEL, input=[text])
            return embed_response.embeddings[0]
        except Exception as e:
            print(f"Error during embedding: {e}")
            return None


    def save_memory(self,text):

        vector = self.embed(text)
        if vector is None:
            return None

        memory_id = str(uuid.uuid4())

        try:
            self.collection.add(
                ids=[memory_id],
                embeddings=[vector],
                documents=[text],
                metadatas=[{"created_at": datetime.now(timezone.utc).isoformat()}],
            )
            print("Memory saved successfully.")
            return memory_id
        except Exception as e:
            print(f"Error during memory saving: {e}")
            return None

    def retrieve_memory(self,query, top_k=2):
        query_vector = self.embed(query)
        if query_vector is None:
            return None

        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            return results.get("documents", [[]])[0]  # Return the list of documents, or an empty list if not found
        except Exception as e:
            print(f"Error during memory retrieval: {e}")
            return None







