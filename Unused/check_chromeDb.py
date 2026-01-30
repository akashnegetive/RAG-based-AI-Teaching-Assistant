import chromadb
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_store")

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection("lecture_embeddings")

print("Total chunks stored:", collection.count())
