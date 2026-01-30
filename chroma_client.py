import os
import chromadb

CHROMA_DIR = os.path.join(os.path.expanduser("~"), "chroma_store")
os.makedirs(CHROMA_DIR, exist_ok=True)

def get_chroma():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name="lecture_embeddings")
    return client, collection
