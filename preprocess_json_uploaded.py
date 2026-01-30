import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from chroma_client import get_chroma

load_dotenv()
client = OpenAI(api_key=os.getenv("api_key"))

def create_embeddings_batch(texts, batch_size=50):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=batch
        )
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings

def embed_json_file(json_file):
    chroma_client, collection = get_chroma()

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = [c for c in data["chunks"] if c["text"]]
    texts = [c["text"] for c in chunks]
    embeddings = create_embeddings_batch(texts)

    ids, documents, metadatas = [], [], []

    for chunk in chunks:
        uid = f"{chunk['title']}__{chunk['number']}__{int(chunk['start']*1000)}"
        ids.append(uid)
        documents.append(chunk["text"])
        metadatas.append({
            "title": chunk["title"],
            "chunk_id": chunk["number"],
            "start": chunk["start"],
            "end": chunk["end"],
            "number": chunk["number"]
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(ids)
