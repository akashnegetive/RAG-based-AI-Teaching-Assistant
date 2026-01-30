import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from chroma_client import get_chroma

load_dotenv()
client = OpenAI(api_key=os.getenv("api_key"))
chroma_client, collection = get_chroma() # small change here

json_folder = "jsons"
all_chunks = []

for file in os.listdir(json_folder):
    if file.endswith(".json"):
        with open(os.path.join(json_folder, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            for c in data["chunks"]:
                if c["text"]:
                    all_chunks.append(c)

texts = [c["text"] for c in all_chunks]

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

embeddings = create_embeddings_batch(texts)

ids, documents, metadatas = [], [], []

for chunk in all_chunks:   # or chunks in uploaded file
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



print("All JSON files embedded into ChromaDB and persisted to disk.")
