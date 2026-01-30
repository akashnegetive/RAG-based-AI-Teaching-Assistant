from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("api_key"))
print(client.embeddings.create(model="text-embedding-3-large", input=["hello"]))
