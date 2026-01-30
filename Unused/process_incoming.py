import json
import requests
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_lists):
    r = requests.post("http://localhost:11434/api/embed",json=
              {"model": "bge-m3",
               "input": text_lists
               })
    return r.json()['embeddings']

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate",json=
              {"model": "llama3.2",
               "prompt": prompt,
               "stream": False
               })
    return r.json()['response']

#Load the saved embeddings DataFrame
df=joblib.load("embedding.joblib")

incoming_query=input("Ask a Question: ")
question_embedding=create_embedding([incoming_query])[0]

#Find cosine similarity of question with all chunks
similarities=cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()

top_results=5
max_indx=similarities.argsort()[-top_results:][::-1] # Top 3 similar chunks


new_df=df.iloc[max_indx]

prompt=f''' I am teaching Machine learning using Krish Naik ML course.Here are video subtitle chunks with their video titles,video number,text at that time,start and end time stamps in seconds and I have the following video chunks data:

{new_df[["title","number","start","end","text"]].to_json(orient="records",lines=False)}



-------------------------------------------------
Now, answer the question based on the above video chunks data.

Question:"{incoming_query}"

User asked the this question realted to the video chunks,you have to answer in human way(dont mention in above format its only for you)where and how much content is taught  in which video. and at what time stamp. and guide the user to that part of the video.
If you are unable to find the answer, simply state that you don't know and tell him to ask question related to the video content only.

'''

with open("prompt.txt","w") as f:
    f.write(prompt)

response=inference(prompt)
print(response)

with open("response.txt","w") as f:
    f.write(response)