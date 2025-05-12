import os
import time
import json
from tqdm.auto import tqdm

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from pinecone import Pinecone, ServerlessSpec  #  Vector Database

import DLAIUtils
from DLAIUtils import Utils

from embedding import *
from extract_text import *
from document_preprocessing import *


files = [
    "files/Bias and Fairness in Large Language Models.pdf",
    "files/Fairness Certification for Natural Language.pdf",
    "files/Fairness in Language Models Beyond English Gaps and Challenges.pdf",
]


documents = extract_text(files)
chunked_documents = chunk_documents(documents)

embeddings = embed(documents)

prompt = "importance of fairness in NLP"
embedded_prompt = embed(prompt)


pc = Pinecone(
    api_key="pcsk_7RjuJW_GMWhcBRHT6wcy2L8Qu9CDnfxtN1pSJMzBGkufgAfdice1FRcJRNhgGCn43Wt3Fg"
)

index_name = "semantic_search_vdb"
pc.create_index(
    name=index_name,
    # dimension=model.get_sentence_embedding_dimension(),
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-west-2"),
)

index = pc.Index(index_name)
print(index)

batch_size = 200
vector_limit = 10000

questions = question[:vector_limit]


for i in tqdm(range(0, len(questions), batch_size)):
    # find end of batch
    i_end = min(i + batch_size, len(questions))
    # create IDs batch
    ids = [str(x) for x in range(i, i_end)]
    # create metadata batch
    metadatas = [{"text": text} for text in questions[i:i_end]]
    # create embeddings
    xc = model.encode(questions[i:i_end])
    # create records list for upsert
    records = zip(ids, xc, metadatas)
    # upsert to Pinecone
    index.upsert(vectors=records)

index.describe_index_stats()


# small helper function so we can repeat queries later
def run_query(query):
    embedding = model.encode(query).tolist()
    results = index.query(
        top_k=10, vector=embedding, include_metadata=True, include_values=False
    )
    for result in results["matches"]:
        print(f"{round(result['score'], 2)}: {result['metadata']['text']}")


query = "how do i make chocolate cake?"
run_query(query)
