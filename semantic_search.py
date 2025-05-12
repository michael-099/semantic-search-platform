import os
import warnings
warnings.filterwarnings('ignore')
import time
import json
from tqdm.auto import tqdm

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from embedding import embed
from extract_text import extract_text
from document_preprocessing import chunk_documents

from pinecone import Pinecone, ServerlessSpec

def save_output_to_file(output, filename="output.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            if isinstance(output, list):
                for item in output:
                    f.write(str(item) + "\n\n")
            else:
                f.write(str(output))
        print(f"Output saved to {filename}")
    except Exception as e:
        print(f"Error saving to file: {e}")

# Files to process
files = [
    "files/Bias and Fairness in Large Language Models.pdf",
    "files/Fairness Certification for Natural Language.pdf",
    "files/Fairness in Language Models Beyond English Gaps and Challenges.pdf",
]

# Step 1: Extract and chunk documents
documents = extract_text(files)
chunked_documents = chunk_documents(documents)
save_output_to_file(chunked_documents)

# Pinecone configuration
PINECONE_API_KEY = "pcsk_7RjuJW_GMWhcBRHT6wcy2L8Qu9CDnfxtN1pSJMzBGkufgAfdice1FRcJRNhgGCn43Wt3Fg"
PINECONE_INDEX_NAME = "semantic-search"
PINECONE_REGION = "us-east-1"
EMBEDDING_DIMENSION = 768
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
    )

index = pc.Index(PINECONE_INDEX_NAME)

# Prepare embeddings
batch_size = 32
vector_limit = 10000
all_records_to_upsert = []

# Corrected iteration for nested structure
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure the model uses GPU if available
model = SentenceTransformer(EMBEDDING_MODEL_NAME).to(device)

for doc_index, doc_pages in enumerate(chunked_documents):
    doc_name = os.path.basename(files[doc_index])
    for page_num, chunk_list in doc_pages:
        for chunk_text in chunk_list:
            embedding = embed(chunk_text, model, device)
            vector_id = f"{doc_name}_page_{page_num}_chunk_{hash(chunk_text[:50])}"
            metadata = {
                "doc_name": doc_name,
                "page_number": page_num,
                "text": chunk_text
            }
            all_records_to_upsert.append((vector_id, embedding.tolist(), metadata))

# Upsert in batches
for i in tqdm(range(0, min(len(all_records_to_upsert), vector_limit), batch_size)):
    batch = all_records_to_upsert[i:i + batch_size]
    ids = [record[0] for record in batch]
    vectors = [record[1] for record in batch]
    metadatas = [record[2] for record in batch]
    index.upsert(vectors=[(id, vector, metadata) for id, vector, metadata in zip(ids, vectors, metadatas)])

print(index.describe_index_stats())

# Query function
def query_pinecone(index, query_text, top_k=3):
    query_vector = embed(query_text, model, device).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    print(f"\n--- Top {top_k} Results for Query: '{query_text}' ---")
    for match in results.matches:
        print(f"Score: {match['score']}")
        print(f"Document: {match['metadata']['doc_name']}, Page: {match['metadata']['page_number']}")
        print(f"Content: {match['metadata']['text'][:300]}...")
        print("-" * 30)

# Example query
query_text = "What are the fairness challenges NLP?"
query_pinecone(index, query_text)
