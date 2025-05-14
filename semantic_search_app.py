import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from embedding import embed
# Initialize Pinecone
PINECONE_API_KEY = "pcsk_7RjuJW_GMWhcBRHT6wcy2L8Qu9CDnfxtN1pSJMzBGkufgAfdice1FRcJRNhgGCn43Wt3Fg"
PINECONE_INDEX_NAME = "semantic-search"
PINECONE_REGION = "us-east-1"
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# Streamlit app
st.title("Semantic Search Application")
# Query input
query = st.text_input("Enter your search query:")
if st.button("Search"):
    if query:
        # Perform the query
        query_vector = embed(query, model, device).tolist()
        results = index.query(vector=query_vector, top_k=3, include_metadata=True)
        # Display results
        st.subheader("Search Results")
        for match in results.matches:
            score = match['score']
            doc_name = match['metadata']['doc_name']
            page_num = match['metadata']['page_number']
            content = match['metadata']['text'][:300] + "..." if len(match['metadata']['text']) > 300 else match['metadata']['text']
            st.write(f"**Score:** {score:.4f}")
            st.write(f"**Document:** {doc_name}, Page: {page_num}")
            st.write(f"**Content:** {content}")
            st.write("---")
    else:
        st.write("Please enter a query.")