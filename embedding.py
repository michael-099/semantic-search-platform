from extract_text import *
from sentence_transformers import SentenceTransformer


def embed(text, model, device):
    # Ensure the model is on the right device (GPU if available)
    model = model.to(device)
    
    # Get the sentence embedding
    embeddings = model.encode([text], device=device)
    
    return embeddings[0]  # Return the embedding for the single text



