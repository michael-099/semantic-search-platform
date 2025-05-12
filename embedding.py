from extract_text import *
from sentence_transformers import SentenceTransformer


def extract(files):
    documents = []  
    for file in files:
        text = extract_text_from_file(file)
        if text:  
            documents.append(text)
    return documents

def embed(files):
    extracted_texts = extract(files)

    # the pre trained model 
    model_name = 'all-mpnet-base-v2' 
    # loading the pretrained model 
    model = SentenceTransformer(model_name)
    # encoding to high dimensional vector 
    embedding = model.encode(extracted_texts)
    
    return embedding


