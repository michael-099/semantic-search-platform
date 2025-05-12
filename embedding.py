from extract_text import *
from sentence_transformers import SentenceTransformer


def embed(extracted_texts):

    # the pre trained model 
    model_name = 'all-mpnet-base-v2' 
    # loading the pretrained model 
    model = SentenceTransformer(model_name)
    
    # dimension = model.get_sentence_embedding_dimension(),
    # encoding to high dimensional vector 
    embedding = model.encode(extracted_texts , convert_to_numpy = True)
    
    return embedding


