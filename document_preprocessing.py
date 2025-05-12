import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# downloads the punkt tokenizer models 
nltk.download('punkt')
# downloads common stopwords 
nltk.download('stopwords')
# load english stopword list from nltk
stop_words = set(stopwords.words('english'))

# chunk_text : to split a long piece of text into smaller, meaningful chunks, where each chunk:
#              -> Contains complete sentences (not mid-sentence cuts),
#              -> Has a maximum number of words (default: 150),

def chunk_text(text, max_words=150):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        word_count = len(sent.split())
        if current_len + word_count > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = word_count
        else:
            current_chunk.append(sent)
            current_len += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def chunk_documents(text , max_words=150):
    chunked_documents=[]
    for doc in text:
        chunked_documents.append(chunk_text(doc,max_words))
    return chunk_documents


# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', '', text)
#     tokens = text.split()
#     filtered = [word for word in tokens if word not in stop_words]
#     return " ".join(filtered)
