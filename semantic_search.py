from embedding import *

files=["files/Bias and Fairness in Large Language Models.pdf",
       "files/Fairness Certification for Natural Language.pdf",
       "files/Fairness in Language Models Beyond English Gaps and Challenges.pdf"]

embeddings = embed(files)
for i, embedding in enumerate(embeddings):
    print(f"Embedding for document {i+1} (first 10 dimensions): {embedding[:10]}")
    
print(f"\nShape of the embeddings: {embeddings.shape}")


