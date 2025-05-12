from embedding import *

files=["files/Bias and Fairness in Large Language Models.pdf",
       "files/Fairness Certification for Natural Language.pdf",
       "files/Fairness in Language Models Beyond English Gaps and Challenges.pdf"]

embeddings = embed(files)


prompt = "importance of fairness in NLP"
embedded_prompt = embed(prompt)






