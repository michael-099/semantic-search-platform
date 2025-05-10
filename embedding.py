from extract_text import *



def extract(files):
    text=""
    for file in files:
        text= text + extract_text_from_file(file)
    return text
    
files=["files/Bias and Fairness in Large Language Models.pdf","files/Fairness Certification for Natural Language.pdf","files/Fairness in Language Models Beyond English Gaps and Challenges.pdf"]
print(extract(files)[-5])

