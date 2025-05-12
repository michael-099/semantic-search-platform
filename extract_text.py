import os
from PyPDF2 import PdfReader
from docx import Document
import csv

def extract_text_from_file(file_path):
    print(f"Extracting {file_path}")
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    print("extension=" + extension)

    # Text from .txt file
    if extension == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                return [(1, text)]
        except Exception as e:
            print(f"Error reading .txt file: {e}")
            return None

    # Text from .pdf file
    elif extension == ".pdf":
        try:
            page_texts = []
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text:
                        page_texts.append((page_num, text))
            return page_texts
        except Exception as e:
            print(f"Error reading .pdf file: {e}")
            return None

    # Text from .docx file
    elif extension == ".docx":
        try:
            doc = Document(file_path)
            full_text = [para.text for para in doc.paragraphs]
            return [(1, '\n'.join(full_text))]
        except Exception as e:
            print(f"Error reading .docx file: {e}")
            return None

    # Text from .csv file
    elif extension == ".csv":
        try:
            all_text = []
            with open(file_path, 'r', newline='', encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    all_text.append(', '.join(row))
            return [(1, '\n'.join(all_text))]
        except Exception as e:
            print(f"Error reading .csv file: {e}")
            return None

    else:
        print(f"Unsupported file type: {extension}")
        return None


def extract_text(files):
    documents = []  
    for file in files:
        text = extract_text_from_file(file)
        if text:  
            documents.append(text)
    return documents
