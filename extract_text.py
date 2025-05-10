import os
from PyPDF2 import PdfReader
from docx import Document
import csv
import fitz 

def extract_text_from_file(file_path):
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading .txt file: {e}")
            return None
    elif extension == ".pdf":
        try:
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            print(f"Error reading .pdf file with PyMuPDF: {e}")
            return None
    
    elif extension == ".docx":
        try:
            doc = Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return '\n'.join(full_text)
        except Exception as e:
            print(f"Error reading .docx file: {e}")
            return None
    elif extension == ".csv":
        try:
            all_text = []
            with open(file_path, 'r', newline='', encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    all_text.append(', '.join(row))
            return '\n'.join(all_text)
        except Exception as e:
            print(f"Error reading .csv file: {e}")
            return None
    else:
        print(f"Unsupported file type: {extension}")
        return None