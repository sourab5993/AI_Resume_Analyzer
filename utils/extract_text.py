import os
import PyPDF2
import docx2txt

def extract_text(file_path):
    """
    Extract raw text from PDF, DOCX, DOC, or TXT resume files.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        text = ''
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ''
        return text
    elif ext in ['.docx', '.doc']:
        return docx2txt.process(file_path)
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    else:
        raise ValueError('Unsupported file type: ' + ext)
