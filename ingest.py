import nltk
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader

# Ensure tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

def chunk_text(text, chunk_size=12, overlap=3):
    """Increased chunk size to 12 sentences for better section-level context."""
    sentences = sent_tokenize(text)
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i : i + chunk_size])
        chunks.append(chunk)
        i += (chunk_size - overlap)
        if i + overlap >= len(sentences):
            break
    return chunks