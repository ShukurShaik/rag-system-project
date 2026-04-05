from langchain_community.document_loaders import PyPDFLoader
from app.config.settings import PDF_PATH

def load_documents():
    loader = PyPDFLoader(PDF_PATH)
    return loader.load()
