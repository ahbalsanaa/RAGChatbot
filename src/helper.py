# Extracting text from PDF files
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter #chunking
from langchain_huggingface import HuggingFaceEmbeddings

from typing import List
from langchain_core.documents import Document

def load_pdfs(pdf_directory):
    loader = DirectoryLoader(pdf_directory, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True) # glob="**/*.pdf" to load all PDFs in the directory and subdirectories
    documents = loader.load()
    return documents

def filter_to_minimal_docs(documents: List[Document]) -> List[Document]:
    """Given a list of Document objects, return a list of documents with minimal metadata (source and page content)."""
    filtered_docs: List[Document] = []
    for doc in documents:
        src = doc.metadata.get("source")
        filtered_docs.append(
            Document(
                page_content=doc.page_content, 
                metadata={"source": src}
            )
        )
    return filtered_docs

def text_split(filtered_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    
    chunked_docs = text_splitter.split_documents(filtered_docs)
    return chunked_docs

def download_embeddings():
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2" #sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 for arabic
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
    )
    return embeddings