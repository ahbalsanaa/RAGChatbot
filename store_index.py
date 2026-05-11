from dotenv import load_dotenv
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.helper import load_pdfs, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv() # Load environment variables from .env file

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_docs = load_pdfs(pdf_directory="data/")
filtered_docs = filter_to_minimal_docs(extracted_docs)
chunked_docs = text_split(filtered_docs)

embeddings = download_embeddings()


pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


index_name = "ragchatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384, # dimension of the embedding vector
        metric="cosine", # similarity metric
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    index_name=index_name
)


# Add new documents to the existing index
"""
new_doc = Document(
    page_content="This is a new document to be added to the existing index.",
    metadata={"source": "new_doc.pdf"}
)
docsearch.add_documents(documents=[new_doc])
"""

print("Indexing complete!")