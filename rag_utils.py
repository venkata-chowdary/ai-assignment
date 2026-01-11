import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from langchain_core.vectorstores import VectorStoreRetriever

def load_and_chunk_pdf(file_path: str):
    """Loads a PDF and splits it into chunks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

def setup_vector_store(chunks, embedding_model=None):
    """Sets up the Chroma vector store."""
    if embedding_model is None:
        if not os.getenv("GOOGLE_API_KEY"):
             raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Using Local mode (persistent)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="pdf_rag",
        persist_directory="./chroma_db"
    )
    return vector_store

def get_retriever(vector_store, search_type="similarity", k=3) -> VectorStoreRetriever:
    """Returns a retriever from the vector store."""
    return vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})
