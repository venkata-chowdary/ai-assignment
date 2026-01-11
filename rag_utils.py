import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

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

import time
from langchain_core.documents import Document

def batch_add_documents(vector_store, documents, batch_size=1, delay=4):
    """adding docs in small batches so google api doesnt give 429 error"""
    total_docs = len(documents)
    print(f"adding {total_docs} docs in batches of {batch_size}...")
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        print(f"processing batch {i//batch_size + 1} out of {(total_docs + batch_size - 1)//batch_size}")
        try:
            vector_store.add_documents(batch)
        except Exception as e:
            print(f"error adding batch: {e}")
            # wait nicely if error comes
            time.sleep(10)
            continue
            
        if i + batch_size < total_docs:
            print(f"sleeping for {delay} seconds to avoid rate limit...")
            time.sleep(delay)

def setup_vector_store(chunks, embedding_model=None, client=None):
    """setting up qdrant vector store here."""
    
    # SAFETY LIMIT: putting limit on chunks so free tier doesnt expire
    if len(chunks) > 20:
        print(f"Warning: too many chunks ({len(chunks)}), checking only first 20.")
        chunks = chunks[:20]

    if embedding_model is None:
        if not os.getenv("GOOGLE_API_KEY"):
             raise ValueError("GOOGLE_API_KEY is missing buddy.")
        # using new model for better results
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # qdrant local path
    qdrant_path = "./qdrant_db"
    
    # 1. Initialize Client
    # using singleton client if passed
    if client is None:
        client = QdrantClient(path=qdrant_path)
    
    collection_name = "pdf_rag"
    
    # 2. Check / Create Collection
    # explicitly creating collection first
    if not client.collection_exists(collection_name):
        print(f"creating new qdrant collection: {collection_name}")
        
        # 768 dims for gemini
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        
        # init store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
        )
        
        # adding docs now
        batch_add_documents(vector_store, chunks)
        
    else:
        print(f"collection {collection_name} already exists. checking content inside...")
        
        # load existing store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
        )
        
        count = client.count(collection_name).count
        
        if count == 0:
            print("collection is empty, adding docs now...")
            batch_add_documents(vector_store, chunks)
        else:
            print(f"found {count} docs already. skipping upload to save quota.")
            
    return vector_store

def get_retriever(vector_store, search_type="similarity", k=3) -> VectorStoreRetriever:
    """Returns a retriever from the vector store."""
    return vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})
