import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_utils import load_and_chunk_pdf, setup_vector_store
from langchain_core.documents import Document

class TestRAGUtils(unittest.TestCase):
    
    @patch('rag_utils.PyPDFLoader')
    @patch('rag_utils.RecursiveCharacterTextSplitter')
    def test_load_and_chunk_pdf(self, mock_splitter, mock_loader):
        """Test: PDF loader correctly reads and chunks document."""
        # Setup mocks
        mock_loader_instance = mock_loader.return_value
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_loader_instance.load.return_value = [mock_doc]
        
        mock_splitter_instance = mock_splitter.return_value
        mock_splitter_instance.split_documents.return_value = ["chunk1", "chunk2"]
        
        # Create a dummy file for os.path.exists check
        with open("dummy.pdf", "w") as f:
            f.write("dummy")
            
        try:
            chunks = load_and_chunk_pdf("dummy.pdf")
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0], "chunk1")
        finally:
            os.remove("dummy.pdf")

    @patch('rag_utils.batch_add_documents')
    @patch('rag_utils.QdrantVectorStore')
    @patch('rag_utils.QdrantClient')
    @patch('rag_utils.GoogleGenerativeAIEmbeddings')
    def test_setup_vector_store(self, mock_embeddings, mock_client, mock_qdrant, mock_batch_add):
        """Test: RAG system initializes new vector store and ingests documents."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test"}):
            chunks = [Document(page_content="chunk1")] * 5
            
            # Mock client methods
            mock_client_instance = mock_client.return_value
            # Case 1: Collection does not exist
            mock_client_instance.collection_exists.return_value = False
            mock_client_instance.count.return_value.count = 0
            
            store = setup_vector_store(chunks)
            
            # Check if collection created
            mock_client_instance.create_collection.assert_called_once()
            
            # Check if store initialized
            mock_qdrant.assert_called()
            
            # Check if batch_add_documents called
            mock_store_instance = mock_qdrant.return_value
            mock_batch_add.assert_called_with(mock_store_instance, chunks)

    @patch('rag_utils.batch_add_documents')
    @patch('rag_utils.QdrantVectorStore')
    @patch('rag_utils.QdrantClient')
    @patch('rag_utils.GoogleGenerativeAIEmbeddings')
    def test_setup_vector_store_existing(self, mock_embeddings, mock_client, mock_qdrant, mock_batch_add):
        """Test: RAG system detects existing collection and skips ingestion (quota optimization)."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test"}):
            chunks = [Document(page_content="chunk1")]
            
            mock_client_instance = mock_client.return_value
            # Case 2: Collection exists and has docs
            mock_client_instance.collection_exists.return_value = True
            mock_client_instance.count.return_value.count = 10
            
            setup_vector_store(chunks)
            
            # Should NOT create collection
            mock_client_instance.create_collection.assert_not_called()
            
            # Should NOT add documents
            mock_batch_add.assert_not_called()


if __name__ == '__main__':
    unittest.main()
