import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_utils import load_and_chunk_pdf, setup_vector_store

class TestRAGUtils(unittest.TestCase):
    
    @patch('rag_utils.PyPDFLoader')
    @patch('rag_utils.RecursiveCharacterTextSplitter')
    def test_load_and_chunk_pdf(self, mock_splitter, mock_loader):
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

    @patch('rag_utils.Chroma')
    @patch('rag_utils.GoogleGenerativeAIEmbeddings')
    def test_setup_vector_store(self, mock_embeddings, mock_chroma):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test"}):
            chunks = ["chunk1"]
            setup_vector_store(chunks)
            mock_chroma.from_documents.assert_called_once()

if __name__ == '__main__':
    unittest.main()
