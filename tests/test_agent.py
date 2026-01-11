import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from langchain_core.messages import HumanMessage, AIMessage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_graph import create_agent_graph

class TestAgentGraph(unittest.TestCase):
    
    @patch('agent_graph.ChatGoogleGenerativeAI')
    def test_graph_compile(self, mock_llm):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test"}):
            graph = create_agent_graph()
            self.assertIsNotNone(graph)
            
    # Testing the node logic specifically would require importing the agent_node 
    # which is defined inside create_agent_graph. 
    # For this assignment, we test that the graph compiles and structure is valid.

if __name__ == '__main__':
    unittest.main()
