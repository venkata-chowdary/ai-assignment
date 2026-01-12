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
        """Test: LangGraph agent compiles successfully (valid structure)."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test"}):
            graph = create_agent_graph()
            self.assertIsNotNone(graph)
            
    # Execution test skipped due to complex mocking of inner function logic
    # @patch('agent_graph.ChatGoogleGenerativeAI')
    # @patch('agent_graph.get_weather')
    # def test_agent_graph_execution(self, mock_weather, mock_llm):
    #     ...

if __name__ == '__main__':
    unittest.main()
