import os
from typing import Annotated, TypedDict, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
import logging

# Configure logging
logging.basicConfig(
    filename='agent_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def create_retriever_tool(retriever, name: str, description: str) -> Tool:
    """Create a tool to do retrieval of documents.

    Args:
        retriever: The retriever to use for the retrieval
        name: The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description: The description for the tool. This will be passed to the language
            model, so should be descriptive.

    Returns:
        Tool class to pass to an agent
    """
    return Tool(
        name=name,
        description=description,
        func=retriever.invoke,
    )

from weather_tool import get_weather
from rag_utils import load_and_chunk_pdf, setup_vector_store, get_retriever

from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def create_agent_graph(pdf_path: str = None):
    """
    Creates the LangGraph agent with optional RAG capabilities.
    If pdf_path is provided, it initializes RAG tool.
    """
    
    tools = [get_weather]
    
    if pdf_path and os.path.exists(pdf_path):
        try:
            chunks = load_and_chunk_pdf(pdf_path)
            # Initialize Qdrant
            # Note: In a real app, we might want to persist this or load existing
            vector_store = setup_vector_store(chunks)
            retriever = get_retriever(vector_store)
            
            retriever_tool = create_retriever_tool(
                retriever,
                "retrieve_documents",
                "Search and retrieve information from the uploaded PDF document. Use this tool when the user asks questions about the document's content."
            )
            tools.append(retriever_tool)
        except Exception as e:
            print(f"Error initializing RAG: {e}")

    # Initialize LLM
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not set in environment.")
        
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # Build Graph
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent") 
    
    return builder.compile()
