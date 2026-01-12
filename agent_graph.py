import os
from typing import Annotated, TypedDict, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
# logging removed for cleaner ui



def create_retriever_tool(retriever, name: str, description: str) -> Tool:
    """
    basically making a tool to fetch docs.
    
    args:
        retriever: the logic to get docs
        name: tool name for llm
        description: what the tool does (imp for llm)
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
    # keeping track of messages in list
    messages: Annotated[list[BaseMessage], add_messages]

def create_agent_graph(retriever_tool: Tool = None):
    """
    creating the main agent graph here.
    """
    
    # default tool is weather
    tools = [get_weather]
    
    # if pdf provided, adding rag tool also
    if retriever_tool is not None:
        tools.append(retriever_tool)

    # checking google key
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY messing.")
        
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    from langchain_core.messages import SystemMessage

    def agent_node(state: AgentState):
        messages = state["messages"]
        # telling agent to use rag tool if asked about pdf
        system_msg = SystemMessage(content="You are a helpful assistant. If the user asks to summarize or asks questions about a document, YOU MUST use the 'retrieve_documents' tool to find the information. Do not ask the user for the text if you can retrieve it.")
        
        # adding system msg to start
        messages_with_system = [system_msg] + messages
        
        return {"messages": [llm_with_tools.invoke(messages_with_system)]}

    # Build Graph
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent") 
    
    return builder.compile()
