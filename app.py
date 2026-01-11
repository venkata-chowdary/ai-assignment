import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from agent_graph import create_agent_graph, create_retriever_tool
from rag_utils import load_and_chunk_pdf, setup_vector_store, get_retriever

# Load environment variables
load_dotenv()

st.set_page_config(page_title="AI Agent Pipeline", layout="wide")

st.title("My AI Assistant")

# Sidebar for Setup
st.sidebar.header("Document Upload")

# Check if keys are set (so it doesn't just crash silently)
if not os.getenv("GOOGLE_API_KEY"):
    st.sidebar.error("Yo! You forgot the GOOGLE_API_KEY in .env")
if not os.getenv("OPENWEATHERMAP_API_KEY"):
    st.sidebar.error("The Weather API key is missing too!")

# File Upload 
# I put this in the sidebar to keep the main chat clean
uploaded_file = st.sidebar.file_uploader("Upload your PDF (Resume, Report, etc.)", type=["pdf"], help="Max size: 2MB")

from qdrant_client import QdrantClient

# --- RAG INITIALIZATION (CACHED) ---
@st.cache_resource
def get_qdrant_client():
    """
    getting singleton qdrant client. works better this way.
    """
    return QdrantClient(path="./qdrant_db")

@st.cache_resource
def init_rag_system(file_path: str):
    """
    initializing rag system here. cached for speed.
    """
    try:
        # getting client
        client = get_qdrant_client()
        
        chunks = load_and_chunk_pdf(file_path)
        # setup_vector_store handles rate limits nicely
        vector_store = setup_vector_store(chunks, client=client)
        retriever = get_retriever(vector_store)
        
        # creating tool for agent
        tool = create_retriever_tool(
            retriever,
            "retrieve_documents",
            "Search and retrieve information from the uploaded PDF document. Use this tool when the user asks questions about the document's content."
        )
        return tool
    except Exception as e:
        st.error(f"failed to init rag: {e}")
        return None

# init session vars
if "messages" not in st.session_state:
    st.session_state.messages = []

# global var
retriever_tool = None

# file upload logic
if uploaded_file:
    # 2MB check for free tier
    MAX_FILE_SIZE = 2 * 1024 * 1024 # 2MB
    
    if uploaded_file.size > MAX_FILE_SIZE:
        st.sidebar.error("File is too big (max 2MB). Please upload a smaller one.")
    else:
        # saving to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
    
        # init rag
        with st.spinner("Reading your document..."):
            retriever_tool = init_rag_system(tmp_path)
            
        if retriever_tool:
            st.sidebar.success("Document loaded! You can ask questions now.")

# recreating graph
st.session_state.graph = create_agent_graph(retriever_tool) 

def context_text(msg_content):
    if isinstance(msg_content, list):
        # clean text from blocks
        return "".join([block["text"] for block in msg_content if "text" in block])
    return msg_content

# display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(context_text(msg.content))
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(context_text(msg.content))

if prompt := st.chat_input("Ask about weather or pdf..."):
    # add user msg
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # prep state
                input_state = {"messages": st.session_state.messages}
                
                # calling graph
                final_state = st.session_state.graph.invoke(input_state)
                
                response_msg = final_state["messages"][-1]
                st.markdown(context_text(response_msg.content))
                
                st.session_state.messages.append(response_msg)
            except Exception as e:
                st.error(f"Error happened: {e}")

# simple debug info
with st.expander("Debug Info"):
    st.write("LangSmith traces running in background.")

