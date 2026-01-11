import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from agent_graph import create_agent_graph

# Load environment variables
load_dotenv()

st.set_page_config(page_title="AI Agent Pipeline", layout="wide")

st.title("LangGraph AI Agent: Weather & RAG")

# Sidebar for Setup
st.sidebar.header("Configuration")

# API Key Validation (Basic)
if not os.getenv("GOOGLE_API_KEY"):
    st.sidebar.error("GOOGLE_API_KEY is missing in .env")
if not os.getenv("OPENWEATHERMAP_API_KEY"):
    st.sidebar.error("OPENWEATHERMAP_API_KEY is missing in .env")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload a PDF for RAG", type=["pdf"])

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    # Initialize basic graph without PDF first
    st.session_state.graph = create_agent_graph()

# Handle File Upload
if uploaded_file:
    # Check if we processed this file already to avoid re-processing on every rerun
    if st.session_state.get("current_file") != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Re-create graph with PDF
            st.session_state.graph = create_agent_graph(tmp_path)
            st.session_state.current_file = uploaded_file.name
            st.sidebar.success(f"Loaded {uploaded_file.name}")
            # Clean up temp file (os.remove) could be done here or later

# Chat Interface
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

if prompt := st.chat_input("Ask about the weather or your PDF..."):
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run Agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare state
                input_state = {"messages": st.session_state.messages}
                
                # Stream or Invoke
                # For simplicity, using invoke. Streaming integration with LangGraph + Streamlit takes a bit more setup.
                final_state = st.session_state.graph.invoke(input_state)
                
                response_msg = final_state["messages"][-1]
                st.markdown(response_msg.content)
                
                st.session_state.messages.append(response_msg)
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Evaluation / Debug Info (Optional)
with st.expander("Debug / Traces"):
    st.write("LangSmith traces are active in the background if configured.")
