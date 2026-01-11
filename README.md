# LangGraph AI Pipeline Assignment

This project implements an agentic AI pipeline using **LangChain**, **LangGraph**, and **LangSmith**. It features a Streamlit UI that allows users to:

1.  **Ask for real-time weather** (using OpenWeatherMap).
2.  **Ask questions about a PDF document** (using RAG with Google Gemini & Qdrant).
3.  **Intelligent routing**: The agent decides which tool to use.

## Prerequisites

-   Python 3.10+
-   **API Keys**:
    -   `GOOGLE_API_KEY`: For Gemini models (using `gemini-2.5-flash` and `text-embedding-004`).
    -   `OPENWEATHERMAP_API_KEY`: For weather data.
    -   `LANGCHAIN_API_KEY`: For LangSmith tracing (optional but recommended).

## Setup

1.  **Clone the repository** (or navigate to the project folder).
2.  **Create a Virtual Environment** (recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment**:
    -   Create a file named `.env` in the root directory.
    -   Add your API keys:
        ```env
        GOOGLE_API_KEY=your_google_key
        OPENWEATHERMAP_API_KEY=your_weather_key
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_PROJECT=langgraph-assignment
        LANGCHAIN_API_KEY=your_langchain_pat
        ```

## Usage

1.  **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```
2.  **Interact**:
    -   **Weather**: Ask "What's the weather in New York?"
    -   **RAG**: Upload a PDF in the sidebar. Once processed, ask questions like "Summarize the document."

## Troubleshooting

-   **Qdrant Lock Error**: If you see "Storage folder ... is already accessed", stop the running app (`Ctrl+C`) and restart it. If it persists, delete the `./qdrant_db` folder.
-   **Google API Quota**: If you hit 429 errors, wait a minute. The app is optimized to batch requests and reuse existing embeddings.

## Project Structure

-   `app.py`: Main Streamlit application.
-   `agent_graph.py`: Defines the LangGraph agent, nodes, and routing logic.
-   `rag_utils.py`: PDF loading, chunking, and Vector Store (Qdrant) setup.
-   `weather_tool.py`: OpenWeatherMap tool implementation.
-   `tests/`: Unit tests.

## Evaluation

-   The application uses **LangSmith** for tracing. Ensure `LANGCHAIN_TRACING_V2=true` is set in `.env` to view traces in your LangSmith project.

## Rate Limits & Constraints

To ensure this demo runs smoothly on free-tier APIs:

-   **PDF Size**: Max 2MB.
-   **Chunk Limit**: Max 20 chunks per document.
-   **Processing Speed**: Documents are processed at 1 chunk per 4 seconds to avoid 429 Errors.

## Deliverables Checklist

-   [x] LangGraph Agent
-   [x] Weather Tool
-   [x] RAG with PDF (Qdrant)
-   [x] Streamlit UI
-   [x] Unit Tests
