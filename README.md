# My AI Assistant Project

This is my project for the AI Engineer assignment. I built an AI agent using **LangGraph** and **LangChain** that can do two main things:

1. Check the weather (using OpenWeatherMap).
2. Answer questions from a PDF you upload (using RAG with Google Gemini & Qdrant).

The agent is smart enough to choose which tool to use based on what you ask.

## Prerequisites

-   You need Python 3.10 or newer installed.
-   **API Keys**: You need these keys in a `.env` file:
    -   `GOOGLE_API_KEY`: For the Gemini AI (I used `gemini-2.5-flash`).
    -   `OPENWEATHERMAP_API_KEY`: To get weather data.
    -   `LANGCHAIN_API_KEY`: Optional, if you want to see traces.

## How to Run (Windows)

1. **Clone my code**:
   Download this folder to your laptop.

2. **Create a Virtual Env**:
   Open your terminal (PowerShell or whatever) and run:

    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```

3. **Install libraries**:
   Run this command to install all the needed packages:

    ```powershell
    pip install -r requirements.txt
    ```

4. **Setup Keys**:

    - Create a new file called `.env`.
    - Paste your keys inside like this:
        ```env
        GOOGLE_API_KEY=replace_with_your_key
        OPENWEATHERMAP_API_KEY=replace_with_your_key
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_PROJECT=my-assignment
        LANGCHAIN_API_KEY=replace_with_your_key
        ```

5. **Run the App**:
   Type this in terminal:
    ```powershell
    streamlit run app.py
    ```
    It will open in your browser automatically.

## How to use it

-   **Weather**: Just ask "weather in Mumbai" or any city.
-   **Docs**: Upload a PDF on the left side. Wait for it to say "Document loaded!" then ask questions like "summary of this pdf".

## Problems?

-   If you get a **Qdrant Lock Error** ("file already accessed"), just close the terminal (`Ctrl+C`) and run it again. It happens sometimes with local DBs.
-   If it says **429 Error** (Rate Limit), just wait a few seconds. I added a delay code to handle this but sometimes free tier causes issues.

## File Info

-   `app.py`: The main file for the UI.
-   `agent_graph.py`: Where the specialized graph logic is.
-   `rag_utils.py`: Code for handling PDFs and Database stuff.
-   `weather_tool.py`: Simple function to call weather API.

## How to Test

If you want to check if everything is working fine (without running the app), run:

```powershell
python -m unittest discover tests -v
```

It should say `OK` if all 7 tests pass.
