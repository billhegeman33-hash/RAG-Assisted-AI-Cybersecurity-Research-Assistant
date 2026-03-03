RAG Cybersecurity Assistant (Streamlit)

Overview
This project is a Retrieval-Augmented Generation (RAG) assistant built with Streamlit. It ingests documents from the data folder, stores embeddings in ChromaDB, and answers questions using a configurable LLM provider.

Features
- Streamlit chat UI
- Document ingestion from data folder (TXT + PDF)
- Local vector store with ChromaDB
- Multiple LLM backend (Google Gemini)

Requirements
- Python 3.11
- A virtual environment
- One API key (Google Gemini)
Quick Start
1) Create and activate a virtual environment
macOS/Linux:
python3.11 -m venv .venv
source .venv/bin/activate

Windows:
python -m venv .venv
.venv\Scripts\activate 

3) Install dependencies
pip install -r requirements.txt

4) Add your API key
Create a .env file at the root directory and add the following
# Google Gemini Configuration
GOOGLE_API_KEY=your_gemini_api_key_here

GOOGLE_MODEL=gemini-2.5-flash

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Groq Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant


# Embedding model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector Database configuration
CHROMA_COLLECTION_NAME=rag_documents



4) Run the app (Streamlit)
streamlit run src/streamlit_app.py

That command will start a local server and open the app in your browser.

Running with python (auto-launch)
You can also run:
python src/streamlit_app.py

This script re-launches itself under Streamlit and opens the browser automatically.

Usage
- Put documents in the data folder.
- Start the app.
- Ask questions in the chat input.
- Adjust “Number of sources” in the sidebar to control retrieval depth.

Where to put the API key
GOOGLE_API_KEY=your_key_here

Notes
- The first run will build embeddings and may take a few minutes.
- ChromaDB data is stored in chroma_db/.
