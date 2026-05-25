# Running the Streamlit App

## Quick Start

### Option 1: VS Code Play Button (Recommended)
1. Make sure you're in the project folder: `AI-RAG-Research-Assistant-AI-Cybersecurity/`
2. Press `Ctrl+Shift+D` to open the Run & Debug view
3. Select **"Streamlit App"** from the dropdown at the top
4. Click the green **Play button** (or press F5)
5. The app will open at http://localhost:8501

### Option 2: Double-click the batch file (Windows)
Simply double-click `run_app.bat` in the project root directory.

### Option 3: Terminal command
```bash
cd AI-RAG-Research-Assistant-AI-Cybersecurity
python -m streamlit run src/streamlit_app.py
```

### Option 4: Use the Python launcher
```bash
cd AI-RAG-Research-Assistant-AI-Cybersecurity
python run_app.py
```

## Requirements
- Python 3.9+
- Dependencies installed from `requirements.txt`
- `.env` file with one of: `GOOGLE_API_KEY`, `OPENAI_API_KEY`, or `GROQ_API_KEY`

## Features
- 🤖 AI-powered cybersecurity assistant (Avery)
- 📚 RAG (Retrieval Augmented Generation) powered by ChromaDB
- 🔍 Automatic query classification (threat, remediation, detection, etc.)
- 💡 Query expansion with cybersecurity synonyms
- 🎯 Retrieval confidence scoring
- 📊 Real-time quality metrics

## Usage
1. Type a cybersecurity question
2. Press Enter to submit
3. View the AI response with:
   - Confidence score (🟢🟡🟠🔴)
   - Query type classification
   - Severity level (if detected)
4. Type "stats" to see retrieval quality statistics
5. Type "quit" to exit (CLI version)
