# Avery — AI Cybersecurity RAG Assistant
> A conversational AI assistant that answers questions about AI security and cybersecurity using a curated set of authoritative documents.

---

## Overview

Avery lets you ask plain-language questions about AI security, risk management, and cybersecurity frameworks — and get focused, source-cited answers drawn directly from trusted government and industry publications. Instead of manually searching through hundreds of pages of NIST guidelines, OWASP reports, and security advisories, you type a question and receive a grounded response in seconds. The system uses Retrieval-Augmented Generation (RAG): it searches a local vector database of pre-indexed document chunks, reranks the best matches, and passes only the most relevant passages to a language model. Conversation history is tracked throughout the session so follow-up questions are understood in context. The assistant is designed to never fabricate information beyond what its source documents contain.

---

## Knowledge Base

The assistant draws answers from 9 authoritative documents:

| # | Document | Publisher |
|---|---|---|
| 1 | AI Risk Management Framework (AI RMF) | NIST |
| 2 | CSI AI Data Security Advisory | CISA / NSA / FBI |
| 3 | Cybersecurity Terms and Definitions for Acquisition (July 2024) | U.S. DoD |
| 4 | Guidelines for Secure AI System Development | NCSC (UK) / CISA (US) |
| 5 | Adversarial Machine Learning Taxonomy (NIST AI 100-2e2025) | NIST |
| 6 | OWASP AI Exchange | OWASP Foundation |
| 7 | SAFE AI Full Report | SAFE AI |
| 8 | Principles for Secure AI Integration in Operational Technology | CISA / NSA |
| 9 | Artificial Intelligence Overview | Curated text |

Place additional `.pdf` or `.txt` files in the `data/` folder to expand the knowledge base. They are ingested automatically on the next app launch.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Framework | LangChain 0.3.27 |
| Vector Store | ChromaDB 1.0.12 |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM (primary) | Google Gemini 2.5 Flash |
| LLM (fallback) | OpenAI GPT-4o-mini / Groq Llama-3.1-8B-Instant |
| Interface | Streamlit |

**API key required**: at least one of Google Gemini, OpenAI, or Groq.

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/CraftKernel/AI-RAG-Research-Assistant-AI-Cybersecurity.git
cd AI-RAG-Research-Assistant-AI-Cybersecurity
```

**2. Create and activate a virtual environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure your API key**

Create a `.env` file in the project root (see `.env.example` for a template):
```
# Primary — Google Gemini (recommended)
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_MODEL=gemini-2.5-flash
```

---

## Usage

**Launch the Streamlit app:**
```bash
streamlit run src/streamlit_app.py
```

The app opens at `http://localhost:8501`. On first launch, embeddings are built from the documents in `data/` — this takes 1–3 minutes. Subsequent launches are fast.

**Ask questions in the chat input.** The assistant identifies the type of question, retrieves the most relevant document passages, and generates a source-cited answer. A confidence score, source count, and query type are shown below each response.

---

## Example Queries

**Q: What is adversarial machine learning?**
> Adversarial machine learning refers to techniques that intentionally exploit vulnerabilities in ML models through crafted inputs or corrupted training data. Key attack types include evasion attacks (manipulating inputs at inference time), poisoning attacks (corrupting training data), and model extraction attacks. — *Source: NIST AI 100-2e2025*

**Q: How do I securely integrate AI into operational technology systems?**
> Key steps include: isolating AI components from OT networks using a DMZ architecture, enforcing least-privilege access controls, validating model and data provenance through the supply chain, and monitoring inference outputs for anomalous behavior. — *Source: CISA/NSA Joint OT Guidance*

**Q: What are the top risks for AI systems according to OWASP?**
> The OWASP AI Exchange identifies risks including prompt injection, training data poisoning, insecure plugin design, excessive model agency, model denial of service, and supply chain vulnerabilities. — *Source: OWASP AI Exchange*

---

## Configuration

All configuration is handled through environment variables in your `.env` file:

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | Google Gemini API key (primary LLM) |
| `GOOGLE_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `OPENAI_API_KEY` | — | OpenAI API key (fallback LLM) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `GROQ_API_KEY` | — | Groq API key (fallback LLM) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHROMA_COLLECTION_NAME` | `rag_documents` | ChromaDB collection name |

The LLM provider is selected automatically based on which key is present (`GOOGLE_API_KEY` checked first). To change the number of retrieved sources per query type, edit `n_results_map` in `src/app.py`.

---

## Project Structure

```
AI-RAG-Research-Assistant-AI-Cybersecurity/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── .gitignore
├── data/                    # Source documents (PDFs + TXT)
├── src/
│   ├── app.py               # RAGAssistant class + CLI entry point
│   ├── streamlit_app.py     # Streamlit UI (main entry point)
│   └── vectordb.py          # ChromaDB, embedding, and reranking
└── chroma_db/               # Auto-generated vector store (gitignored)
```

---

## License

This project is licensed under [CC BY-NC-SA 4.0](LICENSE) — non-commercial use with attribution required.

Source documents are publicly released publications from U.S. government agencies (NIST, CISA, NSA, DoD), the UK NCSC, and the OWASP Foundation. U.S. federal publications are in the public domain under 17 U.S.C. § 105. OWASP materials are licensed under Creative Commons Attribution-ShareAlike. The embedding and reranker models are licensed under Apache 2.0.

---

## Contact & Support

-GitHub Repository: 
https://github.com/billhegeman33-hash/AI-RAG-Research-Assistant-AI-Cybersecurity

-Ready Tensor Profile: 
https://app.readytensor.ai/users/bill.hegeman33

-Bug reports / questions: Open an issue at the GitHub repository link above
