import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
import PyPDF2
# Setting up logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()



def load_documents() -> List[str]:
    """
    Load documents for demonstration.

    Returns:
        List of sample documents
    """
    results = []
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if os.path.isfile(fpath) and fname.lower().endswith(".txt"):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                # Use filename (without extension) as title metadata
                metadata = {"source": fname, "title": os.path.splitext(fname)[0]}
                results.append({"content": content, "metadata": metadata})
            except Exception as e:
                logger.warning(f"Failed to load {fname}: {e}")
        # Support PDF files by extracting text from pages
        elif os.path.isfile(fpath) and fname.lower().endswith(".pdf"):
            try:
                reader = PyPDF2.PdfReader(fpath)
                pages = []
                for page in reader.pages:
                    try:
                        pages.append(page.extract_text() or "")
                    except Exception:
                        pages.append("")
                content = "\n\n".join(pages)
                metadata = {"source": fname, "title": os.path.splitext(fname)[0], "pages": len(reader.pages)}
                results.append({"content": content, "metadata": metadata})
            except Exception as e:
                logger.warning(f"Failed to load PDF {fname}: {e}")
    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        # Conversation history- short term memory
        self.conversation_history = []
 
        
        # Create RAG prompt template
        # This template provides the retrieved context and the user's question to the LLM
        self.prompt_template = ChatPromptTemplate.from_template(
            """
    You are an expert AI Cybersecurity assistant. Only use the following context to answer the user's question. 
    If the context contains relevant information, use it. If the answer is not in the context, say I don't have enough relative information to answer the question. 
Format your answer in a clear and concise manner using Markdown. Always cite the source of the information you use from the context. 
    Format responses in clear readable format using bullet points and sections if needed. If the question is not answerable based on the context, say I don't have enough relative information to answer the question.

    Context:
    {context}

    Previous conversation history:
    {conversation_history}

    Question:
    {question}
    """
        )

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM using Google Gemini.
        """
        # Check for Gemini API key first. If not found, check for OpenAI, then Groq. This allows users to easily 
        # switch between providers by setting the appropriate API key in the .env file.
        if os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )
        
        elif os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )    

        else:
            raise ValueError(
                "No valid API key found. Please set GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def query(self, input: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            String answer from the LLM
        """
        # Retrieve relevant context chunks from the vector database
        search_results = self.vector_db.search(input, n_results=n_results)
        # Combine the retrieved document chunks into a single context string
        context = "\n\n".join(search_results.get("documents", []))
        # Call the LLM chain with the context and question
        conversation_history_str = "\n".join(self.conversation_history)
        answer = self.chain.invoke({"context": context, "question": input, "conversation_history": conversation_history_str})

        # Update conversation history
        self.conversation_history.append(f"Human: {input}")
        self.conversation_history.append(f"AI: {answer}")

        return answer


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} sample documents")

        assistant.add_documents(sample_docs)

        done = False

        while not done:
            question = input("Hi I'm Avery, your smart AI assistant. Enter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.query(question)
                print(result)

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()
