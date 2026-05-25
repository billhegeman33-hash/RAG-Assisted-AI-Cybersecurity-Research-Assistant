import os
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
try:
    from langchain_groq import ChatGroq
except ImportError:  # Optional dependency
    ChatGroq = None
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
import PyPDF2
from datetime import datetime
# Setting up logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# Cybersecurity Domain Refinements
QUERY_TYPES = {
    "definition": ["what is", "define", "explain", "meaning", "refer"],
    "threat": ["threat", "attack", "exploit", "vulnerability", "breach", "malware"],
    "detection": ["detect", "identify", "recognize", "find", "spot", "discover"],
    "remediation": ["fix", "mitigate", "prevent", "solution", "protect", "defend"],
    "compliance": ["comply", "regulation", "requirement", "standard", "framework", "cis", "nist"],
    "investigation": ["investigate", "analyze", "examine", "incident", "forensic"],
}

CYBERSECURITY_SYNONYMS = {
    "ransomware": ["ransomware", "encryption attack", "crypto-locker", "crypto-ransomware"],
    "phishing": ["phishing", "spear phishing", "whaling", "social engineering"],
    "firewall": ["firewall", "network segmentation", "packet filtering", "stateful firewall"],
    "malware": ["malware", "virus", "worm", "trojan", "spyware", "adware"],
    "vulnerability": ["vulnerability", "CVE", "weakness", "flaw", "bug"],
    "authentication": ["authentication", "MFA", "2FA", "multi-factor", "password"],
    "encryption": ["encryption", "AES", "RSA", "TLS", "SSL", "cryptography"],
    "zero trust": ["zero trust", "never trust always verify", "zero trust network"],
}

SEVERITY_KEYWORDS = {
    "critical": ["critical", "severe", "extreme", "catastrophic"],
    "high": ["high", "dangerous", "significant"],
    "medium": ["medium", "moderate", "notable"],
    "low": ["low", "minor", "trivial"],
}


def classify_query(question: str) -> str:
    """Classify query into cybersecurity domain type.
    
    Args:
        question: User's question
        
    Returns:
        Query type string
    """
    question_lower = question.lower()
    for qtype, keywords in QUERY_TYPES.items():
        if any(kw in question_lower for kw in keywords):
            return qtype
    return "general"


def expand_query(question: str) -> str:
    """Expand query with domain-specific synonyms.
    
    Args:
        question: User's question
        
    Returns:
        Expanded query with synonyms
    """
    expanded = question
    question_lower = question.lower()
    
    for term, synonyms in CYBERSECURITY_SYNONYMS.items():
        if term in question_lower:
            # Add synonyms to query for broader retrieval
            expanded += f" {' '.join(synonyms)}"
    
    return expanded


def extract_severity(question: str) -> Optional[str]:
    """Extract severity level from query if present.
    
    Args:
        question: User's question
        
    Returns:
        Severity level or None
    """
    question_lower = question.lower()
    for severity, keywords in SEVERITY_KEYWORDS.items():
        if any(kw in question_lower for kw in keywords):
            return severity
    return None


def load_documents() -> List[Dict]:
    """
    Load documents for demonstration.

    Returns:
        List of documents with content and metadata
    """
    results = []
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    # Check if data directory exists
    if not os.path.isdir(data_dir):
        logger.warning(f"Data directory not found: {data_dir}")
        return results
    
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

    def __init__(self, max_history: int = 10, confidence_threshold: float = 0.5, use_hybrid_search: bool = True):
        """Initialize the RAG assistant.
        
        Args:
            max_history: Maximum number of conversation turns to keep
            confidence_threshold: Minimum similarity score for retrieval results (0-1)
            use_hybrid_search: Whether to use hybrid search (vector + keyword)
        """
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        # Conversation history with size limit
        self.conversation_history: List[str] = []
        self.max_history = max_history
        self.confidence_threshold = confidence_threshold
        self.use_hybrid_search = use_hybrid_search
        
        # Retrieval metrics for debugging
        self.retrieval_log: List[Dict] = []
        
        # Track retrieval quality metrics
        self.quality_metrics = {
            "total_queries": 0,
            "avg_confidence": 0.0,
            "high_confidence_queries": 0,
            "low_confidence_queries": 0,
        }
 
        
        # Create RAG prompt template
        # This template provides the retrieved context and the user's question to the LLM
        self.prompt_template = ChatPromptTemplate.from_template(
            """
    You are an expert AI Cybersecurity assistant. Your task is to provide accurate, well-researched answers based exclusively on the provided context.

    **Instructions:**
    - Use ONLY the context provided to answer questions
    - If the context doesn't contain relevant information, respond: "I don't have enough relevant information to answer this question. Please provide additional context or try a different query."
    - Always cite your sources by including the source name from the context
    - Structure responses using bullet points, numbered lists, or sections for clarity
    - Use Markdown formatting for better readability
    - Be concise and technically precise
    - If uncertain, acknowledge the limitation rather than speculating

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

    def _initialize_llm(self) -> Optional[object]:
        """
        Initialize the LLM using Google Gemini.
        
        Returns:
            Initialized LLM object or None if no API key available
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

        elif os.getenv("GROQ_API_KEY") and ChatGroq is not None:
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        else:
            raise ValueError(
                "No valid API key found. Please set GOOGLE_API_KEY in your .env file"
            )

    def _get_confidence_badge(self, confidence: float) -> str:
        """Convert confidence score to visual badge.
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Markdown badge string
        """
        if confidence >= 0.8:
            return "🟢 High Confidence"
        elif confidence >= 0.6:
            return "🟡 Medium Confidence"
        elif confidence >= self.confidence_threshold:
            return "🟠 Low Confidence"
        else:
            return "🔴 Very Low Confidence"
    
    def _calculate_retrieval_quality(self, confidence_scores: List[float]) -> Dict:
        """Calculate retrieval quality metrics.
        
        Args:
            confidence_scores: List of confidence scores from retrieval
            
        Returns:
            Dictionary with quality metrics
        """
        if not confidence_scores:
            return {"avg_confidence": 0, "max_confidence": 0, "quality_level": "No results"}
        
        avg_conf = sum(confidence_scores) / len(confidence_scores)
        max_conf = max(confidence_scores)
        
        # Determine quality level
        if avg_conf >= 0.8:
            quality = "Excellent"
        elif avg_conf >= 0.6:
            quality = "Good"
        elif avg_conf >= 0.4:
            quality = "Fair"
        else:
            quality = "Poor"
        
        return {
            "avg_confidence": round(avg_conf, 2),
            "max_confidence": round(max_conf, 2),
            "quality_level": quality,
            "num_results": len(confidence_scores)
        }

    def _rewrite_query(self, question: str) -> str:
        if not self.conversation_history:
            return question
        rewrite_prompt = ChatPromptTemplate.from_template(
            "Rewrite the follow-up question as a standalone search query using the conversation context.\n"
            "Return only the rewritten query, nothing else.\n\n"
            "Conversation:\n{history}\n\n"
            "Follow-up: {question}\n"
            "Standalone query:"
        )
        history = "\n".join(self.conversation_history[-6:])
        return (rewrite_prompt | self.llm | StrOutputParser()).invoke(
            {"history": history, "question": question}
        ).strip()

    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents with 'content' and 'metadata' keys
        """
        if not documents:
            logger.warning("No documents provided to add")
            return
        self.vector_db.add_documents(documents)

    def query(self, input: str, n_results: int = 3) -> Tuple[str, Dict]:
        """
        Query the RAG assistant.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            Tuple of (answer string, retrieval metrics dict)
        """
        # Validate input
        if not input or not input.strip():
            return "Please enter a valid question.", {"error": "Empty input"}
        
        try:
            # Classify query type for domain-specific handling
            query_type = classify_query(input)
            
            # Extract severity if mentioned
            severity = extract_severity(input)
            
            # Rewrite follow-up questions into standalone queries for better retrieval
            retrieval_query = self._rewrite_query(input.strip())
            
            # Expand query with domain-specific synonyms for better recall
            expanded_query = expand_query(retrieval_query)
            
            # Search using hybrid approach (vector + keyword)
            if self.use_hybrid_search and hasattr(self.vector_db, '_hybrid_search'):
                search_results = self.vector_db._hybrid_search(expanded_query, n_results=n_results)
            else:
                search_results = self.vector_db.search(expanded_query, n_results=n_results)
            
            # Validate search results format
            if not isinstance(search_results, dict) or "documents" not in search_results:
                logger.error(f"Unexpected search result format: {type(search_results)}")
                return "Error retrieving documents. Please try again.", {"error": "Invalid search format"}
            
            documents = search_results.get("documents", [])
            distances = search_results.get("distances", [])
            
            # Calculate confidence scores
            confidence_scores = [1 - (d / 2) for d in distances] if distances else []
            
            # Filter by confidence threshold
            filtered_docs = []
            filtered_confidence = []
            for doc, conf in zip(documents, confidence_scores):
                if conf >= self.confidence_threshold:
                    filtered_docs.append(doc)
                    filtered_confidence.append(conf)
            
            # Calculate retrieval quality
            quality = self._calculate_retrieval_quality(confidence_scores)
            
            # Log retrieval metrics with domain-specific info
            retrieval_metric = {
                "timestamp": datetime.now().isoformat(),
                "query": input,
                "retrieval_query": retrieval_query,
                "expanded_query": expanded_query,
                "query_type": query_type,
                "severity_level": severity,
                "search_method": search_results.get("search_method", "vector"),
                "total_retrieved": len(documents),
                "filtered_by_threshold": len(filtered_docs),
                "confidence_scores": confidence_scores,
                "avg_confidence": quality["avg_confidence"],
                "quality_level": quality["quality_level"]
            }
            self.retrieval_log.append(retrieval_metric)
            
            # Update quality metrics
            self.quality_metrics["total_queries"] += 1
            all_scores = self.quality_metrics.get("all_scores", [])
            all_scores.extend(confidence_scores)
            self.quality_metrics["all_scores"] = all_scores
            self.quality_metrics["avg_confidence"] = sum(all_scores) / len(all_scores) if all_scores else 0
            self.quality_metrics["high_confidence_queries"] += 1 if quality["avg_confidence"] >= 0.8 else 0
            self.quality_metrics["low_confidence_queries"] += 1 if quality["avg_confidence"] < 0.5 else 0
            
            # Use filtered documents or fall back to all if none pass threshold
            context_docs = filtered_docs if filtered_docs else documents[:1]
            context = "\n\n".join(context_docs)
            
            if not context.strip():
                context = "No relevant information found in the knowledge base."
                confidence_to_display = 0
            else:
                confidence_to_display = quality["avg_confidence"]
            
            # Call the LLM chain with the context and question
            conversation_history_str = "\n".join(self.conversation_history[-self.max_history:])
            answer = self.chain.invoke({
                "context": context, 
                "question": input, 
                "conversation_history": conversation_history_str
            })
            
            # Add confidence indicator to response
            confidence_badge = self._get_confidence_badge(confidence_to_display)
            answer_with_confidence = f"{answer}\n\n---\n**Retrieval Quality:** {confidence_badge} ({confidence_to_display:.1%})"
            
            # Update conversation history with limit
            self.conversation_history.append(f"Human: {input}")
            self.conversation_history.append(f"AI: {answer}")
            
            # Trim history if it exceeds max
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
            
            logger.info(f"Query answered: {input[:50]}... (Confidence: {confidence_to_display:.1%})")
            return answer_with_confidence, retrieval_metric
            
        except Exception as e:
            logger.error(f"Error during query: {str(e)}", exc_info=True)
            return f"Error processing query: {str(e)}", {"error": str(e)}
    
    def get_retrieval_stats(self) -> Dict:
        """Get overall retrieval quality statistics.
        
        Returns:
            Dictionary with aggregated quality metrics
        """
        return {
            "total_queries": self.quality_metrics["total_queries"],
            "avg_confidence": round(self.quality_metrics["avg_confidence"], 2),
            "high_confidence_queries": self.quality_metrics["high_confidence_queries"],
            "low_confidence_queries": self.quality_metrics["low_confidence_queries"],
            "high_confidence_rate": round(
                self.quality_metrics["high_confidence_queries"] / max(1, self.quality_metrics["total_queries"]),
                2
            )
        }


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
            elif question.lower() == "stats":
                # Display retrieval quality statistics
                stats = assistant.get_retrieval_stats()
                print("\n=== Retrieval Quality Statistics ===")
                print(f"Total Queries: {stats['total_queries']}")
                print(f"Average Confidence: {stats['avg_confidence']:.1%}")
                print(f"High Confidence Rate: {stats['high_confidence_rate']:.1%}")
                print(f"Queries with High Confidence (>=80%): {stats['high_confidence_queries']}")
                print(f"Queries with Low Confidence (<50%): {stats['low_confidence_queries']}")
                print()
            else:
                result, metrics = assistant.query(question)
                print(result)

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()
