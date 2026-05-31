import os
import sys
import logging

import streamlit as st

# Must be AFTER `import streamlit` — Streamlit configures its own loggers on import,
# so any filter added before gets wiped. Target the exact logger that emits the warning.
class _SuppressScriptRunContextWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "missing ScriptRunContext" not in record.getMessage()

logging.getLogger(
    "streamlit.runtime.scriptrunner_utils.script_run_context"
).addFilter(_SuppressScriptRunContextWarning())

# Add src directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import RAGAssistant, load_documents


@st.cache_resource(show_spinner=False)
def get_assistant() -> RAGAssistant:
    assistant = RAGAssistant()
    documents = load_documents()
    assistant.add_documents(documents)
    return assistant


def main() -> None:
    st.set_page_config(
        page_title="Avery - Cybersecurity AI Assistant",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ Avery - Cybersecurity AI Assistant")
    st.markdown("Ask questions about cybersecurity, AI safety, and related topics.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar - Settings
    with st.sidebar:
        st.header("Settings")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Initialize assistant
    try:
        assistant = get_assistant()
    except Exception as exc:
        st.error(f"Error initializing assistant: {str(exc)}")
        st.info("Make sure you have a valid API key in your .env file")
        st.stop()

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

    # Chat input and response
    user_input = st.chat_input("Ask a question about cybersecurity...")

    if user_input:
        # Add user message to history
        st.session_state.chat_history.append(("user", user_input))

        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, metrics = assistant.query(user_input)
                    st.write(answer)

                    # Show basic metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{metrics.get('avg_confidence', 0):.1%}")
                    with col2:
                        st.metric("Sources Found", metrics.get('total_retrieved', 0))
                    with col3:
                        st.metric("Query Type", metrics.get('query_type', 'general'))
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    answer = f"Error: {str(e)}"

        # Add assistant response to history
        st.session_state.chat_history.append(("assistant", answer))


if __name__ == "__main__":
    main()
