import streamlit as st
from app import RAGAssistant, load_documents

st.title("Avery: Your Smart AI Cybersecurity Assistant")

# Initialize RAG Assistant (and load documents) only once
@st.cache_resource
def initialize_rag_assistant():
    assistant = RAGAssistant()
    sample_docs = load_documents()
    assistant.add_documents(sample_docs)
    return assistant

assistant = initialize_rag_assistant()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = assistant.query(prompt)
            st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
