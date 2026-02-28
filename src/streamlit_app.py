import os
import sys

import streamlit as st

from app import RAGAssistant, load_documents


@st.cache_resource(show_spinner=False)
def get_assistant() -> RAGAssistant:
	assistant = RAGAssistant()
	documents = load_documents()
	assistant.add_documents(documents)
	return assistant


def _ensure_streamlit_runtime() -> None:
	"""Re-exec under Streamlit when run as a plain Python script."""
	if os.getenv("_STREAMLIT_AUTO_RUN") == "1":
		return
	base_name = os.path.basename(sys.argv[0]).lower()
	if base_name in {"streamlit", "streamlit.exe"}:
		return

	env = os.environ.copy()
	env["_STREAMLIT_AUTO_RUN"] = "1"
	args = [
		sys.executable,
		"-m",
		"streamlit",
		"run",
		os.path.abspath(__file__),
		"--server.headless",
		"false",
	]
	args.extend(sys.argv[1:])
	os.execvpe(sys.executable, args, env)


def main() -> None:
	st.set_page_config(
		page_title="Hi I'm Avery, Your Smart Cybersecurity AI Assistant",
		page_icon="✅",
		layout="centered",
	)

	st.title("Hi I'm Avery, Your Smart Cybersecurity AI Assistant")
	st.caption("Ask questions about the documents in the data folder.")
	st.subheader("Ask a question")
	st.write("Type your question below and press Enter.")

	with st.sidebar:
		st.header("Settings")
		n_results = st.slider("Number of sources", min_value=1, max_value=8, value=3)
		clear_chat = st.button("Clear chat")

	if clear_chat:
		st.session_state.pop("chat_history", None)

	if "chat_history" not in st.session_state:
		st.session_state.chat_history = []

	try:
		assistant = get_assistant()
	except Exception as exc:
		st.error(str(exc))
		st.info("Ensure you have a valid API key in your .env file.")
		return

	for role, message in st.session_state.chat_history:
		with st.chat_message(role):
			st.markdown(message)

	question = st.chat_input("Ask a question")
	if question:
		st.session_state.chat_history.append(("user", question))
		with st.chat_message("user"):
			st.markdown(question)

		with st.chat_message("assistant"):
			with st.spinner("Thinking..."):
				answer = assistant.query(question, n_results=n_results)
				st.markdown(answer)
		st.session_state.chat_history.append(("assistant", answer))

	st.divider()
	st.caption("Edit src/streamlit_app.py to customize this app.")


if __name__ == "__main__":
	_ensure_streamlit_runtime()
	main()
