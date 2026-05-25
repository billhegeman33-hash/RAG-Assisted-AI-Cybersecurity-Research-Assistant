import os
import sys
import logging

import streamlit as st

# Add src directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import RAGAssistant, load_documents

# Suppress Streamlit logger warnings during initialization
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)


@st.cache_resource(show_spinner=False)
def get_assistant() -> RAGAssistant:
	assistant = RAGAssistant()
	documents = load_documents()
	assistant.add_documents(documents)
	return assistant


def _ensure_streamlit_runtime() -> None:
	"""Re-exec under Streamlit when run as a plain Python script."""
	try:
		from streamlit.runtime import exists
		if exists():
			return
	except Exception:
		pass
	
	if os.getenv("_STREAMLIT_AUTO_RUN") == "1":
		return
	
	base_name = os.path.basename(sys.argv[0]).lower()
	if base_name in {"streamlit", "streamlit.exe"}:
		return

	env = os.environ.copy()
	env["_STREAMLIT_AUTO_RUN"] = "1"
	
	# Get the directory of this script
	script_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.dirname(script_dir)
	
	args = [
		sys.executable,
		"-m",
		"streamlit",
		"run",
		os.path.abspath(__file__),
		"--logger.level=debug",
	]
	args.extend(sys.argv[1:])
	
	# Change to project root before executing
	os.chdir(project_root)
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

	# Display retrieval stats after assistant is initialized
	with st.sidebar:
		st.divider()
		st.subheader("Retrieval Stats")
		try:
			stats = assistant.get_retrieval_stats()
			st.metric("Total Queries", stats["total_queries"])
			st.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")
			st.metric("High Confidence Rate", f"{stats['high_confidence_rate']:.1%}")
		except Exception as e:
			st.caption(f"Stats unavailable: {str(e)}")

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
				answer, metrics = assistant.query(question, n_results=n_results)
				st.markdown(answer)
				
				# Show query analysis
				with st.expander("Query Analysis"):
					col1, col2, col3 = st.columns(3)
					with col1:
						st.write(f"**Query Type:** {metrics.get('query_type', 'general')}")
					with col2:
						severity = metrics.get('severity_level')
						if severity:
							st.write(f"**Severity:** {severity}")
					with col3:
						search_method = metrics.get('search_method', 'vector')
						st.write(f"**Search:** {search_method}")
		st.session_state.chat_history.append(("assistant", answer))

	st.divider()
	


if __name__ == "__main__":
	# Ensure we're running under Streamlit, not as plain Python
	try:
		from streamlit.runtime import exists
		if not exists():
			# Not running under Streamlit, re-exec under streamlit
			import subprocess
			import sys
			
			# Get the absolute path of this script
			script_path = os.path.abspath(__file__)
			
			# Re-exec under streamlit
			subprocess.run(
				[sys.executable, "-m", "streamlit", "run", script_path],
				check=False
			)
			sys.exit(0)
	except Exception:
		pass
	
	main()
