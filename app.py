import sys
import os
import io
import shutil
import streamlit as st
import threading
from time import sleep

# Import the refactored main logic
from main import parse_args, init_pipeline

# Configure the Streamlit page
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# TERMINAL OUTPUT REDIRECTOR
# -----------------------------------------------------------------------------
class StreamlitRedirect(io.StringIO):
    """
    A custom StringIO class that captures print() statements from the backend
    and routes them live to a Streamlit element.
    """
    def __init__(self, st_element):
        super().__init__()
        self.st_element = st_element
        self.output = ""

    def write(self, string):
        # We write to the internal buffer
        super().write(string)
        # We also append to our string and update the Streamlit UI immediately
        self.output += string
        # We use a code block to preserve terminal formatting/spacing
        self.st_element.code(self.output, language="log")

    def flush(self):
        pass

# -----------------------------------------------------------------------------
# APP STATE & INITIALIZATION
# -----------------------------------------------------------------------------

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_pipeline():
    # We create a dummy args object since we aren't using CLI args in the UI
    class Args:
        data_dir = "data/uploaded_docs"
        index_path = "faiss_index_uploaded"
        model = "groq-llama3"
        k = 3
        debug = False
        question = None
    
    return init_pipeline(Args())

def process_uploaded_files(uploaded_files):
    """Saves uploaded files to disk and forces a re-index of all accumulated files."""
    upload_dir = "data/uploaded_docs"
    index_dir = "faiss_index_uploaded"
    
    os.makedirs(upload_dir, exist_ok=True)
    
    # We only delete the index so that the backend is forced to re-read
    # ALL files in the upload_dir (both old and new) and build a complete new index!
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
        
    # Save the new files
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

# -----------------------------------------------------------------------------
# SIDEBAR (FILE UPLOADING & TERMINAL OUTPUT)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("📁 Document Upload")
    st.markdown("Upload your PDFs, Word docs, or Text files here.")
    
    uploaded_files = st.file_uploader("Select files", accept_multiple_files=True, type=['pdf', 'txt', 'docx'])
    
    if st.button("Process Documents", type="primary"):
        if uploaded_files:
            with st.spinner("Preparing files..."):
                process_uploaded_files(uploaded_files)
                # Clear session state so it rebuilds the pipeline
                st.session_state.pop("pipeline_loaded", None)
                st.session_state.pop("qa_chain", None)
                st.session_state.messages = []  # Clear chat history for the new context
        else:
            st.warning("Please upload at least one file first.")
    
    st.divider()
    
    st.title("📚 Current Documents")
    has_files = os.path.exists("data/uploaded_docs") and len(os.listdir("data/uploaded_docs")) > 0
    if has_files:
        st.markdown("The following files are currently in the database:")
        for f in sorted(os.listdir("data/uploaded_docs")):
            st.markdown(f"- `{f}`")
    else:
        st.markdown("_No documents uploaded yet._")
        
    st.divider()
    
    st.title("⚙️ Backend Systems")
    st.markdown("This panel shows the exact terminal output from the Python backend.")
    
    with st.expander("Terminal Logs", expanded=True):
        # We use a container with a fixed height to make it vertically scrollable!
        log_container = st.container(height=400)
        log_placeholder = log_container.empty()
        
        # Check if we have files in the upload directory
        has_files = os.path.exists("data/uploaded_docs") and len(os.listdir("data/uploaded_docs")) > 0
        
        if not has_files:
            log_placeholder.code("Waiting for documents...\nPlease upload files and click 'Process Documents'.", language="log")
        else:
            # We do the pipeline loading inside the sidebar so the logs show up here!
            if "pipeline_loaded" not in st.session_state:
                # Hijack the standard output to point to our UI element
                old_stdout = sys.stdout
                sys.stdout = StreamlitRedirect(log_placeholder)
                
                try:
                    # This will trigger all the print() statements in main.py
                    st.session_state.qa_chain = get_pipeline()
                    st.session_state.pipeline_loaded = True
                finally:
                    # Always restore the original stdout so we don't break things!
                    sys.stdout = old_stdout
    
            else:
                # If already loaded, just show a ready message in the terminal log
                log_placeholder.code("============================================================\n  PIPELINE READY — Let's ask some questions!\n============================================================\n✅ Vector Store Loaded\n✅ LLM Connected", language="log")


# -----------------------------------------------------------------------------
# MAIN CHAT INTERFACE
# -----------------------------------------------------------------------------
st.title("🤖 RAG Research Assistant")

has_files = os.path.exists("data/uploaded_docs") and len(os.listdir("data/uploaded_docs")) > 0

if not has_files:
    st.info("👈 Please upload your documents in the sidebar to get started.")
else:
    st.markdown("Ask questions based on your uploaded documents. The AI will retrieve the most relevant chunks and synthesize an answer.")
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # React to user input
    if prompt := st.chat_input("What is the main topic of these documents?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
    
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                try:
                    # Query the RAG chain
                    qa_chain = st.session_state.qa_chain
                    result = qa_chain.invoke({"query": prompt})
                    
                    answer = result["result"]
                    source_docs = result.get("source_documents", [])
                    
                    # Format the sources beautifully
                    source_str = ""
                    if source_docs:
                        source_str = "\n\n**📚 Sources:**\n"
                        seen_sources = set()
                        for doc in source_docs:
                            source = doc.metadata.get("source", "unknown")
                            page = doc.metadata.get("page", "")
                            page_info = f", page {page}" if page != "" else ""
                            source_key = f"{source}{page_info}"
                            
                            if source_key not in seen_sources:
                                source_str += f"- `{source_key}`\n"
                                seen_sources.add(source_key)
                    
                    final_response = answer + source_str
                    
                    # Display the result
                    message_placeholder.markdown(final_response)
                    
                    # Add to history
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    
                except Exception as e:
                    error_msg = f"❌ **Error:** {str(e)}"
                    message_placeholder.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
