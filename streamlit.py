import streamlit as st
import requests
import uuid
import time
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
BACKEND_URL = "http://localhost:8000"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")

# Initialize OpenAI client
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Please add it to your .env file.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Production RAG Service", layout="wide")
st.title("📄 Production-Ready RAG Service")
st.markdown("Upload a document and ask questions about it. Each session is isolated.")

# --- Session State Management ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.file_uploaded = False
    st.session_state.last_uploaded_file_name = None  # Track file name instead of file object
    st.session_state.upload_in_progress = False
    st.session_state.uploader_key = 0  # Key to reset the uploader

# Display Session ID
st.sidebar.title("Session Information")
st.sidebar.markdown(f"**Session ID:**")
st.sidebar.code(st.session_state.session_id)
st.sidebar.markdown("---")

# --- File Uploader ---
st.sidebar.header("Upload Document")

# Use a key that we can increment to reset the uploader
uploaded_file = st.sidebar.file_uploader(
    "Choose a file (PDF, TXT, DOCX, etc.)",
    type=None,
    key=f"file_uploader_{st.session_state.uploader_key}"
)

# Only process if file is uploaded AND it's different from last uploaded file AND no upload is in progress
if (uploaded_file is not None and 
    uploaded_file.name != st.session_state.last_uploaded_file_name and
    not st.session_state.upload_in_progress):
    
    st.session_state.upload_in_progress = True
    st.session_state.last_uploaded_file_name = uploaded_file.name
    
    with st.spinner(f"Processing `{uploaded_file.name}`... This may take a moment."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {"session_id": st.session_state.session_id}
        
        start_time = time.time()
        try:
            response = requests.post(f"{BACKEND_URL}/upload/", files=files, data=data, timeout=300)
            end_time = time.time()
            
            if response.status_code == 200:
                st.session_state.file_uploaded = True
                st.sidebar.success(f"File processed in {end_time - start_time:.2f} seconds.")
                st.sidebar.info("You can now ask questions about the document.")
                
            else:
                st.sidebar.error(f"Error: {response.status_code} - {response.text}")
                # Reset on error so user can retry
                st.session_state.last_uploaded_file_name = None
                
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Connection error: {e}")
            # Reset on error so user can retry
            st.session_state.last_uploaded_file_name = None
            
        finally:
            st.session_state.upload_in_progress = False

# Show current status
if st.session_state.file_uploaded and st.session_state.last_uploaded_file_name:
    st.sidebar.success(f"✅ {st.session_state.last_uploaded_file_name} uploaded successfully")
    
if st.session_state.upload_in_progress:
    st.sidebar.info("⏳ Upload in progress...")

# --- Clear Upload Button ---
if st.session_state.file_uploaded:
    if st.sidebar.button("🗑️ Clear Uploaded Document"):
        # Reset all upload-related states
        st.session_state.file_uploaded = False
        st.session_state.last_uploaded_file_name = None
        st.session_state.messages = []  # Clear chat history
        st.session_state.uploader_key += 1  # Increment key to reset uploader
        st.sidebar.success("Document cleared. You can upload a new one.")
        st.rerun()  # Force a rerun to reset the UI

# --- Chat Interface ---
st.header("Chat with your Document")

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about your document..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if a file has been uploaded
    if not st.session_state.file_uploaded:
        with st.chat_message("assistant"):
            st.warning("Please upload a document before asking questions.")
        st.stop()

    with st.spinner("Thinking..."):
        try:
            # 1. Retrieve context from the RAG backend
            query_payload = {"query": prompt, "session_id": st.session_state.session_id}
            response = requests.post(f"{BACKEND_URL}/query/", json=query_payload, timeout=120)

            if response.status_code != 200:
                st.error(f"Backend Error: {response.text}")
                st.stop()
            
            retrieved_context = response.json()["context"]
            
            if not retrieved_context:
                with st.chat_message("assistant"):
                    st.warning("Could not find relevant context in the document for your query.")
                st.stop()

            context_str = "\n\n---\n\n".join([doc['text'] for doc in retrieved_context])
            
            # Optional: Show retrieved context in expander
            with st.expander("See Retrieved Context", expanded=False):
                st.json(retrieved_context)

            # 2. Generate final response using OpenAI LLM with the context
            system_prompt = (
                "You are an expert assistant. Use the following context to answer the user's question. "
                "If the answer is not found in the context, state that you cannot answer based on the provided document. "
                "Do not make up information.\n\n"
                "CONTEXT:\n"
                f"{context_str}"
            )
            
            messages_for_llm = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            llm_response = client.chat.completions.create(
                model=OPENAI_LLM_MODEL,
                messages=messages_for_llm,
                temperature=0.2,
            )
            final_answer = llm_response.choices[0].message.content

            # Display assistant response and add to chat history
            with st.chat_message("assistant"):
                st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except requests.exceptions.RequestException as e:
            with st.chat_message("assistant"):
                st.error(f"Failed to connect to the backend: {e}")