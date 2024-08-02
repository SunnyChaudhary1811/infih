import streamlit as st
import os
import requests
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import PyPDF2
import time

# Load environment variables from a .env file
load_dotenv()

# Load the documents from a PDF file
def load_documents(filepath):
    text = []
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
                else:
                    st.warning(f"Page {page_num + 1} is empty or could not be read.")
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# Load documents
documents = load_documents(r'D:\infih\chatbot\298.pdf')

# Title of the Streamlit app
st.title("iPsychiatrist")

# Initialize the language model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode all documents
encoded_docs = [model.encode(doc, convert_to_tensor=True) for doc in documents]

# Initialize session state variables to store chat history
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display previous chat history
for answer, user_prompt in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
    st.write(f"User: {user_prompt}")
    st.write(f"Assistant: {answer}")

# Input for user prompt
prompt = st.text_input("Input your prompt here")

# Button to submit the prompt
if st.button("Submit Prompt"):
    if prompt:
        # Record the start time
        start = time.process_time()

        # Ensure documents are not empty
        if not documents:
            st.error("No content found in the documents.")
        else:
            # Prepare the payload with the user prompt
            payload = {'prompt': prompt, 'documents': documents}

            try:
                # URL of the Flask API endpoint
                url = 'http://127.0.0.1:5000/rag'

                # Make the POST request to the Flask API
                response = requests.post(url, json=payload)
                response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

                # Process the response from the API
                data = response.json()
                best_response = data.get('response', 'No response from API')

                # Display the response time
                st.write("Response time:", time.process_time() - start)

                # Display the user prompt and assistant response
                st.write(f"User: {prompt}")
                st.write(f"Assistant: {best_response}")

                # Update session state with the new chat history
                st.session_state["chat_answers_history"].append(best_response)
                st.session_state["user_prompt_history"].append(prompt)
                st.session_state["chat_history"].append((prompt, best_response))
                
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a prompt.")
