from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables from a .env file
load_dotenv()

# Initialize the language model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the documents from a PDF file
def load_documents(filepath):
    text = []
    try:
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                try:
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
                    else:
                        print(f"Page {page_num + 1} is empty or could not be read.")
                except Exception as e:
                    print(f"Error reading page {page_num + 1}: {e}")
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

# Load documents
documents = load_documents(os.getenv('PDF_FILE_PATH', r'D:\infih\chatbot\298.pdf'))

# Encode all documents
encoded_docs = [model.encode(doc, convert_to_tensor=True) for doc in documents]

@app.route('/rag', methods=['POST'])
def rag():
    data = request.json
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    # Encode the prompt
    encoded_prompt = model.encode(prompt, convert_to_tensor=True)
    best_doc = None
    highest_score = -1
    
    # Find the best matching document
    for doc, encoded_doc in zip(documents, encoded_docs):
        score = util.pytorch_cos_sim(encoded_prompt, encoded_doc)[0][0].item()
        if score > highest_score:
            highest_score = score
            best_doc = doc
    
    if best_doc:
        response_text = "Based on the context, here's the best document match:\n" + best_doc[:min(len(best_doc), 500)] + "..."
    else:
        response_text = "No matching document found."

    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)
