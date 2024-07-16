# app.py

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

#os.environ["OPENAI_API_KEY"] = #constants.APIKEY
os.getenv('OPENAI_API_KEY')
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the file
file_path = os.path.join(script_dir, 'RESUME.pdf')
# Load PDF Documents
loader = PyPDFLoader(file_path)
documents = loader.load()

# Chunk the Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Generate Embeddings
embeddings = OpenAIEmbeddings()

# Create Vector Store
vector_store = FAISS.from_documents(docs, embeddings)

# Set Up Retriever
retriever = vector_store.as_retriever()

# Initialize QA Chain
llm = ChatOpenAI(model_name="gpt-4o")
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Define API endpoint for handling queries
@app.route('/api/query', methods=['POST'])
def ask_bot():
    data = request.get_json()
    query = data['query']
    answer = qa_chain.run(query)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5100))
    app.run(host='0.0.0.0', port=port)

