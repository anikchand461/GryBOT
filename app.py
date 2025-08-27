import os
import logging
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure Gunicorn logger
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

# Load environment variables
load_dotenv()

# Initialize the QA chain once when the app starts
def init_qa_chain():
    # Check if FAISS index exists
    faiss_index_path = "faiss_index"
    if os.path.exists(faiss_index_path):
        # Load existing FAISS index
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # Load multiple text documents
        files = ["knowledge_base/gryork1.txt", "knowledge_base/gryork2.txt", "knowledge_base/extra.txt"]
        docs = []
        for file in files:
            loader = TextLoader(file, encoding="utf-8")
            docs.extend(loader.load())

        # Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Create and store in FAISS
        db = FAISS.from_documents(docs, embeddings)
        # Save the index
        db.save_local(faiss_index_path)

    retriever = db.as_retriever()

    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    
    return qa_chain

qa_chain = init_qa_chain()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON in request'}), 400
            
        query = data.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Add timeout to prevent long-running requests
        result = qa_chain.run(query)
        
        # Ensure result is a string
        if result is None:
            return jsonify({'response': 'I could not generate a response. Please try again.'})
            
        return jsonify({'response': str(result)})
    except Exception as e:
        app.logger.error(f"Error processing chat request: {str(e)}")
        # Return a user-friendly error message
        return jsonify({'error': 'An error occurred while processing your request. Please try again later.'}), 500

if __name__ == '__main__':
    # Use environment variable for port with a default of 5000
    port = int(os.environ.get('PORT', 5000))
    # Only enable debug mode in development
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)