import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session

from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')  # Change this in production

# Paths
faiss_index_path = "./faiss_index"

# 1. Load Gryork data (if not already in DB)
if not os.path.exists(faiss_index_path):
    # Load multiple text files
    documents = []
    for file in ["knowledge_base/gryork1.txt", "knowledge_base/gryork2.txt", "knowledge_base/extra.txt"]:
        if os.path.exists(file):
            loader = TextLoader(file, encoding="utf-8")
            documents.extend(loader.load())

    if not documents:
        raise FileNotFoundError("No gryork text files found!")

    # Build embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create FAISS DB
    db = FAISS.from_documents(documents, embeddings)

    # Save FAISS index locally
    db.save_local(faiss_index_path)
else:
    # Load existing FAISS DB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

# 2. Retriever
retriever = db.as_retriever()

# 3. Gemini Flash model
llm = init_chat_model(
    "gemini-2.5-flash",  # Note: Updated to gemini-1.5-flash as gemini-2.5-flash may not exist; adjust if needed
    model_provider="google_genai",
    temperature=0.8
)

system_prompt = """
You are GryBOT, a friendly AI assistant built by Gryork Engineers. Gryork is a company focused on solving liquidity challenges in the infrastructure sector.

## Core Purpose
- Answer questions about Gryork, its solutions (e.g., CWC model, GRYLINK platform), and related terms only when the user explicitly mentions Gryork, Aditya Tiwari, or Gryork-specific terms.
- For general questions (e.g., "What is CWC?") that do not mention Gryork or its specific terms, provide a concise, accurate, and general response without referencing Gryork or its context.
- For questions outside Gryork’s scope, you may politely redirect to Aditya Tiwari or Gryork Engineers if relevant, but you can also handle simple general queries.
- Avoid overusing Gryork references unless the user intends to discuss Gryork.

## Style
- Keep responses short, warm, and conversational. Use different colorful emojis when appropriate to match the context.
- Be clear and simple when discussing technical topics, especially infrastructure or financing concepts.
- Be empathetic when addressing personal or sensitive questions.

## Details
Here’s some context about Gryork (use only when Gryork or its terms are mentioned):
- Aditya Tiwari is the founder of Gryork Engineers, a company focused on solving liquidity challenges in the infrastructure sector through innovative financing solutions.
- Gryork Engineers develops the Credit on Working Capital (CWC) model, which provides subcontractors with short-term credit backed by a Letter of Guarantee (LoG) from infrastructure companies.

## Context
{context}

Question: {question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=system_prompt
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": prompt}
)

small_talk_responses = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hi there!",
    "hey": "Hey! What’s up?",
    "good morning": "Good morning Hope your day is going well!",
    "good afternoon": "Good afternoon",
    "good evening": "Good evening",
    "bye": "See you later!",
    "goodbye": "Goodbye! Have a great day!",
    "thanks": "You're welcome!",
    "thank you": "Glad I could help!",
    "who are you": "I’m the Gryork Bot, created to help you with Gryork and beyond!",
    "what can you do": "I can answer questions, chat casually, and share information about Gryork’s services.",
}

def is_small_talk(query: str):
    q = query.lower().strip()
    return q in small_talk_responses

@app.route('/')
def index():
    session['chat_history'] = []
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    query = request.json.get('message').strip()
    chat_history = session.get('chat_history', [])

    if query.lower() in ["exit", "quit", "goodbye"]:
        response = "Goodbye!"
    elif is_small_talk(query):
        response = small_talk_responses[query.lower()]
    else:
        # Use RAG
        result = qa_chain.invoke({"question": query, "chat_history": chat_history})
        response = result["answer"]
        chat_history.append((query, response))
        session['chat_history'] = chat_history

    return jsonify({'response': response})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)