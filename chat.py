# chatbot_core.py
import os
import random
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from db import get_chats

# Load environment variables
load_dotenv()

# ===== Load / Build FAISS Index =====
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
knowledge_dir = "knowledge_base"
faiss_index_path = "./faiss_index"

if not os.path.exists(faiss_index_path):
    print("⚡ Building FAISS index...")
    documents = []
    for file in os.listdir(knowledge_dir):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(knowledge_dir, file), encoding="utf-8")
            documents.extend(loader.load())

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(faiss_index_path)
else:
    print("✅ Loading existing FAISS index...")
    db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever()

# ===== Gemini Flash =====
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0.8)

system_prompt = """
You are GryBOT, a friendly AI assistant built by Gryork Engineers. Gryork is a company focused on solving liquidity challenges in the infrastructure sector.

## Core Purpose
- Answer questions about Gryork, its solutions (e.g., CWC model, GRYLINK platform), and related terms only when the user explicitly mentions Gryork, Aditya Tiwari, or Gryork-specific terms.
- For general questions (e.g., "What is CWC?") that do not mention Gryork or its specific terms, provide a concise, accurate, and general response without referencing Gryork or its context.
- For questions outside Gryork’s scope, you may politely redirect to Aditya Tiwari or Gryork Engineers if relevant, but you can also handle simple general queries.
- Avoid overusing Gryork references unless the user intends to discuss Gryork.

## Style
- Keep responses short, warm, and conversational. Use different colorful emojis when appropriate to match the context 😊.
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

prompt = PromptTemplate(input_variables=["context", "question"], template=system_prompt)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": prompt},
    verbose=False
)

# ===== Small Talk =====
small_talk_responses = {
    "hi": [
        "Hello! 👋 How can I help you today?",
        "Hey there! 😃",
        "Hi! 🙌 What’s up?"
    ],
    "hello": [
        "Hi there! 🙂",
        "Hello 👋 How’s everything going?",
        "Hey! 🌟 Nice to see you."
    ],
    "hey": [
        "Hey! What’s up?",
        "Yo! 👋",
        "Hey there, how’s it going? 😎"
    ],
    "good morning": [
        "Good morning ☀️ Hope your day is going well!",
        "Morning! 🌞 Wishing you a productive day ahead.",
        "Rise and shine! 🌅"
    ],
    "good afternoon": [
        "Good afternoon 🌞",
        "Hope your afternoon is going great! 🌻",
        "Hey! ☀️ How’s your day so far?"
    ],
    "good evening": [
        "Good evening 🌙",
        "Evening! ✨ How’s everything going?",
        "Hope you had a great day! 🌆"
    ],
    "bye": [
        "See you later! 👋",
        "Bye-bye! Take care 🌸",
        "Catch you soon! 🚀"
    ],
    "goodbye": [
        "Goodbye! 👋 Have a great day!",
        "See you next time! 🌟",
        "Bye! Stay awesome 🤩"
    ],
    "thanks": [
        "You're welcome! 🙌",
        "No problem at all, happy to help! 😊",
        "You got it! 👍",
        "Always here if you need me 🙌"
    ],
    "thank you": [
        "Glad I could help! 😊",
        "Anytime! 🌟",
        "Always here to support you 🙌"
    ],
    "who are you": [
        "I’m the Gryork Bot 🤖, created to help you with Gryork and beyond!",
        "I’m a bot 🤖 created by Gryork Engineers to assist you.",
        "I’m your friendly AI assistant, here to chat and share knowledge 🌟"
    ],
    "what can you do": [
        "I can answer questions, chat casually, and share information about Gryork’s services.",
        "I can help with Gryork-related queries, or just talk about anything you’d like 🙂",
        "I can provide insights on Gryork, answer general questions, and keep you company 🤝"
    ]
}

def is_small_talk(query: str):
    return query.lower().strip() in small_talk_responses

def handle_small_talk(query: str) -> str:
    return random.choice(small_talk_responses[query.lower().strip()])

def safe_invoke(query: str):
    # Get last 10 chats from DB
    history = [(u, b) for u, b, _ in reversed(get_chats(10))]
    result = qa_chain.invoke({"question": query, "chat_history": history})
    if not result["answer"].strip():
        return "I’m here to talk about Gryork, Grylink and its work 🙂"
    return result["answer"]

def get_bot_response(query: str) -> str:
    if is_small_talk(query):
        return handle_small_talk(query)
    return safe_invoke(query)