import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# 1. Load multiple text documents
files = ["knowledge_base/gryork1.txt", "knowledge_base/gryork2.txt", "knowledge_base/extra.txt"]  # Add more if needed
docs = []

for file in files:
    loader = TextLoader(file, encoding="utf-8")
    docs.extend(loader.load())   # extend instead of append

# 2. Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 3. Store in FAISS
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

# 4. LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# 5. QA Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 6. Query
query = "what is cwc?"
result = qa_chain.run(query)

print("Answer:", result)