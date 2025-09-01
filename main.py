# backend.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import asyncio
from db import init_db, save_chat
from chat import get_bot_response

# Initialize DB
init_db()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== API Models =====
class ChatRequest(BaseModel):
    query: str

# Create a lock for DB operations (SQLite is not fully async-safe)
db_lock = asyncio.Lock()

# ===== Routes =====
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(req: ChatRequest):
    # Run the bot response in a separate thread if it's blocking
    answer = await asyncio.to_thread(get_bot_response, req.query)

    # Ensure DB writes don’t collide
    async with db_lock:
        await asyncio.to_thread(save_chat, req.query, answer)

    return {"answer": answer}