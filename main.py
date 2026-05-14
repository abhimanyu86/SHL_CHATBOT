"""
main.py
-------
FastAPI application.
Two endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → {"reply": ..., "recommendations": [...], "end_of_conversation": ...}

On startup, loads the FAISS index and embedding model into memory.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager
import uvicorn
import os

from retriever import search, build_query_from_history, _load as load_retriever
from agent import get_reply


# ─── Pydantic models (defines the API schema) ────────────────────────────────

class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v):
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def messages_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("messages list cannot be empty")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ─── Lifespan: load heavy models once at startup ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server starts.
    Loads embedding model + FAISS index into memory.
    This takes ~30-60s on first cold start — that's expected.
    """
    print("Starting up — loading retriever (model + FAISS index)...")
    load_retriever()
    print("Retriever ready. Server is up.")
    yield
    print("Shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL Individual Test Solutions",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Stateless conversational endpoint.
    Receives full conversation history, returns next agent reply.
    """
    messages = [m.model_dump() for m in request.messages]

    # Guard: max 8 turns total (user + assistant combined)
    if len(messages) > 8:
        raise HTTPException(
            status_code=400,
            detail="Conversation exceeds maximum of 8 turns.",
        )

    # Step 1: Build search query from all user messages in history
    query = build_query_from_history(messages)

    # Step 2: Retrieve top-15 relevant catalog items via FAISS
    catalog_items = search(query, top_k=15)

    # Step 3: Single LLM call — classify + respond + recommend
    result = get_reply(messages, catalog_items)

    return ChatResponse(**result)


# ─── Local dev entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # set True only during development
    )