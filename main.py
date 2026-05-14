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

if os.getenv("RENDER"):
    from retriever_render import search, build_query_from_history, _load_for_health as load_retriever, get_always_include_items
else:
    from retriever import search, build_query_from_history, _load as load_retriever, get_always_include_items
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

    # Step 1: Build enriched search query from full conversation history
    query = build_query_from_history(messages)

    # Step 2: Hybrid search — semantic + keyword, top 20 candidates
    catalog_items = search(query, top_k=20)

    # Step 3: Always append OPQ32r and Verify G+ so agent can include them
    always = get_always_include_items()
    existing_urls = {item["url"] for item in catalog_items}
    for item in always:
        if item["url"] not in existing_urls:
            catalog_items.append(item)

    # Step 4: Pre-filter/boost by job level if mentioned
    all_user_text = " ".join(m["content"] for m in messages).lower()
    level_map = {
        "entry": "Entry-Level",
        "graduate": "Graduate",
        "mid": "Mid-Professional",
        "senior": "Professional Individual Contributor",
        "manager": "Manager",
        "director": "Director",
        "executive": "Executive",
        "cxo": "Executive",
        "leadership": "Director",
    }
    matched_level = None
    for keyword, level in level_map.items():
        if keyword in all_user_text:
            matched_level = level
            break
    if matched_level:
        matching = [i for i in catalog_items if matched_level in i.get("job_levels", [])]
        others = [i for i in catalog_items if matched_level not in i.get("job_levels", [])]
        catalog_items = (matching + others)[:22]

    # Step 5: Single LLM call — classify + respond + recommend
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