"""
retriever_render.py
-------------------
Lightweight retriever for deployment on Render free tier (512MB RAM).
Uses HuggingFace Inference API for embeddings instead of loading
sentence-transformers locally — saves ~800MB RAM at runtime.
"""

import pickle
import numpy as np
import faiss
import requests as http_requests
import os
from dotenv import load_dotenv

load_dotenv()

INDEX_FILE = "faiss.index"
MAP_FILE   = "index_map.pkl"
HF_TOKEN   = os.getenv("HF_TOKEN", "")
HF_API_URL = HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

_index = None
_items = None

ALWAYS_INCLUDE_SLUGS = [
    "occupational-personality-questionnaire-opq32r",
    "shl-verify-interactive-g",
]


def _load():
    global _index, _items
    if _index is None:
        print("Loading FAISS index ...")
        _index = faiss.read_index(INDEX_FILE)
        with open(MAP_FILE, "rb") as f:
            _items = pickle.load(f)
        print(f"Index ready — {_index.ntotal} vectors.")


def _embed(text: str) -> np.ndarray:
    """Call HuggingFace Inference API to get embeddings."""
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    resp = http_requests.post(
        HF_API_URL,
        headers=headers,
        json={"inputs": text},
        timeout=20,
    )
    resp.raise_for_status()
    vec = np.array(resp.json(), dtype="float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.reshape(1, -1)


def keyword_score(item: dict, query: str) -> float:
    query_words = set(query.lower().split())
    item_text = (item.get("embed_text", "") + " " + item.get("name", "")).lower()
    matches = sum(1 for w in query_words if w in item_text)
    return matches / max(len(query_words), 1)


def search(query: str, top_k: int = 20) -> list:
    _load()
    vec = _embed(query)
    scores, indices = _index.search(vec, 30)
    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = dict(_items[idx])
        item["_semantic_score"] = float(score)
        item["_keyword_score"] = keyword_score(item, query)
        item["_score"] = 0.7 * item["_semantic_score"] + 0.3 * item["_keyword_score"]
        candidates.append(item)
    candidates.sort(key=lambda x: x["_score"], reverse=True)
    return candidates[:top_k]


def get_always_include_items() -> list:
    _load()
    result = []
    seen = set()
    for item in _items:
        for slug in ALWAYS_INCLUDE_SLUGS:
            if slug in item.get("url", "") and item["url"] not in seen:
                result.append(dict(item))
                seen.add(item["url"])
    return result


def build_query_from_history(messages: list) -> str:
    user_texts = [m["content"] for m in messages if m.get("role") == "user"]
    full_text = " ".join(user_texts)
    signals = []
    role_keywords = ["developer", "engineer", "analyst", "manager", "director",
                     "executive", "sales", "contact centre", "contact center",
                     "agent", "operator", "graduate", "leadership", "cxo"]
    tech_keywords = ["java", "python", "sql", "aws", "docker", "spring", "react",
                     "angular", "javascript", "rust", "linux", "networking", ".net",
                     "excel", "word"]
    purpose_keywords = ["selection", "development", "screening", "leadership",
                        "cognitive", "personality", "safety", "customer service",
                        "numerical", "verbal", "situational", "finance", "hipaa"]
    for kw in role_keywords + tech_keywords + purpose_keywords:
        if kw.lower() in full_text.lower():
            signals.append(kw)
    return full_text + " " + " ".join(signals)


def _load_for_health():
    """Called at startup — just loads the index, no model needed."""
    _load()