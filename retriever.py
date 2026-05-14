import pickle
import numpy as np
import faiss
import requests
import os
from dotenv import load_dotenv

load_dotenv()

INDEX_FILE = "faiss.index"
MAP_FILE   = "index_map.pkl"
HF_TOKEN   = os.getenv("HF_TOKEN", "")
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

_index = None
_items = None


def _load():
    global _index, _items
    if _index is None:
        print("Loading FAISS index ...")
        _index = faiss.read_index(INDEX_FILE)
        with open(MAP_FILE, "rb") as f:
            _items = pickle.load(f)
        print(f"Index ready — {_index.ntotal} vectors.")


def _embed_via_api(text: str) -> np.ndarray:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    resp = requests.post(
        HF_API_URL,
        headers=headers,
        json={"inputs": text},
        timeout=20,
    )
    resp.raise_for_status()
    vec = np.array(resp.json(), dtype="float32")
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.reshape(1, -1)


def search(query: str, top_k: int = 15) -> list:
    _load()
    vec = _embed_via_api(query)
    scores, indices = _index.search(vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = dict(_items[idx])
        item["_score"] = float(score)
        results.append(item)
    return results


def build_query_from_history(messages: list) -> str:
    user_texts = [
        m["content"]
        for m in messages
        if m.get("role") == "user"
    ]
    return " ".join(user_texts)