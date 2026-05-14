import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_FILE = "faiss.index"
MAP_FILE   = "index_map.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_index = None
_items = None


def _load():
    global _model, _index, _items
    if _model is None:
        print("Loading embedding model ...")
        _model = SentenceTransformer(MODEL_NAME)
    if _index is None:
        print("Loading FAISS index ...")
        _index = faiss.read_index(INDEX_FILE)
        with open(MAP_FILE, "rb") as f:
            _items = pickle.load(f)
        print(f"Index ready — {_index.ntotal} vectors.")


def search(query: str, top_k: int = 15) -> list:
    _load()
    vec = _model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
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