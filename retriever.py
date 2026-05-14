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

ALWAYS_INCLUDE_SLUGS = [
    "occupational-personality-questionnaire-opq32r",
    "shl-verify-interactive-g",
]


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


def keyword_score(item: dict, query: str) -> float:
    """Keyword overlap score between query words and item text."""
    query_words = set(query.lower().split())
    item_text = (item.get("embed_text", "") + " " + item.get("name", "")).lower()
    matches = sum(1 for w in query_words if w in item_text)
    return matches / max(len(query_words), 1)


def search(query: str, top_k: int = 20) -> list:
    """
    Hybrid search: semantic (FAISS) + keyword overlap.
    Returns top_k items sorted by combined score.
    """
    _load()

    vec = _model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    # Get top 30 semantic candidates
    scores, indices = _index.search(vec, 30)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = dict(_items[idx])
        item["_semantic_score"] = float(score)
        item["_keyword_score"] = keyword_score(item, query)
        # 70% semantic + 30% keyword
        item["_score"] = 0.7 * item["_semantic_score"] + 0.3 * item["_keyword_score"]
        candidates.append(item)

    candidates.sort(key=lambda x: x["_score"], reverse=True)
    return candidates[:top_k]


def get_always_include_items() -> list:
    """Always return OPQ32r and Verify G+ so agent can decide to include them."""
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
    """
    Build enriched search query from conversation history.
    Extracts role, tech, and purpose signals for better retrieval.
    """
    user_texts = [m["content"] for m in messages if m.get("role") == "user"]
    full_text = " ".join(user_texts)

    signals = []

    role_keywords = [
        "developer", "engineer", "analyst", "manager", "director",
        "executive", "sales", "contact centre", "contact center", "agent",
        "operator", "administrator", "nurse", "accountant", "graduate",
        "intern", "leadership", "cxo", "recruiter", "coordinator",
    ]
    tech_keywords = [
        "java", "python", "sql", "aws", "docker", "spring", "react",
        "angular", "javascript", "rust", "linux", "networking", ".net",
        "excel", "word", "powerpoint", "sap", "salesforce", "tableau",
    ]
    purpose_keywords = [
        "selection", "development", "screening", "training", "leadership",
        "cognitive", "personality", "safety", "customer service", "numerical",
        "verbal", "situational", "reasoning", "knowledge", "simulation",
        "dependability", "bilingual", "language", "hipaa", "finance",
    ]

    for kw in role_keywords + tech_keywords + purpose_keywords:
        if kw.lower() in full_text.lower():
            signals.append(kw)

    return full_text + " " + " ".join(signals)
