"""
indexer.py
----------
Embeds every catalog item using sentence-transformers
and builds a FAISS index for fast similarity search.

Run once:  python indexer.py
Saves:     faiss.index  +  index_map.json
"""

import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from catalog_loader import load_catalog

MODEL_NAME  = "all-MiniLM-L6-v2"   # 90 MB, 384-dim, fast and accurate
INDEX_FILE  = "faiss.index"
MAP_FILE    = "index_map.pkl"


def build_index():
    print("Loading catalog ...")
    items = load_catalog()
    print(f"Catalog has {len(items)} items.")

    print(f"Loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [item["embed_text"] for item in items]

    print("Embedding all items (this takes ~30-60s first time) ...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,   # cosine similarity via dot product
    )
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    print(f"Embedding dim: {dim}, total vectors: {len(embeddings)}")

    # Flat index — exact search, fast enough for ~400 items
    index = faiss.IndexFlatIP(dim)   # IP = inner product = cosine on normalised vecs
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(MAP_FILE, "wb") as f:
        pickle.dump(items, f)

    print(f"Saved index → {INDEX_FILE}")
    print(f"Saved item map → {MAP_FILE}")
    print("Done.")


if __name__ == "__main__":
    build_index()