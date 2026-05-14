"""
catalog_loader.py
-----------------
Downloads the SHL catalog JSON once and caches it locally.
Also builds a clean text representation for each item
so sentence-transformers can embed it meaningfully.
"""

import json
import os
import re 
import requests
from dotenv import load_dotenv

load_dotenv()

CATALOG_URL = os.getenv(
    "CATALOG_URL",
    "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json",
)
LOCAL_CACHE = "catalog.json"


def download_catalog() -> list[dict]:
    print(f"Downloading catalog from {CATALOG_URL} ...")
    resp = requests.get(CATALOG_URL, timeout=30)
    resp.raise_for_status()

    # Use strict=False to allow control characters in JSON strings
    raw_text = resp.content.decode("utf-8", errors="replace")
    data = json.loads(raw_text, strict=False)

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("results", "products", "data", "items"):
            if key in data:
                items = data[key]
                break
        else:
            items = next((v for v in data.values() if isinstance(v, list)), [])
    else:
        items = []

    print(f"Downloaded {len(items)} items.")
    with open(LOCAL_CACHE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    return items


def load_catalog() -> list[dict]:
    """
    Load catalog from local cache if exists, else download.
    Returns cleaned list of assessment dicts.
    """
    if os.path.exists(LOCAL_CACHE):
        with open(LOCAL_CACHE) as f:
            items = json.load(f)
        print(f"Loaded {len(items)} catalog items from cache.")
    else:
        items = download_catalog()

    return [clean_item(item) for item in items if is_valid(item)]


def is_valid(item: dict) -> bool:
    """Keep only items that have a name and a URL."""
    return bool(item.get("name")) and bool(item.get("link"))


def clean_item(item: dict) -> dict:
    """
    Normalise field names and build a rich text string
    used for embedding. Every field we care about ends up
    in embed_text so FAISS can match on any of them.
    """
    name        = item.get("name", "").strip()
    url         = item.get("link", "").strip()
    description = item.get("description", "").strip()
    duration    = item.get("duration", item.get("duration_raw", "")).strip()
    languages   = item.get("languages", [])
    job_levels  = item.get("job_levels", [])
    keys        = item.get("keys", [])          # test-type labels
    remote      = item.get("remote", "")
    adaptive    = item.get("adaptive", "")

    # Build test_type code string from keys
    # A=Ability, K=Knowledge, P=Personality, B=Biodata/SJT, S=Simulation,
    # C=Competency, D=Development
    key_map = {
        "Ability & Aptitude":               "A",
        "Knowledge & Skills":               "K",
        "Personality & Behavior":           "P",
        "Biodata & Situational Judgment":   "B",
        "Simulations":                      "S",
        "Competencies":                     "C",
        "Development & 360":                "D",
        "Assessment Exercises":             "E",
    }
    test_type_codes = list(dict.fromkeys(
        key_map[k] for k in keys if k in key_map
    ))

    # Rich text for embedding — include everything meaningful
    parts = [
        f"Name: {name}",
        f"Description: {description}" if description else "",
        f"Test types: {', '.join(keys)}" if keys else "",
        f"Job levels: {', '.join(job_levels)}" if job_levels else "",
        f"Languages: {', '.join(languages[:5])}" if languages else "",
        f"Duration: {duration}" if duration else "",
        f"Remote: {remote}",
        f"Adaptive: {adaptive}",
    ]
    embed_text = " | ".join(p for p in parts if p)

    return {
        "name":             name,
        "url":              url,
        "description":      description,
        "duration":         duration,
        "languages":        languages,
        "job_levels":       job_levels,
        "keys":             keys,
        "test_type_codes":  test_type_codes,
        "remote":           remote,
        "adaptive":         adaptive,
        "embed_text":       embed_text,
    }


if __name__ == "__main__":
    items = load_catalog()
    print(f"\nSample item:\n{json.dumps(items[0], indent=2)}")