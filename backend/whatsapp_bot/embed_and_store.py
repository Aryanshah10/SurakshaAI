"""
Citizen Fraud Shield - Embedding + ChromaDB Store Script
Step: Cleaned chunks map → Embeddings → ChromaDB

This script:
  1. Loads the cleaned chunks_map.json (with RAG relevance in metadata, not text)
  2. Loads safe_scenarios.json and merges them in
  3. Embeds everything and stores in ChromaDB

Usage:
    pip install sentence-transformers chromadb
    python embed_and_store.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
import torch
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DB_PATH = str(Path(__file__).parent.parent.parent / "chroma_db")
COLLECTION_NAME = "citizen_fraud_shield"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 32


def load_and_merge_data() -> list:
    """Load chunks_map.json and safe_scenarios.json, merge into one list."""
    chunks_path = DATA_DIR / "chunks_map.json"
    safe_path = DATA_DIR / "safe_scenarios.json"

    all_items = []

    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks from chunks_map.json")
        all_items.extend(chunks)
    else:
        print(f"WARNING: {chunks_path} not found.")

    if safe_path.exists():
        with open(safe_path, "r", encoding="utf-8") as f:
            safe_scenarios = json.load(f)
        print(f"Loaded {len(safe_scenarios)} safe scenarios from safe_scenarios.json")
        for sc in safe_scenarios:
            if "metadata" not in sc:
                sc["metadata"] = {}
            sc["metadata"]["category"] = "safe"
        all_items.extend(safe_scenarios)
    else:
        print(f"No safe_scenarios.json found at {safe_path}")

    print(f"Total items to embed: {len(all_items)}")
    return all_items


def main():
    items = load_and_merge_data()
    if not items:
        print("No data to embed. Exiting.")
        return

    texts = [item["text"] for item in items]
    ids = [item["id"] for item in items]
    metadatas = [item.get("metadata", {}) for item in items]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    print(f"Embedding {len(texts)} texts in batches of {BATCH_SIZE} ...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"Setting up ChromaDB at {CHROMA_DB_PATH} ...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("Adding to ChromaDB collection ...")
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    print(f"ChromaDB collection '{COLLECTION_NAME}' now has {collection.count()} items.")

    print("\nDone! ChromaDB is ready.")


if __name__ == "__main__":
    main()
