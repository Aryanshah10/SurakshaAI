"""
Citizen Fraud Shield - Embedding + Vector Store Script
Step: Chunks (JSON) -> Embeddings -> ChromaDB

Run this in your tf-gpu WSL2 environment (or a fresh venv).
Install deps first:
    pip install sentence-transformers chromadb

Usage:
    python embed_and_store.py
"""

import json
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ---------- CONFIG ----------
CHUNKS_JSON_PATH = "./backend/data/chunks.json"   # <-- update to your actual chunks file path
CHROMA_DB_PATH = "./chroma_db"            # local persistent storage folder
COLLECTION_NAME = "citizen_fraud_shield"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"   # multilingual (50+ languages), 384-dim
BATCH_SIZE = 32
# -----------------------------


def load_chunks(path: str):
    """Load chunk JSON. Expects a list of dicts with at least a 'content' field."""
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError("Expected chunks.json to contain a list of chunk objects.")

    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def normalize_chunk(chunk: dict, idx: int):
    """
    Handles slightly different key names so this works even if your
    chunking script used different field names (heading/title, content/text).
    """
    text = chunk.get("content") or chunk.get("text") or chunk.get("body")
    if text is None:
        raise ValueError(f"Chunk at index {idx} has no 'content'/'text'/'body' field: {chunk}")

    heading = chunk.get("heading") or chunk.get("title") or ""
    source = chunk.get("source_file") or chunk.get("source") or "unknown"

    chunk_id = chunk.get("chunk_id") or chunk.get("id") or f"chunk_{idx}"

    return {
        "id": str(chunk_id),
        "text": text,
        "metadata": {
            "heading": heading,
            "source_file": source,
        },
    }


def main():
    # 1. Load chunks
    if not Path(CHUNKS_JSON_PATH).exists():
        print(f"ERROR: {CHUNKS_JSON_PATH} not found. Update CHUNKS_JSON_PATH at top of script.")
        return

    raw_chunks = load_chunks(CHUNKS_JSON_PATH)
    chunks = [normalize_chunk(c, i) for i, c in enumerate(raw_chunks)]

    # 2. Load embedding model (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")

    print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # 3. Set up ChromaDB (persistent, local)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
    )

    # 4. Embed + add in batches
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print(f"Embedding {len(texts)} chunks in batches of {BATCH_SIZE} ...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print("Adding to ChromaDB collection ...")
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Done. Collection '{COLLECTION_NAME}' now has {collection.count()} items.")
    print(f"Stored persistently at: {CHROMA_DB_PATH}")


if __name__ == "__main__":
    main()
