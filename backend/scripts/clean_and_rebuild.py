"""
Cleans up chunks data:
1. Removes "RAG relevance" lines from the `text` field
2. Stores them in `metadata.rag_relevance` instead
3. Merges in safe_scenarios.json
4. Rebuilds the ChromaDB collection

Run from backend/ directory:
    python scripts/clean_and_rebuild.py
"""

import json
import re
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DB_PATH = str(Path(__file__).parent.parent.parent / "chroma_db")
COLLECTION_NAME = "citizen_fraud_shield"

RAG_RE_PATTERN = re.compile(
    r'\*{0,2}RAG relevance:\*{0,2}.*',
    re.DOTALL
)


def clean_chunk(chunk: dict) -> dict:
    """Strip 'RAG relevance' text from chunk['text'] and store in metadata."""
    text = chunk.get("text", "")
    rag_match = RAG_RE_PATTERN.search(text)

    clean_text = text
    rag_relevance = ""

    if rag_match:
        raw_match = rag_match.group(0)
        rag_relevance = raw_match.strip().lstrip("*").strip()
        clean_text = text[:rag_match.start()].strip().rstrip("*").strip()

    metadata = chunk.get("metadata", {})
    if rag_relevance:
        metadata["rag_relevance"] = rag_relevance

    if not clean_text:
        if rag_relevance:
            final_text = rag_relevance
            final_text = re.sub(r'^RAG relevance:\s*', '', final_text, flags=re.IGNORECASE)
            if final_text:
                final_text = final_text[0].upper() + final_text[1:]
        else:
            topic = metadata.get("topic", "")
            source = metadata.get("source", "")
            final_text = f"Context about {topic} from {source}" if topic else text
    else:
        final_text = clean_text

    return {
        "id": chunk["id"],
        "text": final_text,
        "metadata": metadata,
    }


def load_chunks(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def main():
    print("=" * 50)
    print("1. Loading and cleaning chunks.json...")
    raw_chunks = load_chunks(DATA_DIR / "chunks.json")
    cleaned = [clean_chunk(c) for c in raw_chunks]

    rag_count = sum(1 for c in cleaned if c["metadata"].get("rag_relevance"))
    print(f"   Cleaned {rag_count}/{len(cleaned)} chunks with RAG relevance text moved to metadata.")

    print("\n2. Loading safe_scenarios.json...")
    safe_path = DATA_DIR / "safe_scenarios.json"
    if safe_path.exists():
        safe_chunks = load_chunks(safe_path)
        for sc in safe_chunks:
            sc["metadata"]["category"] = "safe"
        print(f"   Loaded {len(safe_chunks)} safe scenario chunks.")
        all_chunks = cleaned + safe_chunks
    else:
        print("   No safe_scenarios.json found, skipping.")
        all_chunks = cleaned

    print(f"\n3. Total chunks to index: {len(all_chunks)}")

    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    ids = [c["id"] for c in all_chunks]

    print("\n4. Loading embedding model: paraphrase-multilingual-MiniLM-L12-v2...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("5. Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print(f"\n6. Setting up ChromaDB at {CHROMA_DB_PATH} ...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("   Deleted existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("7. Adding chunks to ChromaDB collection ...")
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    print(f"   ChromaDB collection '{COLLECTION_NAME}' has {collection.count()} items.")

    print("\n" + "=" * 50)
    print(f"Done! {len(all_chunks)} chunks indexed in ChromaDB.")
    print(f"  - Fraud advisory chunks: {len(cleaned)}")
    print(f"  - Safe scenario chunks:  {len(all_chunks) - len(cleaned)}")


if __name__ == "__main__":
    main()
