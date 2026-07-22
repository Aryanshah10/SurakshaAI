"""
Quick test: query the ChromaDB collection built by embed_and_store.py
Run after embed_and_store.py has finished.

Usage:
    python test_retrieval.py "your query here"
"""

import sys
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "citizen_fraud_shield"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How does UPI fraud usually happen?"

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K,
    )

    print(f"\nQuery: {query}\n")
    print(f"Top {TOP_K} results:\n" + "-" * 50)

    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        text = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print(f"\n[{i+1}] id={doc_id} | source={meta.get('source_file')}")
        print(f"    similarity_distance={distance:.4f}")
        print(f"    full text:\n{text}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()
