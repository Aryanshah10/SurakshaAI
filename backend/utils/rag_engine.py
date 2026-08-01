import os
import logging

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

logger = logging.getLogger(__name__)

TOP_K = 10
RERANK_K = 5
SIMILARITY_THRESHOLD = 0.35

_model = None
_collection = None
_reranker = None

_CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")
_COLLECTION_NAME = "citizen_fraud_shield"


def _load():
    global _model, _collection, _reranker
    if _collection is None:
        logger.info("Loading embedding model & ChromaDB collection...")
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        client = chromadb.PersistentClient(path=os.path.abspath(_CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{_COLLECTION_NAME}' loaded with "
            f"{_collection.count()} items."
        )
    if _reranker is None:
        try:
            logger.info("Loading cross-encoder reranker model...")
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Reranker loaded.")
        except Exception as e:
            logger.warning(f"Could not load reranker (will skip reranking): {e}")
            _reranker = None


def query_rag(question: str, top_k: int = TOP_K, rerank_k: int = RERANK_K):
    """
    Retrieve relevant chunks using ChromaDB + optional cross-encoder reranking.

    Returns (context_str, source_refs, distances_info)
      - context_str: joined text of filtered chunks
      - source_refs: list of source strings
      - distances_info: list of dicts with chunk_id, distance, similarity
    """
    _load()

    query_embedding = _model.encode([question]).tolist()

    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=min(max(top_k * 2, rerank_k * 2), _collection.count()),
    )

    candidates = []
    for i in range(len(results["ids"][0])):
        cosine_distance = results["distances"][0][i]
        cosine_similarity = round(1.0 - cosine_distance, 4)

        if cosine_similarity < SIMILARITY_THRESHOLD:
            continue

        metadata = results["metadatas"][0][i] if results["metadatas"] else {}
        candidates.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": metadata,
            "cosine_similarity": cosine_similarity,
            "cosine_distance": round(cosine_distance, 4),
        })

    if not candidates:
        return "", [], []

    if _reranker is not None and len(candidates) > 1:
        try:
            pairs = [(question, c["text"]) for c in candidates]
            rerank_scores = _reranker.predict(pairs)

            for j, score in enumerate(rerank_scores):
                candidates[j]["rerank_score"] = round(float(score), 4)
            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            candidates = candidates[:rerank_k]
        except Exception as e:
            logger.warning(f"Reranking failed, falling back to cosine: {e}")
            candidates.sort(key=lambda x: x["cosine_similarity"], reverse=True)
            candidates = candidates[:top_k]
    else:
        candidates.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        candidates = candidates[:top_k]

    context = "\n\n".join(c["text"] for c in candidates)
    sources = [
        f"{c['metadata'].get('source', 'Unknown')} — {c['metadata'].get('url', '')}"
        for c in candidates
    ]
    distances_info = [
        {
            "chunk_id": c["chunk_id"],
            "cosine_similarity": c["cosine_similarity"],
            "cosine_distance": c["cosine_distance"],
            "source": c["metadata"].get("source", "Unknown"),
        }
        for c in candidates
    ]

    return context, sources, distances_info