import json
import os
import logging

# Disable Xet storage — the hf_xet DLL is blocked on this machine.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

logger = logging.getLogger(__name__)

TOP_K = 10             # Number to retrieve before reranking
RERANK_K = 5           # Number to keep after reranking
SIMILARITY_THRESHOLD = 0.35  # Cosine similarity threshold (minimum to consider relevant)
# Note: 0.35 is chosen for multilingual RAG where Hindi/Tamil/etc. queries
# are matched against English chunks. Typical cross-lingual relevant pairs
# have cosine similarity between 0.35 and 0.70. The reranker then boosts the best ones.

_model     = None
_index     = None
_chunks    = None
_reranker  = None

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _load():
    global _model, _index, _chunks, _reranker
    if _model is None:
        logger.info("Loading embedding model & FAISS index...")
        _model    = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        _index    = faiss.read_index(os.path.join(_DATA_DIR, "index.faiss"))
        with open(os.path.join(_DATA_DIR, "chunks_map.json"), "r", encoding="utf-8") as f:
            _chunks = json.load(f)
        logger.info(f"Loaded {len(_chunks)} chunks. FAISS index dimension: {_index.d}")
    if _reranker is None:
        try:
            logger.info("Downloading/loading cross-encoder reranker model (~420 MB on first run)...")
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Reranker loaded.")
        except Exception as e:
            logger.warning(f"Could not load reranker (will skip reranking): {e}")
            _reranker = None


def _cosine_similarity_from_l2(l2_dist: float) -> float:
    """
    Convert L2 distance to cosine similarity.
    For normalized embeddings, L2^2 = 2 - 2*cos_sim, so cos_sim = 1 - L2^2/2.
    """
    return max(0.0, min(1.0, 1.0 - (l2_dist ** 2) / 2.0))


def query_rag(question: str, top_k: int = TOP_K, rerank_k: int = RERANK_K):
    """
    Retrieve relevant chunks using FAISS + optional cross-encoder reranking.

    Returns (context_str, source_refs, distances_info)
      - context_str: joined text of filtered chunks
      - source_refs: list of source strings
      - distances_info: list of dicts with chunk_id, distance, similarity
    """
    _load()

    # 1. Embed query and normalize (index stores unit-length vectors)
    embedding = _model.encode([question]).astype("float32")
    embedding /= max(np.linalg.norm(embedding), 1e-12)
    retrieve_k = max(top_k * 2, rerank_k * 2)
    distances, indices = _index.search(embedding, min(retrieve_k, len(_chunks)))

    # 2. Collect candidates with their L2 distances
    candidates = []
    for i, idx in enumerate(indices[0]):
        if idx >= len(_chunks):
            continue
        l2_dist = distances[0][i]
        sim = _cosine_similarity_from_l2(l2_dist)
        # Skip low-similarity chunks
        if sim < SIMILARITY_THRESHOLD:
            continue
        candidates.append({
            "chunk": _chunks[idx],
            "l2_distance": round(l2_dist, 4),
            "cosine_similarity": round(sim, 4),
        })

    if not candidates:
        return "", [], []

    # 3. Optionally rerank with cross-encoder
    if _reranker is not None and len(candidates) > 1:
        try:
            pairs = [(question, c["chunk"]["text"]) for c in candidates]
            rerank_scores = _reranker.predict(pairs)

            # Sort candidates by reranker score descending
            for j, score in enumerate(rerank_scores):
                candidates[j]["rerank_score"] = round(float(score), 4)
            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

            # Keep top reranked
            candidates = candidates[:rerank_k]
        except Exception as e:
            logger.warning(f"Reranking failed, falling back to cosine: {e}")
            # Fall back to top-k by cosine similarity
            candidates.sort(key=lambda x: x["cosine_similarity"], reverse=True)
            candidates = candidates[:top_k]
    else:
        # No reranker, just take top-k by cosine similarity
        candidates.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        candidates = candidates[:top_k]

    # 4. Build output
    context = "\n\n".join([c["chunk"]["text"] for c in candidates])
    sources = [
        f"{c['chunk']['metadata'].get('source', 'Unknown')} — {c['chunk']['metadata'].get('url', '')}"
        for c in candidates
    ]
    distances_info = [
        {
            "chunk_id": c["chunk"]["id"],
            "cosine_similarity": c["cosine_similarity"],
            "l2_distance": c["l2_distance"],
            "source": c["chunk"]["metadata"].get("source", "Unknown"),
        }
        for c in candidates
    ]

    return context, sources, distances_info