import json
import os

# Disable Xet storage — the hf_xet DLL is blocked on this machine.
# Forces HuggingFace hub to use standard HTTP download / local cache.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

TOP_K = 3

_model  = None
_index  = None
_chunks = None

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def _load():
    global _model, _index, _chunks
    if _model is None:
        _model  = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        _index  = faiss.read_index(os.path.join(_DATA_DIR, "index.faiss"))
        with open(os.path.join(_DATA_DIR, "chunks_map.json")) as f:
            _chunks = json.load(f)

def query_rag(question: str, top_k: int = TOP_K):
    """
    Returns (retrieved_text, source_references)
    Called by routes/citizen.py to feed context into OmniRoute LLM.
    """
    _load()
    embedding = _model.encode([question]).astype("float32")
    _, indices = _index.search(embedding, top_k)

    results = [_chunks[i] for i in indices[0] if i < len(_chunks)]
    
    context = "\n\n".join([r["text"] for r in results])
    sources = [
        f"{r['metadata'].get('source', 'Unknown')} — {r['metadata'].get('url', '')}"
        for r in results
    ]
    return context, sources