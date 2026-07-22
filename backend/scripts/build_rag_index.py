
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
     
OUTPUT_DIR  = "../data"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load JSON
with open("../data/chunks.json") as f:
    data = json.load(f)

# Extract text and metadata separately
texts    = [item["text"] for item in data]
metadata = [item["metadata"] for item in data]
ids      = [item["id"] for item in data]

# Embed
model      = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(texts, show_progress_bar=True)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype("float32"))

# Save index + full chunk map (text + metadata together)
faiss.write_index(index, f"{OUTPUT_DIR}/index.faiss")

chunk_map = [{"id": ids[i], "text": texts[i], "metadata": metadata[i]} 
             for i in range(len(texts))]

with open(f"{OUTPUT_DIR}/chunks_map.json", "w", encoding="utf-8") as f:
    json.dump(chunk_map, f, ensure_ascii=False, indent=2)

print(f"Done. {len(texts)} chunks indexed.")