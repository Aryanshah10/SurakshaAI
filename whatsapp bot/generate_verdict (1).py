"""
Citizen Fraud Shield - Verdict Generation Script
Step: User query -> Retrieve chunks (ChromaDB) -> LLM verdict (Groq) -> Structured output

Setup:
1. Get a free Groq API key: https://console.groq.com/keys
2. Set it as an environment variable (recommended) OR paste directly below:
       setx GROQ_API_KEY "your-key-here"      (Windows, then restart terminal)
   OR just set GROQ_API_KEY = "your-key-here" directly in this file (quick for hackathon).
3. Install the groq package:
       pip install groq

Usage:
    python generate_verdict.py "mujhe ek call aaya bola main CBI se hu, digital arrest hoga"
"""

import sys
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------- CONFIG ----------
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "citizen_fraud_shield"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"   # multilingual (50+ languages), 384-dim
TOP_K = 4

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")   # <-- or paste your key directly here as a string
GROQ_MODEL = "llama-3.3-70b-versatile"               # free, strong reasoning model on Groq
# -----------------------------

LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "te": "Telugu",
    "ta": "Tamil",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "ur": "Urdu",
}


def detect_language(text: str):
    """
    Auto-detects language of incoming message.
    Returns (language_name, detection_failed).
    """
    try:
        from langdetect import detect
        code = detect(text)
        lang = LANGUAGE_MAP.get(code)
        if lang:
            return lang, False
        return None, True
    except Exception:
        return None, True


SYSTEM_PROMPT_TEMPLATE = """You are Citizen Fraud Shield, an AI assistant that helps Indian citizens identify \
potential fraud in suspicious calls, messages, or payment requests.

You will be given:
1. A user's description of something suspicious that happened to them.
2. Reference material retrieved from a verified knowledge base (CERT-In, RBI, NCRB advisories).

Your job:
- Classify the situation as one of: "Fraud", "Suspicious", or "Safe".
- Give a short, clear reason in {language} that a non-technical citizen can easily understand. \
If the language is not English, write the reasoning and next steps natively in that language's script, not transliterated.
- If classified as "Fraud" or "Suspicious", give 2-3 concrete next steps in {language} (e.g. do not share OTP, \
block the number, report at cybercrime.gov.in or call 1930).
- Base your answer ONLY on the reference material provided. Do not invent facts.
- Respond ONLY in valid JSON with this exact structure, no extra text:

{{
  "verdict": "Fraud" | "Suspicious" | "Safe",
  "confidence": "High" | "Medium" | "Low",
  "reasoning": "short explanation in {language}",
  "next_steps": ["step 1", "step 2"],
  "matched_pattern": "short name of the fraud pattern if applicable, else null"
}}
"""


def retrieve_chunks(query: str, embed_model, collection, top_k=TOP_K):
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source_file", "unknown"),
            "distance": results["distances"][0][i],
        })
    return chunks


def build_user_prompt(user_query: str, chunks: list):
    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"""User's situation:
"{user_query}"

Reference material:
{context_block}

Based only on the reference material above, classify this situation and respond in the required JSON format."""


def main():
    if len(sys.argv) < 2:
        print('Usage: python generate_verdict.py "your suspicious message/call description here"')
        return

    user_query = sys.argv[1]

    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys")
        print("Then set it as an environment variable, or paste it into GROQ_API_KEY in this script.")
        return

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    print(f"Retrieving top {TOP_K} relevant chunks...")
    chunks = retrieve_chunks(user_query, embed_model, collection)

    language, detection_failed = detect_language(user_query)
    if language is None:
        language = "English"
    print(f"Detected language: {language}")

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(language=language)

    print("Generating verdict...")
    groq_client = Groq(api_key=GROQ_API_KEY)

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(user_query, chunks)},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw_output = response.choices[0].message.content

    try:
        verdict = json.loads(raw_output)
    except json.JSONDecodeError:
        print("Could not parse LLM output as JSON. Raw output:")
        print(raw_output)
        return

    print("\n" + "=" * 50)
    print(f"VERDICT: {verdict.get('verdict')}  (confidence: {verdict.get('confidence')})")
    print("=" * 50)
    print(f"\nReasoning: {verdict.get('reasoning')}")
    if verdict.get("matched_pattern"):
        print(f"Matched pattern: {verdict.get('matched_pattern')}")
    if verdict.get("next_steps"):
        print("\nNext steps:")
        for step in verdict["next_steps"]:
            print(f"  - {step}")

    print("\n(Sources used for this verdict:)")
    for c in chunks:
        print(f"  - {c['source']} (distance: {c['distance']:.3f})")


if __name__ == "__main__":
    main()
