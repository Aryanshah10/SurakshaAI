"""
Citizen Fraud Shield - Verdict Generation Script (Upgraded)
Step: User query -> FAISS retrieval (with threshold + reranker) -> LLM verdict (Groq) -> Structured output

Features:
  - 5 verdicts: Fraud, Suspicious, Safe, Need More Information, Information
  - Cosine similarity threshold filtering
  - Cross-encoder reranker
  - Safe scenarios knowledge base
  - RAG relevance stored only as metadata

Setup:
  1. Get a free Groq API key: https://console.groq.com/keys
  2. Set GROQ_API_KEY as environment variable

Usage:
    python "generate_verdict (1).py" "mujhe ek call aaya bola main CBI se hu, digital arrest hoga"
"""

import sys
import os
import json
from pathlib import Path

# Ensure backend/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from utils.rag_engine import query_rag

# ---------- CONFIG ----------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
# -----------------------------

LANGUAGE_MAP = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "bn": "Bengali",
    "te": "Telugu", "ta": "Tamil", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "pa": "Punjabi", "or": "Odia", "ur": "Urdu",
}

EMOJI_MAP = {
    "Fraud": "\U0001F534",
    "Suspicious": "\U0001F7E1",
    "Safe": "\U0001F7E2",
    "Need More Information": "\U0001F535",
}


def detect_language(text: str):
    try:
        from langdetect import detect
        code = detect(text)
        lang = LANGUAGE_MAP.get(code)
        if lang:
            return lang, False
        return None, True
    except Exception:
        return None, True


def main():
    if len(sys.argv) < 2:
        print('Usage: python "generate_verdict (1).py" "your suspicious message/call description here"')
        return

    user_query = sys.argv[1]

    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set in .env or environment variables.")
        return

    # RAG retrieval with threshold + reranker
    print("Retrieving relevant advisories (FAISS + reranker + threshold)...")
    try:
        context, sources, distances = query_rag(user_query)
    except Exception as e:
        print(f"RAG retrieval failed: {e}")
        context = ""
        sources = []
        distances = []

    has_relevant = bool(context and len(distances) > 0)

    if has_relevant:
        print(f"Found {len(distances)} relevant chunks (similarity threshold >= 0.65):")
        for d in distances:
            print(f"  - {d['source']} (cosine: {d['cosine_similarity']})")
    else:
        print("No chunks passed the similarity threshold. The query is not closely related to known fraud patterns.")

    language, detection_failed = detect_language(user_query)
    if language is None:
        language = "English"
    print(f"Detected language: {language}\n")

    system_prompt = f"""You are Citizen Fraud Shield, an AI assistant helping Indian citizens identify
potential fraud in suspicious calls, messages, or payment requests.

You will be given:
1. A user's description of something suspicious.
2. Reference material from verified advisories (CERT-In, RBI, NCRB) AND safe scenarios.

Your job:
- Classify the situation as one of: "Fraud", "Suspicious", "Safe", OR "Need More Information", OR "Information".
- Use "Need More Information" when the user's description is too vague, unrelated to the reference material,
  or the reference material does not clearly match the situation. This is a safe fallback to avoid false alarms.
- **Use "Information" when the user is asking for educational information or explanation** about a scam/fraud type.
  Examples: "what is UPI scam", "tell me about digital arrest", "explain OTP fraud", "how does phishing work",
  "UPI scam kya hai", "digital arrest ke baare mein batao". These are NOT incident reports — they are learning requests.
- For "Information" verdict: provide a clear explanation of what the scam is, how it works, red flags to watch for,
  and **prevention tips** (how to stay safe). Include all this in the "reasoning" field.
- If the user greets you (e.g. "hi", "hello", "namaste", "kaise ho"), or asks about you ("what do you do", "who are you"),
  respond with a "Safe" verdict and give a friendly introduction as reasoning.
- If the user's message is completely unrelated to fraud, scams, or suspicious activity (e.g. random words, jokes,
  gibberish, or topics outside cybersecurity), respond with "Safe" and politely say this is outside your area of
  expertise — you specialize in fraud and scam identification only.
- Give a short, clear explanation in {language} that a non-technical citizen can understand.
  Write natively in that language's script (e.g. Devanagari for Hindi), NOT transliterated.
- If classified as "Fraud" or "Suspicious", give 2-3 concrete next steps in {language}.
- For "Need More Information", ask clarifying questions in {language} to better understand the situation.
- For "Safe" verdict, provide reassurance and general safety tips.
- Base your answer on reference material. Do not invent facts.
- Keep the entire response under 300 words.
- If the topic involves fraud/scams, always end with: "Helpline: 1930".

Respond in valid JSON:
{{
  "verdict": "Fraud" | "Suspicious" | "Safe" | "Need More Information" | "Information",
  "confidence": "High" | "Medium" | "Low",
  "reasoning": "short explanation in {language}",
  "next_steps": ["step 1 in {language}", "step 2 in {language}"] or ["Ask specific question 1", "Ask specific question 2"] for Need More Information,
  "matched_pattern": "fraud pattern name or null"
}}"""

    if has_relevant:
        context_block = "\n\n".join(
            f"[Source: {s}]\n{c}" for s, c in zip(sources, context.split("\n\n"))
        )
    else:
        context_block = "No specific advisories found for this situation."

    user_prompt = f"""User's situation:
"{user_query}"

Reference material:
{context_block}

Classify this situation and respond in JSON."""

    print("Generating verdict via Groq...")
    groq_client = Groq(api_key=GROQ_API_KEY)

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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

    verdict_type = verdict.get("verdict", "Unknown")
    emoji = EMOJI_MAP.get(verdict_type, "")

    print("\n" + "=" * 50)
    print(f"{emoji} VERDICT: {verdict_type}  (confidence: {verdict.get('confidence')})")
    print("=" * 50)
    print(f"\nReasoning: {verdict.get('reasoning')}")
    if verdict.get("matched_pattern"):
        print(f"Matched pattern: {verdict.get('matched_pattern')}")
    if verdict.get("next_steps"):
        print("\nNext steps:")
        for step in verdict["next_steps"]:
            print(f"  - {step}")

    if distances:
        print("\n(Retrieval distances for this verdict:)")
        for d in distances:
            print(f"  - {d['source']} (cosine sim: {d['cosine_similarity']}, l2: {d['l2_distance']})")


if __name__ == "__main__":
    main()
