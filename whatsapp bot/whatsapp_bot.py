"""
Citizen Fraud Shield - Twilio WhatsApp Bot
Receives WhatsApp messages via Twilio webhook, runs them through the
RAG + verdict pipeline, and replies with the fraud verdict.

Setup:
1. pip install flask twilio groq chromadb sentence-transformers
2. Set GROQ_API_KEY below (or as env var)
3. Run: python whatsapp_bot.py
4. Expose it to the internet (Twilio needs a public URL to send webhooks to):
       - Easiest: install ngrok (https://ngrok.com/download), then run: ngrok http 5000
       - Copy the https://xxxx.ngrok-free.app URL it gives you
5. In Twilio Console -> Messaging -> Try it out -> WhatsApp Sandbox Settings:
       - Set "When a message comes in" to: https://xxxx.ngrok-free.app/whatsapp
       - Method: POST
6. Message your Twilio WhatsApp sandbox number from your phone. Replies will
   come back through this server.
"""

import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.twiml.voice_response import VoiceResponse, Gather

# ---------- CONFIG ----------
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "citizen_fraud_shield"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"   # multilingual (50+ languages), 384-dim
TOP_K = 4

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")   # <-- or paste your key directly here
GROQ_MODEL = "llama-3.3-70b-versatile"
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
    Auto-detects language of incoming WhatsApp message.
    Returns (language_name, detection_failed).
    language_name is None if detection failed or language is unsupported.
    """
    try:
        from langdetect import detect
        code = detect(text)
        lang = LANGUAGE_MAP.get(code)
        if lang:
            return lang, False
        return None, True      # detected but unsupported language
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
If the language is not English, write the reasoning and next steps natively in that language's script \
(e.g. Devanagari for Hindi/Marathi, Bengali script for Bengali, etc.), not transliterated.
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

VERDICT_EMOJI = {"Fraud": "\U0001F534", "Suspicious": "\U0001F7E1", "Safe": "\U0001F7E2"}

# Error message shown when something goes wrong, in each supported language.
# Falls back to English if the user's language isn't in this map or detection fails.
ERROR_MESSAGES = {
    "English": "Sorry, a technical issue occurred. Please try again.",
    "Hindi": "क्षमा करें, कुछ तकनीकी समस्या आ गई। कृपया दोबारा प्रयास करें।",
    "Marathi": "क्षमस्व, काहीतरी तांत्रिक समस्या आली. कृपया पुन्हा प्रयत्न करा.",
    "Bengali": "দুঃখিত, একটি প্রযুক্তিগত সমস্যা হয়েছে। অনুগ্রহ করে আবার চেষ্টা করুন।",
    "Telugu": "క్షమించండి, సాంకేతిక సమస్య వచ్చింది. దయచేసి మళ్లీ ప్రయత్నించండి.",
    "Tamil": "மன்னிக்கவும், தொழில்நுட்ப சிக்கல் ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்.",
    "Gujarati": "માફ કરશો, કંઈક ટેકનિકલ સમસ્યા આવી. કૃપા કરી ફરીથી પ્રયાસ કરો.",
    "Kannada": "ಕ್ಷಮಿಸಿ, ತಾಂತ್ರಿಕ ಸಮಸ್ಯೆ ಉಂಟಾಗಿದೆ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
    "Malayalam": "ക്ഷമിക്കണം, ഒരു സാങ്കേതിക പ്രശ്നം ഉണ്ടായി. ദയവായി വീണ്ടും ശ്രമിക്കുക.",
    "Punjabi": "ਮਾਫ਼ ਕਰਨਾ, ਕੋਈ ਤਕਨੀਕੀ ਸਮੱਸਿਆ ਆ ਗਈ। ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
    "Odia": "କ୍ଷମା କରନ୍ତୁ, କିଛି ଯାନ୍ତ୍ରିକ ସମସ୍ୟା ହୋଇଛି। ଦୟାକରି ପୁଣି ଚେଷ୍ଟା କରନ୍ତୁ।",
    "Urdu": "معذرت، کچھ تکنیکی مسئلہ پیش آگیا۔ براہ کرم دوبارہ کوشش کریں۔",
}


def get_error_message(user_query: str) -> str:
    """Picks the technical-error reply in the same language as the user's message.
    If detection fails or the language isn't supported, falls back to English."""
    language, _ = detect_language(user_query)
    if language is None:
        language = "English"
    return ERROR_MESSAGES.get(language, ERROR_MESSAGES["English"])

app = Flask(__name__)

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def retrieve_chunks(query: str, top_k=TOP_K):
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source_file", "unknown"),
        })
    return chunks


def build_user_prompt(user_query: str, chunks: list):
    context_block = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
    return f"""User's situation:
"{user_query}"

Reference material:
{context_block}

Based only on the reference material above, classify this situation and respond in the required JSON format."""


def generate_verdict(user_query: str):
    chunks = retrieve_chunks(user_query)

    language, detection_failed = detect_language(user_query)
    if language is None:
        # Detection failed or unsupported language -> fall back to English
        language = "English"

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(language=language)

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(user_query, chunks)},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def format_whatsapp_reply(verdict: dict) -> str:
    emoji = VERDICT_EMOJI.get(verdict.get("verdict"), "")
    lines = [
        f"{emoji} *{verdict.get('verdict', 'Unknown')}* (confidence: {verdict.get('confidence', '-')})",
        "",
        verdict.get("reasoning", ""),
    ]
    if verdict.get("matched_pattern"):
        lines.append(f"\n_Pattern: {verdict['matched_pattern']}_")
    if verdict.get("next_steps"):
        lines.append("\n*Next steps:*")
        for step in verdict["next_steps"]:
            lines.append(f"- {step}")
    return "\n".join(lines)


@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.values.get("Body", "").strip()
    sender = request.values.get("From", "unknown")
    print(f"Incoming from {sender}: {incoming_msg}")

    resp = MessagingResponse()
    msg = resp.message()

    if not GROQ_API_KEY:
        msg.body("Server not configured: GROQ_API_KEY missing.")
        return str(resp)

    if not incoming_msg:
        msg.body("Please describe your Suspicious call/message.")
        return str(resp)

    try:
        verdict = generate_verdict(incoming_msg)
        msg.body(format_whatsapp_reply(verdict))
    except Exception as e:
        print(f"Error: {e}")
        msg.body(get_error_message(incoming_msg))

    return str(resp)


@app.route("/", methods=["GET"])
def health_check():
    return "Citizen Fraud Shield WhatsApp + IVR bot is running."




if __name__ == "__main__":
    app.run(port=5000, debug=True)