#routes/whatsapp.py

from fastapi import APIRouter, Form, Response
from utils.rag_engine import query_rag
from openai import OpenAI
import os

router = APIRouter(prefix="/api/citizen", tags=["WhatsApp"])

client = OpenAI(
    base_url=os.getenv("OMNIROUTE_URL", "http://localhost:20128/v1"),
    api_key=os.getenv("OMNIROUTE_API_KEY", ""),
)

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


def detect_language(text: str) -> str:
    """
    Auto-detects language of incoming WhatsApp message.
    Returns full language name for LLM prompt.
    Returns (language_name, detection_failed)
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


def build_twiml(answer: str) -> str:
    """Wraps answer in Twilio TwiML XML. Escapes special chars."""
    answer = (
        answer
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Message>{answer}</Message></Response>"
    )


@router.post("/whatsapp")
async def whatsapp_webhook(
    Body: str = Form(...),
    From: str = Form(...),
):
    """
    Twilio WhatsApp webhook endpoint.

    Flow:
      1. Citizen sends WhatsApp message in ANY of 12 languages
      2. langdetect identifies the language automatically
      3. ChromaDB retrieves relevant advisory chunks
      4. OmniRoute LLM answers in the SAME language as the citizen

    Supported languages:
      English, Hindi, Marathi, Bengali, Telugu, Tamil,
      Gujarati, Kannada, Malayalam, Punjabi, Odia, Urdu

    Twilio sandbox setup:
      twilio.com/console -> Messaging -> WhatsApp Sandbox
      Set webhook to: https://your-domain/api/citizen/whatsapp
    """
    user_message  = Body.strip()
    detected_lang, failed = detect_language(user_message)

    # If detection failed — ask citizen to specify
    if failed or not detected_lang:
        clarify = (
            "We could not detect your language. Please reply with your preferred language:\n"
            "Hindi / Tamil / Telugu / Marathi / Bengali / "
            "Gujarati / Kannada / Malayalam / Punjabi / Odia / Urdu / English"
        )
        return Response(content=build_twiml(clarify), media_type="application/xml")

    context, sources = query_rag(user_message)

    prompt = f"""You are a citizen safety assistant for an Indian government
fraud-detection platform. A citizen has messaged you on WhatsApp.

CRITICAL INSTRUCTION: Respond ONLY in {detected_lang}.
Do NOT switch languages mid-response.
Do NOT respond in English if the message is in {detected_lang}.
Keep response under 300 words — WhatsApp must be concise.
Use simple words suitable for elderly citizens as well.
If the topic involves fraud or scams, always end with: helpline number 1930.

Relevant government advisories:
{context}

Citizen message: {user_message}

Response in {detected_lang}:"""

    try:
        response = client.chat.completions.create(
            model=os.getenv("OMNIROUTE_MODEL", "google/gemini-2.5-flash"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        answer = response.choices[0].message.content.strip()

    except Exception:
        answer = (
            f"{context[:400]}...\n\nMadad ke liye call karein: 1930"
            if context
            else "Seva uplabdh nahi hai. Turant 1930 par call karein."
        )

    return Response(
        content=build_twiml(answer),
        media_type="application/xml",
    )