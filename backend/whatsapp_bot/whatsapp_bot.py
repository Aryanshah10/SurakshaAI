"""
WhatsApp webhook route for Twilio — used via the main FastAPI app.
 
Receives WhatsApp messages from Twilio, runs them through:
  1. Language detection (langdetect)
  2. RAG retrieval (ChromaDB / FAISS)
  3. Groq LLM for a contextual answer in the SAME language
 
Twilio sandbox setup:
  twilio.com/console -> Messaging -> WhatsApp Sandbox
  Set "When a message comes in" to: https://your-domain/api/citizen/whatsapp
  Method: POST
"""
 
import os
import re
import json
import logging
from typing import Optional
from fastapi import APIRouter, Form, Response
from utils.rag_engine import query_rag
from groq import Groq
 
logger = logging.getLogger(__name__)
 
router = APIRouter(prefix="/api/citizen", tags=["WhatsApp"])
 
# ─── Groq Client ───────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
 
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
 
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
    "Information": "\U0001F4D6",
}

ERROR_MESSAGES = {
    "English": "Sorry, a technical issue occurred. Please try again later. For urgent help, call 1930.",
    "Hindi": "क्षमा करें, कोई तकनीकी समस्या आ गई। कृपया बाद में पुनः प्रयास करें। तत्काल सहायता के लिए 1930 पर कॉल करें।",
    "Marathi": "क्षमस्व, तांत्रिक अडचण आली आहे. कृपया नंतर पुन्हा प्रयत्न करा. तातडीच्या मदतीसाठी 1930 वर कॉल करा.",
    "Bengali": "দুঃখিত, একটি প্রযুক্তিগত সমস্যা হয়েছে। অনুগ্রহ করে পরে আবার চেষ্টা করুন। জরুরি সাহায্যের জন্য 1930 কল করুন।",
    "Telugu": "క్షమించండి, సాంకేతిక సమస్య ఏర్పడింది. దయచేసి తర్వాత మళ్లీ ప్రయత్నించండి. అత్యవసర సహాయం కోసం 1930 కాల్ చేయండి.",
    "Tamil": "மன்னிக்கவும், தொழில்நுட்ப சிக்கல் ஏற்பட்டது. தயவுசெய்து பின்னர் மீண்டும் முயற்சிக்கவும். அவசர உதவிக்கு 1930 ஐ அழைக்கவும்.",
    "Gujarati": "માફ કરશો, ટેકનિકલ સમસ્યા આવી છે. કૃપા કરીને પછીથી ફરી પ્રયાસ કરો. તાત્કાલિક મદદ માટે 1930 પર કૉલ કરો.",
    "Kannada": "ಕ್ಷಮಿಸಿ, ತಾಂತ್ರಿಕ ಸಮಸ್ಯೆ ಉಂಟಾಗಿದೆ. ದಯವಿಟ್ಟು ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ. ತುರ್ತು ಸಹಾಯಕ್ಕಾಗಿ 1930 ಕಾಲ್ ಮಾಡಿ.",
    "Malayalam": "ക്ഷമിക്കണം, ഒരു സാങ്കേതിക പ്രശ്നം ഉണ്ടായി. ദയവായി പിന്നീട് വീണ്ടും ശ്രമിക്കുക. അടിയന്തര സഹായത്തിനായി 1930 എന്ന നമ്പറിൽ വിളിക്കുക.",
    "Punjabi": "ਮੁਆਫ ਕਰਨਾ, ਕੋਈ ਤਕਨੀਕੀ ਸਮੱਸਿਆ ਆ ਗਈ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਬਾਅਦ ਵਿੱਚ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ। ਤੁਰੰਤ ਮਦਦ ਲਈ 1930 'ਤੇ ਕਾਲ ਕਰੋ।",
    "Odia": "କ୍ଷମା କରନ୍ତୁ, ଏକ ଯାନ୍ତ୍ରିକ ସମସ୍ୟା ଘଟିଛି। ଦୟାକରି ପରେ ପୁଣି ଚେଷ୍ଟା କରନ୍ତୁ। ଜରୁରୀ ସହାୟତା ପାଇଁ 1930 କୁ କଲ୍ କରନ୍ତୁ।",
    "Urdu": "معذرت، ایک تکنیکی مسئلہ پیش آیا ہے۔ براہ کرم بعد میں دوبارہ کوشش کریں۔ فوری مدد کے لیے 1930 پر کال کریں۔",
}
 
 
def detect_language(text: str) -> tuple[Optional[str], bool]:
    try:
        from langdetect import detect
        code = detect(text)
        lang = LANGUAGE_MAP.get(code)
        if lang:
            return lang, False
        return None, True
    except Exception:
        return None, True
 
 
def build_twiml(answer: str) -> str:
    safe = (
        answer
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Message>{safe}</Message></Response>"
    )
 
 
def get_error_message(language_name: Optional[str]) -> str:
    if language_name is None:
        language_name = "English"
    return ERROR_MESSAGES.get(language_name, ERROR_MESSAGES["English"])


# ─── Greeting & About-Bot Detection ───────────────

_GREETING_PATTERNS = [
    "hi", "hello", "hey", "heyo", "hola", "hii", "hiii", "helloo",
    "how are you", "how r u", "hru", "howdy", "sup", "wassup", "what's up",
    "good morning", "good evening", "good afternoon", "good night",
    "namaste", "namaskar", "namaskaram", "pranam", "vanakkam",
    "nomoshkar", "sat sri akal", "jal shree",
    "kaise ho", "kya haal hai", "kya kar rahe ho", "kese ho",
    "sab theek", "kaisa hai", "namaskara",
]
_ABOUT_BOT_PATTERNS = [
    "what do you do", "what can you do", "what is your purpose",
    "who are you", "tell me about yourself", "introduce yourself",
    "what is this bot", "what is this number", "kaun ho tum",
    "aap kya karte ho", "aap kaun ho", "aapke baare mein batao",
    "yeh kya hai", "yeh number kiska hai", "bot kya karta hai",
    "aap kya hain", "yeh kaam kya hai", "what are you", "whats your name",
    "tu kaun hai", "tum kaun ho",
]


def _is_greeting(text: str) -> bool:
    """Check if the user is just greeting the bot."""
    lower = text.lower().strip().rstrip("!?.,;:")
    if lower in _GREETING_PATTERNS:
        return True
    first_word = lower.split()[0] if lower.split() else ""
    return first_word in _GREETING_PATTERNS


def _is_about_bot(text: str) -> bool:
    lower = text.lower().strip().rstrip("!?.,;:")
    return any(p in lower for p in _ABOUT_BOT_PATTERNS)


_GREETING_RESPONSE = (
    "\U0001F44B *Namaste! I'm SurakshaAI, your Citizen Fraud Shield.*\n\n"
    "I help you identify if a call, message, or payment request is a scam. "
    "Here's what I can do for you:\n"
    "\u2022 \U0001F50D Analyze suspicious messages and calls for fraud\n"
    "\u2022 \U0001F4D6 Explain different types of scams (UPI fraud, digital arrest, OTP scams, etc.)\n"
    "\u2022 \U0001F6E1\uFE0F Give prevention tips to keep you safe\n"
    "\u2022 \U0001F4DE Help you report incidents (Helpline: 1930)\n\n"
    "Just describe your situation or ask me anything about scams!"
)

_ABOUT_BOT_RESPONSE = (
    "\U0001F916 *I am SurakshaAI — Citizen Fraud Shield*\n\n"
    "I'm an AI assistant designed to help Indian citizens identify and protect themselves "
    "from financial fraud and cyber scams.\n\n"
    "\U0001F4AC *You can:*\n"
    "\u2022 Paste a suspicious message or call transcript — I'll analyze it for fraud\n"
    "\u2022 Ask about scam types — I'll explain how they work with prevention tips\n"
    "\u2022 Ask for safety advice — I'll guide you on staying safe\n\n"
    "\U0001F6A8 *Emergency Helpline: 1930* (National Cybercrime Reporting Portal)\n"
    "\U0001F4F1 *Website:* cybercrime.gov.in"
)
 
 
@router.post("/whatsapp")
async def whatsapp_webhook(
    Body: str = Form(...),
    From: str = Form(...),
):
    user_message = Body.strip()
 
    if not groq_client:
        logger.error("GROQ_API_KEY not configured")
        return Response(
            content=build_twiml("Service configuration error. Please contact support. Call 1930 for urgent help."),
            media_type="application/xml",
        )    # ── Pre-processing: Greetings & About-Bot ──
    if _is_greeting(user_message):
        return Response(
            content=build_twiml(_GREETING_RESPONSE),
            media_type="application/xml",
        )
    if _is_about_bot(user_message):
        return Response(
            content=build_twiml(_ABOUT_BOT_RESPONSE),
            media_type="application/xml",
        )

    detected_lang, failed = detect_language(user_message)

    if failed or not detected_lang:
        unclear = (
            "\U0001F937\u200D\u2642\uFE0F I'm sorry, I couldn't fully understand your message. "
            "It seems outside my area of expertise.\n\n"
            "I specialize in helping citizens identify fraud and scams related to "
            "suspicious calls, messages, and payment requests.\n\n"
            "If you've received something suspicious, please describe it to me and I'll help you. "
            "You can message me in: Hindi, English, Tamil, Telugu, Marathi, Bengali, "
            "Gujarati, Kannada, Malayalam, Punjabi, Odia, or Urdu."
        )
        return Response(content=build_twiml(unclear), media_type="application/xml")
 
    try:
        context, sources = query_rag(user_message)
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")
        context = ""
        sources = []
 
    system_prompt = f"""You are Citizen Fraud Shield, an AI assistant helping Indian citizens identify 
potential fraud in suspicious calls, messages, or payment requests.
 
You will be given:
1. A user's description of something suspicious.
2. Reference material from verified advisories (CERT-In, RBI, NCRB).Your job:
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
- Give a short, clear explanation in {detected_lang} that a non-technical citizen can understand.
  Write natively in that language's script (e.g. Devanagari for Hindi), NOT transliterated.
- If classified as "Fraud" or "Suspicious", give 2-3 concrete next steps in {detected_lang}.
- For "Need More Information", ask clarifying questions in {detected_lang} to better understand the situation.
- For "Safe" verdict, provide reassurance and general safety tips.
- Base your answer on reference material. Do not invent facts.
- Keep the entire response under 300 words — WhatsApp messages must be concise.
- If the topic involves fraud/scams, always end with: "Helpline: 1930".

Respond in valid JSON:
{{
  "verdict": "Fraud" | "Suspicious" | "Safe" | "Need More Information" | "Information",
  "confidence": "High" | "Medium" | "Low",
  "reasoning": "short explanation in {detected_lang}",
  "next_steps": ["step 1 in {detected_lang}", "step 2 in {detected_lang}"] or ["Ask specific question 1", "Ask specific question 2"] for Need More Information,
  "matched_pattern": "fraud pattern name or null"
}}"""
 
    context_block = "\n\n".join(
        f"[Source: {s}]\n{c}" for s, c in zip(sources, context.split("\n\n"))
    ) if context else "No specific advisories found."
 
    user_prompt = f"""User's situation:
"{user_message}"
 
Reference material:
{context_block}
 
Classify this situation and respond in JSON."""
 
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        verdict = json.loads(response.choices[0].message.content)
 
        emoji = EMOJI_MAP.get(verdict.get("verdict"), "")
        lines = [
            f"{emoji} *{verdict.get('verdict', 'Unknown')}* (confidence: {verdict.get('confidence', '-')})",
            "",
            verdict.get("reasoning", ""),
        ]
        if verdict.get("matched_pattern"):
            lines.append(f"\n_Pattern: {verdict['matched_pattern']}_")
        if verdict.get("next_steps"):
            lines.append("")
            for step in verdict["next_steps"]:
                lines.append(f"\u2022 {step}")
        if verdict.get("verdict") in ("Fraud", "Suspicious"):
            lines.append("\n\U0001F6A8 Helpline: 1930")
        elif verdict.get("verdict") == "Safe":
            lines.append("\n\u2705 Stay safe! Reach out if you have more questions.")
        elif verdict.get("verdict") == "Information":
            lines.append("\n\U0001F6E1\uFE0F *Prevention Tips:*")
            lines.append("\u2022 Never share OTP, PIN, or bank details with anyone")
            lines.append("\u2022 Always verify caller identity through official channels")
            lines.append("\u2022 Report suspicious activity at cybercrime.gov.in")
            lines.append("\n\U0001F6A8 Helpline: 1930")
 
        answer = "\n".join(lines)
 
    except Exception as e:
        logger.error(f"Groq LLM call failed: {e}")
        fallback = context[:400] if context else ""
        answer = (
            f"{fallback}\n\n\U0001F6A8 Madad ke liye 1930 par call karein."
            if fallback
            else get_error_message(detected_lang)
        )
 
    return Response(
        content=build_twiml(answer),
        media_type="application/xml",
    )