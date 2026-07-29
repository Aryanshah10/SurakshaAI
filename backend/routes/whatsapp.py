"""
WhatsApp webhook route for Twilio — used via the main FastAPI app.

Receives WhatsApp messages from Twilio, runs them through:
  1. Language detection (langdetect)
  2. RAG retrieval (FAISS + cross-encoder reranker + similarity threshold)
  3. Groq LLM for a contextual answer in the SAME language

Supports 4 verdicts: "Fraud", "Suspicious", "Safe", "Need More Information"

Twilio sandbox setup:
  twilio.com/console -> Messaging -> WhatsApp Sandbox
  Set "When a message comes in" to: https://your-domain/api/citizen/whatsapp
  Method: POST
"""

import os
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
}

ERROR_MESSAGES = {
    "English": "Sorry, a technical issue occurred. Please try again later. For urgent help, call 1930.",
    "Hindi": "\u0915\u094d\u0937\u092e\u093e \u0915\u0930\u0947\u0902, \u0915\u094b\u0908 \u0924\u0915\u0928\u0940\u0915\u0940 \u0938\u092e\u0938\u094d\u092f\u093e \u0906 \u0917\u0908\u0964 \u0915\u0943\u092a\u092f\u093e \u092c\u093e\u0926 \u092e\u0947\u0902 \u092a\u0941\u0928\u0903 \u092a\u094d\u0930\u092f\u093e\u0938 \u0915\u0930\u0947\u0902\u0964 \u0924\u0924\u094d\u0915\u093e\u0932 \u0938\u0939\u093e\u092f\u0924\u093e \u0915\u0947 \u0932\u093f\u090f 1930 \u092a\u0930 \u0915\u0949\u0932 \u0915\u0930\u0947\u0902\u0964",
    "Marathi": "\u0915\u094d\u0937\u092e\u0938\u094d\u0935, \u0924\u093e\u0902\u0924\u094d\u0930\u093f\u0915 \u0905\u0921\u091a\u0923 \u0906\u0932\u0940 \u0906\u0939\u0947. \u0915\u0943\u092a\u092f\u093e \u0928\u0902\u0924\u0930 \u092a\u0941\u0928\u094d\u0939\u093e \u092a\u094d\u0930\u092f\u0924\u094d\u0928 \u0915\u0930\u093e. \u0924\u093e\u0924\u0921\u0940\u091a\u094d\u092f\u093e \u092e\u0926\u0924\u0940\u0938\u093e\u0920\u0940 1930 \u0935\u0930 \u0915\u0949\u0932 \u0915\u0930\u093e.",
    "Bengali": "\u09a6\u09c1\u0983\u0996\u09bf\u09a4, \u098f\u0995\u099f\u09bf \u09aa\u09cd\u09b0\u09af\u09c1\u0995\u09cd\u09a4\u09bf\u0997\u09a4 \u09b8\u09ae\u09b8\u09cd\u09af\u09be \u09b9\u09df\u09c7\u099b\u09c7\u0964 \u0985\u09a8\u09c1\u0997\u09cd\u09b0\u09b9 \u0995\u09b0\u09c7 \u09aa\u09b0\u09c7 \u0986\u09ac\u09be\u09b0 \u099a\u09c7\u09b7\u09cd\u099f\u09be \u0995\u09b0\u09c1\u09a8\u0964 \u099c\u09b0\u09c1\u09b0\u09bf \u09b8\u09b9\u09be\u09af\u09bc\u09a4\u09be\u09b0 \u099c\u09a8\u09cd\u09af 1930 \u0995\u09b2 \u0995\u09b0\u09c1\u09a8\u0964",
    "Telugu": "\u0c15\u0c4d\u0c37\u0c2e\u0c3f\u0c02\u0c1a\u0c02\u0c21\u0c3f, \u0c38\u0c3e\u0c02\u0c15\u0c47\u0c24\u0c3f\u0c15 \u0c38\u0c2e\u0c38\u0c4d\u0c2f \u0c0f\u0c30\u0c4d\u0c2a\u0c21\u0c3f\u0c02\u0c26\u0c3f. \u0c26\u0c2f\u0c1a\u0c47\u0c38\u0c3f \u0c24\u0c30\u0c4d\u0c35\u0c3e\u0c24 \u0c2e\u0c33\u0c4d\u0c32\u0c40 \u0c2a\u0c4d\u0c30\u0c2f\u0c24\u0c4d\u0c28\u0c3f\u0c02\u0c1a\u0c02\u0c21\u0c3f. \u0c05\u0c24\u0c4d\u0c2f\u0c35\u0c38\u0c30 \u0c38\u0c39\u0c3e\u0c2f\u0c02 \u0c15\u0c4b\u0c38\u0c02 1930 \u0c15\u0c3e\u0c32\u0c4d \u0c1a\u0c47\u0c2f\u0c02\u0c21\u0c3f.",
    "Tamil": "\u0bae\u0ba9\u0bcd\u0ba9\u0bbf\u0b95\u0bcd\u0b95\u0bb5\u0bc1\u0bae\u0bcd, \u0ba4\u0bca\u0bb4\u0bbf\u0bb2\u0bcd\u0ba8\u0bc1\u0b9f\u0bcd\u0baa \u0b9a\u0bbf\u0b95\u0bcd\u0b95\u0bb2\u0bcd \u0b8f\u0bb1\u0bcd\u0baa\u0b9f\u0bcd\u0b9f\u0bc1\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1. \u0ba4\u0baf\u0bb5\u0bc1\u0b9a\u0bc6\u0baf\u0bcd\u0ba4\u0bc1 \u0baa\u0bbf\u0ba9\u0bcd\u0ba9\u0bb0\u0bcd \u0bae\u0bc0\u0ba3\u0bcd\u0b9f\u0bc1\u0bae\u0bcd \u0bae\u0bc1\u0baf\u0bb1\u0bcd\u0b9a\u0bbf\u0b95\u0bcd\u0b95\u0bb5\u0bc1\u0bae\u0bcd. \u0b85\u0bb5\u0b9a\u0bb0 \u0b89\u0ba4\u0bb5\u0bbf\u0b95\u0bcd\u0b95\u0bc1 1930 \u0b90 \u0b85\u0bb4\u0bc8\u0b95\u0bcd\u0b95\u0bb5\u0bc1\u0bae\u0bcd.",
    "Gujarati": "\u0aae\u0abe\u0aab \u0a95\u0ab0\u0ab6\u0acb, \u0a9f\u0ac7\u0a95\u0aa8\u0abf\u0a95\u0ab2 \u0ab8\u0aae\u0ab8\u0acd\u0aaf\u0abe \u0a86\u0ab5\u0ac0 \u0a9b\u0ac7. \u0a95\u0ac3\u0aaa\u0abe \u0a95\u0ab0\u0ac0\u0aa8\u0ac7 \u0aaa\u0a9b\u0ac0\u0aa5\u0ac0 \u0aab\u0ab0\u0ac0 \u0aaa\u0acd\u0ab0\u0aaf\u0aa4\u0acd\u0aa8 \u0a95\u0ab0\u0acb. \u0aa4\u0abe\u0aa4\u0acd\u0a95\u0abe\u0ab2\u0abf\u0a95 \u0aae\u0aa6\u0aa6 \u0aae\u0abe\u0a9f\u0ac7 1930 \u0aaa\u0ab0 \u0a95\u0ac9\u0ab2 \u0a95\u0ab0\u0acb.",
    "Kannada": "\u0c95\u0ccd\u0cb7\u0cae\u0cbf\u0cb8\u0cbf, \u0ca4\u0cbe\u0c82\u0ca4\u0ccd\u0cb0\u0cbf\u0c95 \u0cb8\u0cae\u0cb8\u0ccd\u0caf\u0cc6 \u0c89\u0c82\u0c9f\u0cbe\u0c97\u0cbf\u0ca6\u0cc6. \u0ca6\u0caf\u0cb5\u0cbf\u0cb0\u0cc1 \u0ca8\u0c82\u0ca4\u0cb0 \u0cae\u0ca4\u0ccd\u0ca4\u0cc6 \u0caa\u0ccd\u0cb0\u0caf\u0ca4\u0ccd\u0ca8\u0cbf\u0cb8\u0cbf. \u0ca4\u0cc1\u0cb0\u0cc1\u0ca4 \u0cb8\u0cb9\u0cbe\u0caf\u0ca6\u0cb2\u0ccd\u0cb2\u0cbf 1930 \u0c95\u0cbe\u0cb2\u0ccd \u0cae\u0cbe\u0ca1\u0cbf.",
    "Malayalam": "\u0d15\u0d4d\u0d37\u0d2e\u0d3f\u0d15\u0d4d\u0d15\u0d23\u0d02, \u0d12\u0d30\u0d41 \u0d38\u0d3e\u0d19\u0d4d\u0d15\u0d47\u0d24\u0d3f\u0d15 \u0d2a\u0d4d\u0d30\u0d36\u0d4d\u0d28\u0d02 \u0d09\u0d23\u0d4d\u0d1f\u0d3e\u0d2f\u0d3f. \u0d26\u0d2f\u0d35\u0d3e\u0d2f\u0d3f \u0d2a\u0d3f\u0d28\u0d4d\u0d28\u0d40 \u0d35\u0d40\u0d23\u0d4d\u0d1f\u0d41\u0d02 \u0d36\u0d4d\u0d30\u0d2e\u0d3f\u0d15\u0d4d\u0d15\u0d41\u0d15. \u0d05\u0d1f\u0d3f\u0d2f\u0d28\u0d4d\u0d24\u0d30 \u0d38\u0d39\u0d3e\u0d2f\u0d24\u0d4d\u0d24\u0d3f\u0d28\u0d3e\u0d2f\u0d3f 1930 \u0d0e\u0d28\u0d4d\u0d28 \u0d28\u0d2e\u0d4d\u0d2a\u0d30\u0d3f\u0d7d \u0d35\u0d3f\u0d33\u0d3f\u0d15\u0d4d\u0d15\u0d41\u0d15.",
    "Punjabi": "\u0a2e\u0a41\u0a06\u0a2b \u0a15\u0a30\u0a28\u0a3e, \u0a15\u0a4b\u0a08 \u0a24\u0a71\u0a15\u0a28\u0a40\u0a15\u0a40 \u0a38\u0a2e\u0a38\u0a3f\u0a06 \u0a06 \u0a17\u0a08 \u0a39\u0a48\u0964 \u0a15\u0a3f\u0a30\u0a2a\u0a3e \u0a15\u0a30\u0a15\u0a47 \u0a2c\u0a3e\u0a26 \u0a35\u0a3f\u0a71\u0a1a \u0a26\u0a41\u0a2c\u0a3e\u0a30\u0a3e \u0a15\u0a4b\u0a38\u0a3c\u0a3f\u0a38\u0a3c \u0a15\u0a30\u0a4b\u0964 \u0a24\u0a41\u0a30\u0a70\u0a24 \u0a2e\u0a26\u0a26 \u0a32\u0a08 1930 '\u0a24\u0a47 \u0a15\u0a3e\u0a32 \u0a15\u0a30\u0a4b\u0964",
    "Odia": "\u0b15\u0b4d\u0b37\u0b2e\u0b3e \u0b15\u0b30\u0b28\u0b4d\u0b24\u0b41, \u0b0f\u0b15 \u0b2f\u0b3e\u0b28\u0b4d\u0b24\u0b4d\u0b30\u0b3f\u0b15 \u0b38\u0b2e\u0b38\u0b4d\u0b2f\u0b3e \u0b18\u0b1f\u0b3f\u0b1b\u0b3f\u0964 \u0b26\u0b2f\u0b3e\u0b15\u0b30\u0b3f \u0b2a\u0b30\u0b47 \u0b2a\u0b41\u0b28\u0b30\u0b3e\u0b2f \u0b1a\u0b47\u0b37\u0b4d\u0b1f\u0b3e \u0b15\u0b30\u0b28\u0b4d\u0b24\u0b41\u0964 \u0b1c\u0b30\u0b41\u0b30\u0b40 \u0b38\u0b39\u0b3e\u0b2f\u0b24\u0b3e \u0b2a\u0b3e\u0b07\u0b71 1930 \u0b15\u0b41 \u0b15\u0b32\u0b4d \u0b15\u0b30\u0b28\u0b4d\u0b24\u0b41\u0964",
    "Urdu": "\u0645\u0639\u0630\u0631\u062a\u060c \u0627\u06cc\u06a9 \u062a\u06a9\u0646\u06cc\u06a9\u06cc \u0645\u0633\u0626\u0644\u06c1 \u067e\u06cc\u0634 \u0622\u06cc\u0627 \u06c1\u06d2\u06d4 \u0628\u0631\u0627\u06c1 \u06a9\u0631\u0645 \u0628\u0639\u062f \u0645\u06cc\u06ba \u062f\u0648\u0628\u0627\u0631\u06c1 \u06a9\u0648\u0634\u0634 \u06a9\u0631\u06cc\u06ba\u06d4 \u0641\u0648\u0631\u06cc \u0645\u062f\u062f \u06a9\u06d2 \u0644\u06cc\u06d2 1930 \u067e\u0631 \u06a9\u0627\u0644 \u06a9\u0631\u06cc\u06ba\u06d4",
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
        )

    detected_lang, failed = detect_language(user_message)

    if failed or not detected_lang:
        clarify = (
            "We could not detect your language. Please reply with your preferred language:\n"
            "Hindi / Tamil / Telugu / Marathi / Bengali / "
            "Gujarati / Kannada / Malayalam / Punjabi / Odia / Urdu / English"
        )
        return Response(content=build_twiml(clarify), media_type="application/xml")

    # RAG retrieval with similarity threshold + reranker
    try:
        context, sources, distances = query_rag(user_message)
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")
        context = ""
        sources = []
        distances = []

    # If no chunks passed the similarity threshold, we don't have enough information
    has_relevant_context = bool(context and len(distances) > 0)

    system_prompt = f"""You are Citizen Fraud Shield, an AI assistant helping Indian citizens identify
potential fraud in suspicious calls, messages, or payment requests.

You will be given:
1. A user's description of something suspicious.
2. Reference material from verified advisories (CERT-In, RBI, NCRB) AND safe scenarios.

Your job:
- Classify the situation as one of: "Fraud", "Suspicious", "Safe", OR "Need More Information".
- Use "Need More Information" when the user's description is too vague, unrelated to the reference material,
  or the reference material does not clearly match the situation. This is a safe fallback to avoid false alarms.
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
  "verdict": "Fraud" | "Suspicious" | "Safe" | "Need More Information",
  "confidence": "High" | "Medium" | "Low",
  "reasoning": "short explanation in {detected_lang}",
  "next_steps": ["step 1 in {detected_lang}", "step 2 in {detected_lang}"] or ["Ask specific question 1", "Ask specific question 2"] for Need More Information,
  "matched_pattern": "fraud pattern name or null"
}}"""

    if has_relevant_context:
        context_block = "\n\n".join(
            f"[Source: {s}]\n{c}" for s, c in zip(sources, context.split("\n\n"))
        )
    else:
        context_block = "No specific advisories found for this situation. The query does not closely match known fraud patterns or safe scenarios."

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

        verdict_type = verdict.get("verdict", "Unknown")
        emoji = EMOJI_MAP.get(verdict_type, "")
        lines = [
            f"{emoji} *{verdict_type}* (confidence: {verdict.get('confidence', '-')})",
            "",
            verdict.get("reasoning", ""),
        ]
        if verdict.get("matched_pattern"):
            lines.append(f"\n_Pattern: {verdict['matched_pattern']}_")
        if verdict.get("next_steps"):
            lines.append("")
            for step in verdict["next_steps"]:
                lines.append(f"\u2022 {step}")
        if verdict_type in ("Fraud", "Suspicious"):
            lines.append("\n\U0001F6A8 Helpline: 1930")
        elif verdict_type == "Safe":
            lines.append("\n\u2705 Stay safe! Reach out if you have more questions.")

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
