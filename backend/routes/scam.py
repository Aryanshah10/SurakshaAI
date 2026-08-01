import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import APIRouter, Depends
from models.schemas import ScamAssessRequest, ScamAssessResponse, RiskLevel
from utils.auth import require_officer
from utils.scam_engine import detect_call, generate_mha_alert
from routes.websocket import broadcast_alert, alert_history
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scam", tags=["Scam Detection"])


import re

_GREETING_PATTERNS = [
    "hi", "hello", "hey", "heyo", "hola", "hii", "hiii", "helloo",
    "how are you", "how r u", "hru", "howdy", "sup", "wassup", "what's up",
    "good morning", "good evening", "good afternoon", "good night",
    "namaste", "namaskar", "namaskaram", "pranam", "vanakkam",
    "nomoshkar", "sat sri akal",
    "kaise ho", "kya haal hai", "kya kar rahe ho", "kese ho",
    "sab theek", "kaisa hai",
]

_ABOUT_BOT_PATTERNS = [
    r"what\s+do\s+(you|u)\s+do",
    r"what\s+(can|all)\s+(you|u)\s+do",
    r"what\s+(is|are)\s+(your|ur)\s+(purpose|job|work|role)",
    r"who\s+are\s+(you|u)",
    r"tell\s+me\s+(about|abt)\s+(yourself|urself|you|u)",
    r"introduce\s+(yourself|urself|you|u)",
    r"what\s+is\s+(this|the)\s+bot",
    r"kaun\s+ho\s+(tum|aap)",
    r"aap\s+kya\s+karte\s+ho",
    r"aap\s+kaun\s+ho",
    r"aapke?\s+baare\s+mein\s+batao",
    r"what\s+are\s+you",
    r"whats?\s+your\s+name",
]

def _is_greeting(text: str) -> bool:
    lower = text.lower().strip().rstrip("!?.,;:")
    words = lower.split()
    if len(words) > 4:
        return False
    if lower in _GREETING_PATTERNS:
        return True
    if len(words) <= 2 and words[0] in _GREETING_PATTERNS:
        return True
    return False

def _is_about_bot(text: str) -> bool:
    lower = text.lower().strip().rstrip("!?.,;:")
    words = lower.split()
    if len(words) > 8:
        return False
    return any(re.search(p, lower) for p in _ABOUT_BOT_PATTERNS)


def _get_rag_answer(user_text: str, risk_level: str = "low") -> str:
    """
    Runs RAG retrieval + LLM to produce a contextual advisory answer.
    Greeting check only fires when risk_level is 'low'.
    """
    # Only show greeting for genuinely low-risk, short greetings
    if risk_level == "low" and (_is_greeting(user_text) or _is_about_bot(user_text)):
        return (
            "👋 **Namaste! I am SurakshaAI — Citizen Fraud Shield.**\n\n"
            "I am an AI assistant designed to help citizens identify financial fraud, suspicious calls, and cyber scams.\n\n"
            "**Here is what you can ask me:**\n"
            "• Paste a suspicious message or call transcript — I'll analyze it for fraud risk.\n"
            "• Ask about scam types (UPI scams, Digital Arrest, OTP fraud, etc.).\n"
            "• Get safety tips and guidance on reporting scams (Emergency Helpline: 1930).\n\n"
            "How can I help you keep safe today?"
        )

    try:
        from utils.rag_engine import query_rag
        context, sources, distances = query_rag(user_text)

        if not context or not context.strip():
            if risk_level in ["high", "critical"]:
                return (
                    "This communication matches common pressure tactics used in cyber frauds.\n\n"
                    "**Precautions & Next Steps:**\n"
                    "• Hang up immediately and do not respond further.\n"
                    "• Never share OTPs, PINs, or bank account details.\n"
                    "• Verify caller claims directly through official websites or phone numbers.\n"
                    "• Report incidents on cybercrime.gov.in or call 1930."
                )
            else:
                return (
                    "I could not find specific advisory information for your query.\n\n"
                    "If you received a suspicious call, message, or payment request, "
                    "please describe the situation in detail and I will analyze it for fraud risk.\n\n"
                    "For immediate help, call the National Cybercrime Helpline: 1930."
                )

        # Try LLM-powered answer
        try:
            from groq import Groq
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            prompt = f"""You are Citizen Fraud Shield, an AI safety assistant for an Indian government fraud-detection platform.
The citizen provided this situation/transcript: "{user_text}"
Assessed Risk Level: {risk_level.upper()}

Using the verified advisory reference material below:
1. Provide a concise explanation of the situation/scam pattern.
2. Outline 3-4 concrete precautions and safety steps (e.g. do not share OTP, verify caller identity, report on cybercrime portal).
3. Keep the language simple, direct, reassuring, and clear.
4. Keep the entire output around 150-200 words.
5. End with helpline info: "National Cybercrime Helpline: 1930 | Report: cybercrime.gov.in"

Reference Material:
{context}

Answer:"""

            response = groq_client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"[scam] LLM call failed, returning RAG context directly: {e}")
            return (
                f"**Advisory Summary:** {context[:350]}...\n\n"
                "**Precautions:**\n"
                "• Never share banking credentials, PINs, or OTPs.\n"
                "• Do not stay on video calls under threat of arrest.\n"
                "• Call 1930 immediately to report suspicious activity."
            )

    except Exception as e:
        logger.warning(f"[scam] RAG retrieval failed: {e}")
        return "Stay cautious. Verify caller identity through official channels. For help, call 1930."


def _build_response(turns: list) -> tuple[ScamAssessResponse, dict | None]:
    """
    Runs detection, builds response + raw alert dict.
    Returns (ScamAssessResponse, alert_dict or None)
    alert_dict is only returned when risk is high or critical.
    """
    from utils.scam_engine import score_turn_rule_based

    result = detect_call(turns, verbose=False)
    alert  = generate_mha_alert(result) if result else None

    # Score the full concatenated text directly for single-message inputs
    full_text = " ".join(turns)
    direct_score, _ = score_turn_rule_based(full_text)

    # Use whichever score is higher: the rolling detection or the direct score
    rolling = result["rolling_score"] if result else 0.0
    final_score = max(rolling, direct_score)

    if final_score >= 8.0:
        risk = RiskLevel.critical
    elif final_score >= 4.0:
        risk = RiskLevel.high
    elif final_score >= 2.0:
        risk = RiskLevel.medium
    else:
        risk = RiskLevel.low

    ncrb_draft = None
    if risk in [RiskLevel.high, RiskLevel.critical]:
        ncrb_draft = (
            f"I received a suspicious call. The caller used pressure tactics and scam phrases. "
            f"Filing this complaint on the National Cybercrime Reporting Portal."
        )

    # Use RAG + LLM for a contextual, detailed explanation & precautions
    recommended = _get_rag_answer(full_text, risk_level=risk.value)

    response = ScamAssessResponse(
        risk_level=risk,
        confidence=round(min(final_score / 15.0, 1.0), 2),
        scam_type="digital_arrest" if final_score > 4.0 else "unknown",
        red_flags=[],
        recommended_action=recommended,
        ncrb_draft=ncrb_draft,
    )
 
    # Only return alert payload when actionable
    alert_payload = None
    if risk in [RiskLevel.high, RiskLevel.critical] and alert:
        alert_payload = {
            "risk_level":       risk.value,
            "confidence":       round(score, 2),
            "red_flags":        red_flags,
            "transcript_excerpt": alert.get("transcript_excerpt", ""),
            "recommended_action": alert.get("recommended_action", ""),
        }
 
    return response, alert_payload


@router.post("/assess", response_model=ScamAssessResponse)
async def assess_scam(body: ScamAssessRequest):
    """
    Citizen or officer pastes suspicious call transcript or message.
    Returns risk level, red flags, and pre-filled NCRB complaint draft.
    Uses two-layer detection: rule-based + OmniRoute LLM for gray zone.
    For low-risk inputs, uses RAG retrieval + LLM to give contextual advisory.
    """
    # Split multi-turn transcript by newline; single message treated as one turn
    turns = [t.strip() for t in body.text.split("\n") if t.strip()] or [body.text]
 
    response, alert_payload = _build_response(turns)
 
    # Broadcast to officer dashboards if high/critical
    if alert_payload:
        await broadcast_alert(alert_payload)
 
    return response


# ─── Officer-only: real-time alert feed placeholder ─
@router.get("/alerts", dependencies=[Depends(require_officer)])
def get_alerts():
    """
    Returns last 50 high/critical scam alerts.
    Officer dashboard polls this on load; live updates come via WebSocket.
    """
    return {
        "alerts": list(alert_history),
        "total":  len(alert_history),
    }