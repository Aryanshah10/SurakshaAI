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


def _get_rag_answer(user_text: str) -> str:
    """
    Runs RAG retrieval + LLM to produce a contextual advisory answer.
    Falls back to a static message if RAG or LLM is unavailable.
    """
    try:
        from utils.rag_engine import query_rag
        context, sources = query_rag(user_text)

        if not context or not context.strip():
            return "Stay cautious. Verify caller identity through official channels. For help, call 1930."

        # Try LLM-powered answer
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=os.getenv("OMNIROUTE_URL", "http://localhost:20128/v1"),
                api_key=os.getenv("OMNIROUTE_API_KEY"),
            )
            prompt = f"""You are a citizen safety assistant for an Indian government fraud-detection platform.
The citizen has sent a message. Answer helpfully using the context below.
Be concise and clear. Use simple language. Keep the response under 150 words.
If the topic involves fraud or scams, end with: "For help, call national cybercrime helpline 1930."

Context:
{context}

Citizen message: {user_text}

Answer:"""

            response = client.chat.completions.create(
                model=os.getenv("OMNIROUTE_MODEL", "google/gemini-2.5-flash"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"[scam] LLM call failed, returning RAG context directly: {e}")
            # Fallback: return the raw RAG context truncated
            return f"Advisory: {context[:400]}... For help, call 1930."

    except Exception as e:
        logger.warning(f"[scam] RAG retrieval failed: {e}")
        return "Stay cautious. Verify caller identity through official channels. For help, call 1930."


def _build_response(turns: list) -> tuple[ScamAssessResponse, dict | None]:
    """
    Runs detection, builds response + raw alert dict.
    Returns (ScamAssessResponse, alert_dict or None)
    alert_dict is only returned when risk is high or critical.
    """
    result = detect_call(turns, verbose=False)
    alert  = generate_mha_alert(result) if result else None
 
    score = (result["rolling_score"] / 15) if result else 0.0
    score = min(score, 1.0)
 
    if score >= 0.8:
        risk = RiskLevel.critical
    elif score >= 0.5:
        risk = RiskLevel.high
    elif score >= 0.2:
        risk = RiskLevel.medium
    else:
        risk = RiskLevel.low
 
    red_flags = alert["matched_patterns"] if alert else []
 
    ncrb_draft = None
    if risk in [RiskLevel.high, RiskLevel.critical]:
        ncrb_draft = (
            f"I received a suspicious call. The caller used phrases indicating: "
            f"{', '.join(red_flags[:3]) if red_flags else 'scam patterns detected'}. "
            f"Filing this complaint on the National Cybercrime Reporting Portal."
        )

    # For high/critical risk: urgent action message
    # For low/medium risk: run RAG to give a helpful contextual answer
    if risk in [RiskLevel.high, RiskLevel.critical]:
        recommended = "DO NOT pay anything. Hang up and call 1930 immediately."
    else:
        # Use RAG + LLM for a contextual, helpful response
        full_text = " ".join(turns)
        recommended = _get_rag_answer(full_text)
 
    response = ScamAssessResponse(
        risk_level=risk,
        confidence=round(score, 2),
        scam_type="digital_arrest" if score > 0.5 else "unknown",
        red_flags=red_flags,
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