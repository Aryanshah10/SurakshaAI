import os
import re
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
 
OMNIROUTE_URL     = os.getenv("OMNIROUTE_URL", "http://localhost:20128/v1")
OMNIROUTE_API_KEY = os.getenv("OMNIROUTE_API_KEY")
OMNIROUTE_SCAM_MODEL = os.getenv("OMNIROUTE_SCAM_MODEL", "groq/llama-3.3-70b-versatile")
 
# ─── Layer 1: Rule-based pattern weights ─────────────────────────────────────
PATTERN_WEIGHTS = {
    r"\b(arrest|non[- ]bailable|warrant|jail|custody)\b":                      3.0,
    r"\b(cbi|enforcement directorate|customs|cyber crime|income tax department|trai)\b": 2.5,
    r"\b(otp|one time password)\b":                                            2.5,
    r"\bdo not disconnect\b|\bstay on the line\b|\bdon't hang up\b":           3.0,
    r"\b(transfer|deposit|pay)\b.*\b(account|rbi|verification)\b":            3.0,
    r"\bconfidential(ity)? investigation\b":                                   2.0,
    r"\bimmediately|urgent(ly)?|right now|within (an? )?hour\b":               1.0,
    r"\bblocked|frozen|seized|suspended\b":                                    1.5,
    r"\bcooperate|non[- ]cooperation\b":                                       1.5,
}
 
ALERT_THRESHOLD = 6.0
GRAY_ZONE_LOW   = 2.0
GRAY_ZONE_HIGH  = 6.0

def score_turn_rule_based(text: str):
    text_l = text.lower()
    score, matched = 0.0, []
    for pattern, weight in PATTERN_WEIGHTS.items():
        if re.search(pattern, text_l):
            score += weight
            matched.append(pattern)
    return score, matched
 
 
# ─── Layer 2: OmniRoute LLM ──────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = """
Turn: "This is Inspector Sharma from CBI, case number 482913. You must cooperate or face arrest."
Label: scam (confidence 0.95)
 
Turn: "Good morning, this is HDFC Bank. Could you confirm the last 4 digits of your mobile?"
Label: legitimate (confidence 0.9)
 
Turn: "Please share the OTP sent to your mobile so I can update your KYC immediately."
Label: scam (confidence 0.9)
 
Turn: "Your package is out for delivery today between 2-5 PM, will you be available?"
Label: legitimate (confidence 0.95)
"""
 
 
def _get_llm_client():
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            return Groq(api_key=groq_key), "groq"
        except ImportError:
            pass
            
    if OMNIROUTE_API_KEY:
        try:
            from openai import OpenAI
            return OpenAI(
                base_url=OMNIROUTE_URL,
                api_key=OMNIROUTE_API_KEY,
            ), "omniroute"
        except Exception:
            pass
    return None, None


def score_turn_llm(text: str, recent_context: str = "", client=None):
    client_type = None
    if client is None:
        client, client_type = _get_llm_client()
    else:
        client_type = "groq" if hasattr(client, "chat") else "omniroute"

    if client is None:
        return None, None

    prompt = f"""You are a scam-call detection classifier for an Indian public-safety platform.
Classify the LATEST turn as: neutral, scam, scam_response, legitimate, or suspicious.

Examples:
{FEW_SHOT_EXAMPLES}

Recent call context:
{recent_context}

Latest turn:
"{text}"

Respond ONLY as JSON: {{"label": "...", "confidence": 0.0}}"""

    try:
        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile") if client_type == "groq" else OMNIROUTE_SCAM_MODEL
        kwargs = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 50,
        }
        if client_type == "groq":
            kwargs["response_format"] = {"type": "json_object"}
            
        response = client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return parsed.get("label"), float(parsed.get("confidence", 0.5))
    except Exception as e:
        print(f"[WARN] Layer 2 LLM call failed: {e}")
        return None, None
 
 
# ─── Combined Detection ───────────────────────────────────────────────────────
def detect_call(turns: list, client=None, verbose=True):
    rolling_score  = 0.0
    context_window = []
    alert_result   = None
 
    for step, text in enumerate(turns, 1):
        turn_score, matched       = score_turn_rule_based(text)
        layer2_label, layer2_conf = None, None
 
        if GRAY_ZONE_LOW <= turn_score < GRAY_ZONE_HIGH:
            recent_context = " | ".join(context_window[-3:])
            layer2_label, layer2_conf = score_turn_llm(
                text, recent_context, client=client
            )
            if (
                layer2_label in ("scam", "scam_response")
                and layer2_conf
                and layer2_conf >= 0.7
            ):
                turn_score += 2.5 * layer2_conf
 
        rolling_score = rolling_score * 0.7 + turn_score
        context_window.append(text)
 
        if verbose:
            l2 = f" | layer2={layer2_label}({layer2_conf:.2f})" if layer2_label else ""
            print(f"Step {step:>2} | turn={turn_score:.1f} | rolling={rolling_score:.1f}{l2}")
 
        if alert_result is None and rolling_score >= ALERT_THRESHOLD:
            alert_result = {
                "alert_step":        step,
                "rolling_score":     round(rolling_score, 1),
                "matched_patterns":  matched,
                "transcript_so_far": list(context_window),
            }
            if verbose:
                print(f"  <<< ALERT at step {step}: SCAM THRESHOLD CROSSED")
 
    return alert_result
 
 
def generate_mha_alert(alert_result: dict, conversation_id=None):
    if alert_result is None:
        return None
    return {
        "conversation_id":    conversation_id,
        "alert_step":         alert_result["alert_step"],
        "confidence_score":   min(1.0, alert_result["rolling_score"] / 15),
        "matched_patterns":   list(set(alert_result["matched_patterns"])),
        "transcript_excerpt": " | ".join(alert_result["transcript_so_far"][-3:]),
        "recommended_action": "flag_number_to_telecom + notify_citizen + escalate_to_mha",
    }
 
 
# ─── Test / Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    client = _get_omniroute_client()
    if client:
        print(f"[INFO] OmniRoute connected | model: {OMNIROUTE_SCAM_MODEL}\n")
    else:
        print("[INFO] Running Layer 1 only (no OmniRoute key)\n")
 
    df       = pd.read_csv("data/digital_arrest_scripts.csv")
    test_ids = df["conversation_id"].unique()[:3]
 
    for cid in test_ids:
        turns  = (
            df[df["conversation_id"] == cid]
            .sort_values("turn_id")["text"]
            .tolist()
        )
        print(f"\n=== Conversation {cid} ===")
        result = detect_call(turns, client=client)
        if result:
            alert = generate_mha_alert(result, conversation_id=cid)
            print("MHA ALERT:", json.dumps(alert, indent=2))