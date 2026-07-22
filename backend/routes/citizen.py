import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import APIRouter, HTTPException
from models.schemas import CitizenQueryRequest, CitizenQueryResponse, ScamAssessRequest, ScamAssessResponse
from routes.scam import assess_scam
from utils.rag_engine import query_rag
from openai import OpenAI
import os

router = APIRouter(prefix="/api/citizen", tags=["Citizen Shield"])
client = OpenAI(
    base_url=os.getenv("OMNIROUTE_URL", "http://localhost:20128/v1"),
    api_key=os.getenv("OMNIROUTE_API_KEY") 
)

SUPPORTED_LANGUAGES = ["en", "hi", "mr", "bn", "te", "ta", "gu", "kn", "ml", "pa", "od", "ur"]

LANGUAGE_NAMES = {
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
    "od": "Odia",
    "ur": "Urdu",
}
 
@router.post("/query", response_model=CitizenQueryResponse)
def citizen_query(body: CitizenQueryRequest):
    if body.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Language '{body.language}' not supported. Supported: {SUPPORTED_LANGUAGES}"
        )
 
    # Retrieve relevant chunks from FAISS
    context, sources = query_rag(body.question)
 
    language_name = LANGUAGE_NAMES.get(body.language, "English")
 
    prompt = f"""You are a citizen safety assistant for an Indian government fraud-detection platform.
Answer the question using ONLY the context below.
Be concise and clear. Use simple language suitable for any age group.
IMPORTANT: Respond in {language_name} only, regardless of the language of the question.
End with the national cybercrime helpline 1930 if the question involves fraud or scams.
 
Context:
{context}
 
Question: {body.question}
 
Answer in {language_name}:"""
 
    try:
        response = client.chat.completions.create(
            model=os.getenv("OMNIROUTE_MODEL", "google/gemini-2.5-flash"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        answer = response.choices[0].message.content.strip()
 
    except Exception as e:
        # OmniRoute down — return context directly as fallback
        answer = (
            f"Advisory: {context[:500]}..."
            if context
            else "Service temporarily unavailable. Call 1930 for immediate assistance."
        )
 
    return CitizenQueryResponse(
        answer=answer,
        language=body.language,
        source_references=sources,
        suggested_action="File complaint at cybercrime.gov.in",
        helpline="1930",
    )

@router.post("/assess", response_model=ScamAssessResponse)
async def citizen_assess(body: ScamAssessRequest):
    """Citizen-facing scam risk check — no auth required.
    Also triggers WebSocket broadcast to officer dashboards if high/critical."""
    return await assess_scam(body)