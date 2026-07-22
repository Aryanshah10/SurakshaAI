from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

# ─── AUTH ───────────────────────────────────────────
class UserRole(str, Enum):
    officer = "officer"
    citizen = "citizen"

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    role: UserRole

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[UserRole] = None


# ─── SCAM DETECTION (Module 1) ──────────────────────
class ScamAssessRequest(BaseModel):
    text: str                        # call transcript or message text
    language: Optional[str] = "en"  # language code

class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

class ScamAssessResponse(BaseModel):
    risk_level: RiskLevel
    confidence: float               # 0.0 to 1.0
    scam_type: Optional[str]        # e.g. "digital_arrest", "parcel_fraud"
    red_flags: List[str]            # list of detected red flag phrases
    recommended_action: str
    ncrb_draft: Optional[str]       # pre-filled complaint text if high risk


# ─── CURRENCY DETECTION (Module 2) ──────────────────
class CurrencyScanResponse(BaseModel):
    is_genuine: bool
    confidence: float
    red_flags: List[str]            # e.g. ["blurred watermark", "missing thread"]
    uv_simulation_note: str


# ─── FRAUD GRAPH (Module 3) ─────────────────────────
class GraphQueryRequest(BaseModel):
    node_id: str                    # UPI ID, phone, or account number
    depth: Optional[int] = 2       # how many hops to traverse

class GraphNode(BaseModel):
    id: str
    label: str                      # "account", "phone", "device"
    fraud_score: float
    transaction_count: int
    is_fraudster: Optional[bool] = False
    account_status: Optional[str] = "N/A"
    bank: Optional[str] = "N/A"
    jurisdiction: Optional[str] = "N/A"
    fraud_rate: Optional[float] = 0.0

class GraphEdge(BaseModel):
    source: str
    target: str
    amount: float
    timestamp: str
    flagged: bool

class GraphQueryResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    fraud_ring_detected: bool
    total_suspicious_amount: float


# ─── GEOSPATIAL (Module 4) ──────────────────────────
class GeospatialPoint(BaseModel):
    lat: float
    lon: float
    district: str
    state: str
    crime_type: str
    complaint_count: int
    year: int
    month: Optional[int]

class GeospatialResponse(BaseModel):
    points: List[GeospatialPoint]
    total_complaints: int
    hotspot_states: List[str]


# ─── RAG CITIZEN AGENT (Module 5) ───────────────────
class CitizenQueryRequest(BaseModel):
    question: str
    language: Optional[str] = "en"

class CitizenQueryResponse(BaseModel):
    answer: str
    language: str
    source_references: List[str]
    suggested_action: Optional[str]
    helpline: Optional[str] = "1930"   # National Cybercrime Helpline


 
 
class HotspotSummary(BaseModel):
    cluster_id: str
    districts_spanned: str
    states_spanned: str
    dominant_crime_category: str
    event_count: int
    num_districts: int
    centroid_lat: float
    centroid_lon: float
    patrol_priority_score: float
 
 
class HotspotEvent(BaseModel):
    district: str
    state: str
    lat: float
    lon: float
    incident_category: str
    crime_type: str
    complaint_count: int
    priority_level: str
    year: int
    month: int
    cluster_id: int
 
 
class HotspotDetailResponse(BaseModel):
    summary: List[HotspotSummary]
    events: List[HotspotEvent]
 
 
class JointPatrolRecommendation(BaseModel):
    cluster_a: str
    cluster_b: str
    states_involved: str
    distance_km: float
    combined_priority_score: float