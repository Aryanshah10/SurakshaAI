import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import APIRouter, Depends, HTTPException
from models.schemas import GraphQueryRequest, GraphQueryResponse, GraphNode, GraphEdge
from utils.auth import require_officer
from utils.graph_agent import get_subgraph, build_intelligence_packages
from routes.websocket import broadcast_alert

router = APIRouter(prefix="/api/graph", tags=["Fraud Graph"])

@router.post("/query", response_model=GraphQueryResponse, dependencies=[Depends(require_officer)])
async def query_graph(body: GraphQueryRequest):
    """
    Officer inputs an account ID, phone number, or device ID.
    Returns subgraph of connected entities up to `depth` hops.
    Frontend renders this with react-force-graph.

    When reported or frozen accounts are found in the graph,
    a live alert is broadcast to all connected officer dashboards via WebSocket.
    """
    if not body.node_id:
        raise HTTPException(status_code=400, detail="node_id required")
 
    result = get_subgraph(body.node_id, body.depth)
 
    if not result["nodes"]:
        raise HTTPException(status_code=404, detail=f"Node '{body.node_id}' not found in graph")
 
    # ── Broadcast alerts for any reported/frozen accounts found in the graph ──
    for node in result["nodes"]:
        status = (node.get("account_status") or "").lower()
        if status in ("reported", "frozen"):
            risk_level = "CRITICAL" if status == "frozen" else "HIGH"
            title_prefix = "❄️ Frozen Account" if status == "frozen" else "📋 Reported Account"
            alert_data = {
                "alert_type": "graph_fraud",
                "risk_level": risk_level,
                "title": f"{title_prefix}: {node['id']}",
                "account_id": node["id"],
                "account_status": status,
                "fraud_score": node.get("fraud_score", 0),
                "bank": node.get("bank", "N/A"),
                "jurisdiction": node.get("jurisdiction", "N/A"),
                "message": (
                    f"Account {node['id']} (status: {status.upper()}) detected during "
                    f"fraud network graph query from node '{body.node_id}'. "
                    f"Fraud score: {node.get('fraud_score', 0)}. "
                    f"Bank: {node.get('bank', 'N/A')}. "
                    f"Jurisdiction: {node.get('jurisdiction', 'N/A')}."
                ),
                "matched_patterns": ["graph_reported_frozen_account"],
                "recommended_action": "Review account details and coordinate with bank for freeze/escalation.",
                "timestamp": datetime.utcnow().isoformat(),
            }
            await broadcast_alert(alert_data)
 
    return GraphQueryResponse(
        nodes=[
            GraphNode(
                id=n["id"],
                label=n["label"],
                fraud_score=n["fraud_score"],
                transaction_count=n["transaction_count"],
                is_fraudster=n.get("is_fraudster", False),
                account_status=n.get("account_status") or "N/A",
                bank=n.get("bank") or "N/A",
                jurisdiction=n.get("jurisdiction") or "N/A",
                fraud_rate=n.get("fraud_rate", 0.0),
            )
            for n in result["nodes"]
        ],
        edges=[
            GraphEdge(
                source=e["source"],
                target=e["target"],
                amount=e["amount"],
                timestamp=e["timestamp"] or "N/A",
                flagged=e["flagged"],
            )
            for e in result["edges"]
        ],
        fraud_ring_detected=result["fraud_ring_detected"],
        total_suspicious_amount=result["total_suspicious_amount"],
    )
 
 
@router.get("/intelligence", dependencies=[Depends(require_officer)])
def get_intelligence_packages():
    """
    Runs community detection on full graph.
    Returns court-admissible intelligence packages per fraud cluster.
    Officer can export these as evidence.
    """
    packages = build_intelligence_packages()
    if not packages:
        return {"packages": [], "message": "No fraud clusters detected"}
    return {"packages": packages, "total_clusters": len(packages)}