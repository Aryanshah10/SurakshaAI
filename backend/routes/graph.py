import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import APIRouter, Depends, HTTPException
from models.schemas import GraphQueryRequest, GraphQueryResponse, GraphNode, GraphEdge
from utils.auth import require_officer
from utils.graph_agent import get_subgraph, build_intelligence_packages

router = APIRouter(prefix="/api/graph", tags=["Fraud Graph"])

@router.post("/query", response_model=GraphQueryResponse, dependencies=[Depends(require_officer)])
def query_graph(body: GraphQueryRequest):
    """
    Officer inputs an account ID, phone number, or device ID.
    Returns subgraph of connected entities up to `depth` hops.
    Frontend renders this with react-force-graph.
    """
    if not body.node_id:
        raise HTTPException(status_code=400, detail="node_id required")
 
    result = get_subgraph(body.node_id, body.depth)
 
    if not result["nodes"]:
        raise HTTPException(status_code=404, detail=f"Node '{body.node_id}' not found in graph")
 
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