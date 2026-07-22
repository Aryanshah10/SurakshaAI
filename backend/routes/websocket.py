from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List
import asyncio, json
from collections import deque
from datetime import datetime

router = APIRouter(tags=["WebSocket"])

# ─── In-memory alert history (last 50) ────────────────────────────────────────
# deque with maxlen auto-drops oldest when full
alert_history: deque = deque(maxlen=50)

# Global connection pool
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        """Send to all connected officer dashboards. Drop dead connections."""
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()


@router.websocket("/ws/alerts")
async def alert_websocket(websocket: WebSocket):
    """
    Officer dashboard connects here.
    TM2 wires frontend with:
        const ws = new WebSocket("ws://localhost:8000/ws/alerts")
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data)
            if (data.type === "scam_alert") { showAlert(data) }
        }
 
    On connect: replays last 10 alerts so officer sees recent history immediately.
    Every 30s: sends heartbeat to keep connection alive.
    On new scam: broadcast_alert() pushes in real time.
    """
    await manager.connect(websocket)
    # Replay recent alerts on connect so officer isn't starting blind
    for past_alert in list(alert_history)[-10:]:
        try:
            await websocket.send_text(json.dumps({
                "type":      "scam_alert",
                "replayed":  True,
                **past_alert,
            }))
        except Exception:
            break
 
    try:
        while True:
            await websocket.send_text(json.dumps({
                "type":              "heartbeat",
                "timestamp":         datetime.utcnow().isoformat(),
                "active_connections": len(manager.active),
            }))
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
 
 
# ─── Called by scam.py on HIGH/CRITICAL detections ────────────────────────────
async def broadcast_alert(alert_data: dict):
    """
    Saves alert to history + pushes to all connected officer dashboards.
    Called from routes/scam.py when risk_level is HIGH or CRITICAL.
    """
    payload = {
        "type":      "scam_alert",
        "timestamp": datetime.utcnow().isoformat(),
        **alert_data,
    }
 
    # Save to in-memory history (GET /api/scam/alerts reads this)
    alert_history.append(payload)
 
    # Push live to all connected officer dashboards
    await manager.broadcast(payload)