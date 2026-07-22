from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Existing authentication
from auth import authenticate_user

# AI middleware
from middleware.audit_log import AuditLogMiddleware

# AI routes
from routes import (
    auth,
    scam,
    currency,
    graph,
    geospatial,
    citizen,
    websocket,
    whatsapp
)

app = FastAPI(
    title="SurakshaAI",
    description="AI-powered Digital Public Safety Platform",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "null",
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Audit middleware
app.add_middleware(AuditLogMiddleware)


# ---------------- Authentication ----------------

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


from utils.auth import create_access_token

@app.post("/login")
def login(data: LoginRequest):

    officer = authenticate_user(
        data.username,
        data.password
    )

    if officer is None:
        return {
            "success": False,
            "message": "Invalid username or password"
        }

    token = create_access_token({
        "sub": data.username,
        "role": officer["role"]
    })

    return {
        "success": True,
        "access_token": token,
        "token_type": "bearer",
        "username": data.username,
        "name": officer["name"],
        "email": officer["email"],
        "department": officer["department"],
        "badge_id": officer["badge_id"],
        "station": officer["station"],
        "role": officer["role"]
    }
'''
@app.on_event("startup")
def warm_up_geo_engine():
    try:
        from utils.geo_query_engine import load_data
        load_data()
    except Exception as e:
        print(f"[WARN] geospatial engine warm-up failed, hotspot endpoints "
              f"will degrade gracefully: {e}")'''
# ---------------- Health ----------------


@app.get("/")
def root():
    return {
        "status": "ok",
        "platform": "SurakshaAI",
        "modules": [
            "authentication",
            "scam_detection",
            "currency_scan",
            "fraud_graph",
            "geospatial",
            "citizen_rag",
            "whatsapp_bot"
            "geospatial_hotspots"
        ],
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------- AI Routers ----------------

app.include_router(auth.router)
app.include_router(scam.router)
app.include_router(currency.router)
app.include_router(graph.router)
app.include_router(geospatial.router)
app.include_router(citizen.router)
app.include_router(websocket.router)
app.include_router(whatsapp.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )