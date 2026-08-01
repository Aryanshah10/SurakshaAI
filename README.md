# SurakshaAI

**AI-powered Digital Public Safety Platform** — built for the ET AI Hackathon 2026

Defeating counterfeiting, fraud, and digital arrest scams through a unified intelligence platform for citizens and law enforcement.

## Problem Statement

India registered 1.14 million cybercrime complaints in 2023 — a 60% jump from 2022 — and digital arrest scams alone defrauded citizens of over ₹1,776 crore in the first nine months of 2024. These are industrialised operations, not opportunistic crimes: fraud compounds running spoofed numbers, AI-generated voices, and fake government portals, alongside persistent counterfeit currency circulation flagged in the RBI's 2025 Annual Report. Law enforcement doesn't lack evidence after the fact — it lacks intelligence *before* mass victimisation occurs.

**SurakshaAI** shifts law enforcement from reactive case investigation to predictive threat neutralisation, converging financial transaction intelligence, communication network analysis, computer vision, and real-time public safety coordination into one platform.

## Team

| Name |
|---|
| Saksham |
| Shaurya Mishra |
| Sakshi Sahu |
| Aryan Shah |

## Live Modules

| Module | What it does |
|---|---|
| **Scam Detection** | Two-layer digital arrest scam classifier — rule-based pattern scoring + Groq LLM escalation for ambiguous calls, flags before financial transfer occurs |
| **Currency Scan** | YOLOv8 classifier verifying note authenticity, with denomination classification in progress |
| **Fraud Graph** | Neo4j-backed graph intelligence linking accounts, devices, and phones into fraud ring clusters with court-admissible intelligence packages |
| **Geospatial Hotspots** | Clusters fraud complaints + counterfeit seizure points into patrol-priority zones for command-centre use |
| **Citizen RAG Chatbot** | FAISS + ChromaDB-backed retrieval-augmented assistant answering citizen fraud queries |
| **WhatsApp Bot** | Multi-channel citizen fraud shield via Twilio, with guided NCRB reporting |
| **Live Alert Feed** | WebSocket-based real-time scam/fraud alert stream for the officer dashboard |
| **Officer Auth** | JWT-based login with role/department/station-scoped access |
| **Audit Logging** | Middleware logging every request for evidentiary auditability |

## Repository Structure
```
SurakshaAI/
│
├── backend/
│   ├── main.py                     # FastAPI entrypoint, mounts all routers
│   ├── auth.py                     # officer authentication
│   ├── officers.py                 # officer records
│   ├── requirements.txt
│   │
│   ├── routes/
│   │   ├── auth.py
│   │   ├── scam.py                 # digital arrest scam detection endpoints
│   │   ├── currency.py             # currency authenticity + denomination scan
│   │   ├── graph.py                # fraud network graph endpoints
│   │   ├── geospatial.py           # hotspot/patrol priority endpoints
│   │   ├── citizen.py              # citizen RAG chatbot endpoints
│   │   ├── websocket.py            # live alert feed
│   │   └── whatsapp.py             # WhatsApp bot webhook
│   │
│   ├── utils/
│   │   ├── scam_engine.py          # two-layer scam scoring engine
│   │   ├── graph_agent.py          # graph construction + clustering
│   │   ├── geo_query_engine.py     # hotspot lookup + patrol recommendations
│   │   ├── rag_engine.py           # citizen chatbot retrieval logic
│   │   ├── train_currency_model.py
│   │   ├── prepare_currency_data.py
│   │   └── auth.py                 # JWT token creation/validation
│   │
│   ├── models/
│   │   └── schemas.py              # Pydantic response models
│   │
│   ├── middleware/
│   │   └── audit_log.py            # request audit trail middleware
│   │
│   ├── scripts/
│   │   ├── clean_and_rebuild.py
│   │   ├── currency_copy.py
│   │   ├── prep_denom.py           # denomination dataset prep
│   │   └── split_dataset.py
│   │
│   ├── whatsapp_bot/
│   │   ├── whatsapp_bot.py
│   │   ├── embed_and_store.py
│   │   └── generate_verdict.py
│   │
│   ├── data/
│   │   ├── currency_model.pt       # genuine/fake authenticity classifier
│   │   ├── geospatial_data.json
│   │   ├── chunks.json / chunks_map.json / index.faiss   # RAG vector store
│   │   ├── safe_scenarios.json
│   │   └── graph/
│   │       ├── accounts_nodes.csv
│   │       ├── account_network_edges.csv
│   │       ├── account_linkages.csv
│   │       ├── device_fingerprints.csv
│   │       ├── call_records.csv
│   │       └── victim_reports.csv
│   │
│   ├── chroma_db/                  # ChromaDB persistent vector store (RAG)
│   ├── logs/
│   │   └── audit_log.jsonl
│   └── yolov8c-clas.pt
│
├── chroma_db/                      # top-level vector store (shared/legacy)
├── frontend/
│   ├── index.html                  # landing page — citizen/officer split
│   ├── login.html                  #

```

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI, Uvicorn |
| Auth | python-jose (JWT), passlib/bcrypt |
| Graph Database | Neo4j |
| Computer Vision | Ultralytics YOLOv8, OpenCV, Pillow |
| RAG / Retrieval | LangChain, sentence-transformers, FAISS, ChromaDB |
| LLM Providers | Groq, OpenAI |
| Messaging | Twilio (WhatsApp), python websockets |
| ML Utilities | scikit-learn, imbalanced-learn, pandas, numpy |
| Frontend | HTML, CSS, vanilla JavaScript |
| Ops | python-dotenv, pyngrok (tunneling for demo), aiofiles |


## Architecture Flow

```Landing Page → "Are you a Citizen or Officer?"
| |
/citizen /officer (JWT login)
├─ Fraud Risk Check ├─ Geospatial Heatmap
├─ Guided NCRB Report ├─ Currency Scanner
├─ AI Chatbot (RAG) ├─ Fraud Network Graph
└─ Currency Scanner └─ Live Scam Alert Feed
```

Currency Scanner is shared across both citizen and officer views, backed by the same YOLOv8 models.

## Setup

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # fill in your API keys (Groq, OpenAI, Twilio, Neo4j credentials)
python main.py
```
API docs available at `http://localhost:8000/docs` once running.

### Frontend
```bash
cd frontend
# serve statically, e.g.:
python -m http.server 5500
```
Open `http://localhost:5500/index.html`.

### Environment variables required (`.env`)
GROQ_API_KEY  

OPENAI_API_KEY  

TWILIO_ACCOUNT_SID  

TWILIO_AUTH_TOKEN  

NEO4J_URI  

NEO4J_USERNAME  

NEO4J_PASSWORD  

SECRET_KEY  # JWT signing key

## API Endpoints (mounted in `main.py`)

| Prefix | Module |
|---|---|
| `/login` | Officer authentication |
| `/scam/*` | Digital arrest scam detection |
| `/currency/*` | Currency authenticity + denomination scan |
| `/graph/*` | Fraud network graph queries |
| `/geospatial/*` | Hotspot + patrol priority queries |
| `/citizen/*` | RAG chatbot |
| `/ws/*` | Live alert WebSocket feed |
| `/whatsapp/*` | WhatsApp bot webhook |

Health check: `GET /health` · Root status: `GET /`

## Known Limitations

- Synthetic data supplements real datasets where no public source existed (call records, victim reports, counterfeit seizure geolocation) — documented per-module
- Denomination classification is in active development (`scripts/prep_denom.py`, `yolov8c-clas.pt`)
- Geospatial coordinates are approximate district-level centroids, not precise geocoded locations

## Judging Criteria Alignment

| Criteria | Weight | Addressed by |
|---|---|---|
| Innovation | 25% | Hybrid rule+LLM scam detection, Neo4j-backed fraud ring clustering |
| Business Impact | 25% | Pre-transfer scam alerting, patrol prioritisation, audit-logged evidentiary trail |
| Technical Excellence | 20% | Multi-model CV pipeline, graph intelligence, RAG chatbot, WebSocket live feed |
| Scalability | 15% | Modular FastAPI routers, shared models across citizen/officer views |
| User Experience | 15% | Single command-centre for officers, multi-channel (web + WhatsApp) for citizens |
