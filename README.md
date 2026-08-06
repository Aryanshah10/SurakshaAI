# SurakshaAI

**AI-powered Digital Public Safety Platform** — built for the ET AI Hackathon 2026

Defeating counterfeiting, fraud, and digital arrest scams through a unified intelligence platform for citizens and law enforcement.

## Problem Statement

India registered 1.14 million cybercrime complaints in 2023 — a 60% jump from 2022 — and digital arrest scams alone defrauded citizens of over ₹1,776 crore in the first nine months of 2024. These are industrialised operations, not opportunistic crimes: fraud compounds running spoofed numbers, AI-generated voices, and fake government portals, alongside persistent counterfeit currency circulation flagged in the RBI's 2025 Annual Report. Law enforcement doesn't lack evidence after the fact — it lacks intelligence *before* mass victimisation occurs.

**SurakshaAI** shifts law enforcement from reactive case investigation to predictive threat neutralisation, converging financial transaction intelligence, fake currency detection, communication network analysis, and real-time public safety coordination into one platform.

## Live Modules

| Module | What it does |
|---|---|
| **Officer Auth** | JWT-based login with role/department/station-scoped access |
| **Currency Scan** | YOLOv8 classifier verifying note authenticity |
| **Fraud Graph** | NetworkX graph intelligence linking accounts, devices, and phones into fraud ring clusters with court-admissible intelligence packages |
| **Geospatial Hotspots** | Clusters fraud complaints + counterfeit seizure points into patrol-priority zones for command-centre use |
| **Citizen RAG Chatbot** | ChromaDB-backed RAG assistant answering citizen fraud queries |
| **Live Alert Feed** | WebSocket-based real-time scam/fraud alert stream for the officer dashboard |
| **WhatsApp Bot** | Multi-channel citizen fraud shield via Twilio, with guided NCRB reporting |

## Repository Structure
```
SurakshaAI/
│
├── backend/
│   ├── main.py                     # FastAPI entrypoint, mounts all routers
│   ├── auth.py                     # officer authentication
│   ├── requirements.txt
|   ├── run_ngrok.py                # Runs the entire codebase without the need for deployment
│   │
│   ├── routes/
│   │   ├── auth.py
│   │   ├── scam.py                 # digital arrest scam detection endpoints
│   │   ├── currency.py             # currency authenticity 
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
│   │   └── split_dataset.py
│   │
│   ├── whatsapp_bot/
│   │   ├── embed_and_store.py
│   │
│   ├── data/
│   │   ├── currency_model.pt       # genuine/fake authenticity classifier
│   │   ├── geospatial_data.json
│   │   ├── chunks.json             
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
└── frontend/
    ├── index.html                  # landing page — citizen/officer split
    ├── login.html
    ├── officer.html
    ├── citizen.html
    ├── counterfeit.html
    ├── script.js
    └── style.css              

```

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, uvicorn |
| Auth | python-jose (JWT), passlib |
| Fraud Network Graph | NetworkX |
| Currency Scan | Ultralytics YOLOv8s, Pillow |
| Website Chatbot | RAG, LangChain, sentence-transformers, ChromaDB |
| API Keys | HuggingFace, Groq, Gemini  |
| Geospatial Intelligence | DBScan (sklearn clustering) |
| Messaging | Twilio (WhatsApp), python websockets, pyngrok |
| ML Utilities | scikit-learn, imbalanced-learn, pandas, numpy |
| Frontend | HTML, CSS, vanilla JavaScript |

## Architecture Flow

```Landing Page → "Are you a Citizen or Officer?"

/citizen                 /officer (JWT Authorisation)
├─ Fraud Risk Check     ├─ Geospatial Heatmap
├─ Guided NCRB Report   ├─ Currency Scanner
├─ RAG Chatbot          ├─ Fraud Network Graph
└─ Currency Scanner     └─ Live Scam Alert Feed
```

Currency Scanner is shared across both citizen and officer views, backed by the same YOLOv8s models.

## Setup

### Files to be created in backend
#### .env file with :
GROQ_API_KEY
GROQ_MODEL

TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
TWILIO_WHATSAPP_NUMBER

NEO4J_URI
NEO4J_USERNAME
NEO4J_PASSWORD

OMNIROUTE_API_KEY
OMNIROUTE_URL
OMNIROUTE_MODEL
OMNIROUTE_SCAM_MODEL  

SECRET_KEY # JWT signing key

#### officer.py file
Containing the officer details

### Running Backend + Frontend 
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # fill in your API keys (Groq, Twilio, Gemini credentials)
python main.py                  # Server for the main website
ngrok http 8000                 # Server for Twilio WhatsApp Bot

```
API docs available at `http://localhost:8000/docs` once running.

## API Endpoints (mounted in `main.py`)

| Prefix | Module |
|---|---|
| `/login` | Officer authentication |
| `/scam/*` | Digital arrest scam detection |
| `/currency/*` | Currency authenticity scan |
| `/graph/*` | Fraud network graph queries |
| `/geospatial/*` | Hotspot + patrol priority queries |
| `/citizen/*` | RAG chatbot |
| `/ws/*` | Live alert WebSocket feed |
| `/whatsapp/*` | WhatsApp bot webhook |

Health check: `GET /health` · Root status: `GET /`
