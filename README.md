# ClimatePulse — Unified Climate Intelligence Platform
**PCCOE Hackathon MVP**

ClimatePulse is an end-to-end climate analytics and decision-support system integrating:
1. **AQI Prediction & SHAP Explainability** — Machine learning (Random Forest / XGBoost) with local SHAP force decomposition, WHO Guideline retrieval (TF-IDF RAG), and Groq LLM cited health advisory.
2. **Hyperlocal Forecaster** — 7-day multi-step neural AQI forecasting using stacked LSTMs.
3. **Climate Risk & Adaptation Simulator** — Policy scenario simulation (farm ponds, traffic restrictions, green cover) with CPCB/NCAP target progress scoring.
4. **Carbon Footprint Audit Copilot** — Enterprise activity-level emission calculations with unsupervised KMeans risk clustering (Green, Amber, Critical Red).
5. **Interactive Full-Stack Web App** — Modern React 19 + Vite dashboard with live framer-motion visualizations and REST API connectivity.

---

## Architecture Overview

```
                      ┌─────────────────────────────────────────┐
                      │        ClimatePulse Frontend (Vite)     │
                      │   Dashboard | Predict | Explainability  │
                      └────────────────────┬────────────────────┘
                                           │ HTTP / Proxy (:5173 -> :8000)
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │    Unified FastAPI Backend (Port 8000)  │
                      │               backend/main.py           │
                      └────┬──────────┬──────────┬─────────┬────┘
                           │          │          │         │
            ┌──────────────┘          │          │         └──────────────┐
            ▼                         ▼          ▼                        ▼
 ┌──────────────────────┐  ┌────────────────┐  ┌──────────────┐  ┌───────────────────┐
 │   /api/aqi           │  │ /api/forecast  │  │ /api/climate │  │   /api/carbon     │
 │ • RF / XGBoost Model │  │ • 7-day Stacked│  │   simulator  │  │ • Emission Math   │
 │ • SHAP TreeExplainer │  │   LSTM Model   │  │ • CPCB / NCAP│  │ • KMeans Clusters │
 │ • WHO PDF RAG        │  │ • PyTorch /    │  │   Target     │  │ • CSV / Excel /   │
 │ • Groq LLM Advisory  │  │   extrapolator │  │   Scoring    │  │   PDF Audit Parser│
 └──────────────────────┘  └────────────────┘  └──────────────┘  └───────────────────┘
```

---

## Unified API Endpoints

Interactive Swagger UI documentation is available at: **`http://localhost:8000/docs`**

| Module | Method | Path | Description |
|---|---|---|---|
| **System** | `GET` | `/health` | System health, model readiness, and Groq status |
| **System** | `GET` | `/` | API module directory and status |
| **AQI Prediction** | `POST` | `/api/aqi/predict` | Predict AQI with local SHAP & WHO cited evidence |
| **Explainability** | `GET` | `/api/aqi/explainability/global` | Global feature importance across 25 features |
| **Forecasting** | `POST` | `/api/forecast/7day` | 7-day neural multi-step AQI trajectory |
| **Forecasting** | `GET` | `/api/forecast/latest` | Latest 7-day outlook for Pune |
| **Simulator** | `GET` | `/api/climate-simulator/defaults` | Baseline parameters, coefficients, and limits |
| **Simulator** | `POST` | `/api/climate-simulator/simulate` | Run policy interventions over 1–30 years |
| **Carbon Audit** | `POST` | `/api/carbon/calculate` | Compute emissions and KMeans risk profile |
| **Carbon Audit** | `POST` | `/api/carbon/upload` | Ingest enterprise emission files (.csv, .xlsx, .pdf) |
| **Dashboard** | `GET` | `/api/dashboard/overview` | Live regional statistics, trends, pollutant levels |

---

## Quickstart & Deployment Steps

### 1. Setup Virtual Environment & Install Dependencies
```bash
# From workspace root:
cd C:\PCCOE26
.\climatepulsevenv\Scripts\Activate.ps1

# Install backend dependencies:
pip install -r Climate_Pulse/backend/requirements.txt
```

### 2. Launch Backend API Server
```bash
cd C:\PCCOE26\Climate_Pulse
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
*API is live at: `http://localhost:8000`*  
*Swagger Documentation: `http://localhost:8000/docs`*

### 3. Launch Frontend Client
```bash
cd C:\PCCOE26\Climate_Pulse\frontend
npm install
npm run dev
```
*Frontend is live at: `http://localhost:5173`*

---

## Teammate Models & Training Instructions

If any teammate model needs to be re-trained or updated:

1. **AQI Explainability Model**:
   ```bash
   cd C:\PCCOE26\Climate_Pulse\aqi_explainability
   python run_all.py
   ```
   *Generates `models/best_aqi_model.joblib`, `models/background_sample.joblib`, and SHAP plots.*

2. **Hyperlocal Forecaster (LSTM)**:
   ```bash
   cd C:\PCCOE26\Climate_Pulse\hyperlocal_forecaster
   python -m model.train
   ```
   *Trains the PyTorch model and writes `outputs/best_model.pt` and `outputs/model_config.json`.*

3. **Climate Risk Simulator**:
   ```bash
   cd C:\PCCOE26\Climate_Pulse
   python -m pytest tests/ -q
   ```
   *Executes the 29 unit tests covering engine trajectory and scoring rules.*

4. **Carbon Footprint Audit**:
   ```bash
   cd C:\PCCOE26\Climate_Pulse\carbonfootprintmodel
   python carbonfootprint.py
   ```
