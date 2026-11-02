"""Standalone demo server (NOT for merging into the main repo).

    uvicorn demo_app:app --reload     ->  http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI

from climate_risk_simulator import router

app = FastAPI(title="Climate Risk & Adaptation Simulator (standalone demo)")
app.include_router(router)
