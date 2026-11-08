"""
================================================================================
  ClimatePulse  —  Unified Production Backend API
  PCCOE Hackathon  —  Climate Intelligence Platform
================================================================================
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to sys.path so modules can find each other cleanly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.core.config import settings
from backend.core.logging import setup_logging, logger
from backend.routers import (
    aqi_router,
    forecast_router,
    simulator_router,
    carbon_router,
    dashboard_router,
)
from backend.services.aqi_service import aqi_service
from backend.services.forecast_service import forecast_service

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for warm-up and graceful shutdown."""
    logger.info("=" * 65)
    logger.info(f"  Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 65)

    # Preload models in background on startup
    try:
        aqi_service.load()
        logger.info("✓ AQI Explainability model loaded into memory.")
    except Exception as e:
        logger.warning(f"⚠ AQI model deferred loading: {e}")

    try:
        forecast_service.load()
        logger.info("✓ Hyperlocal Forecaster initialized.")
    except Exception as e:
        logger.warning(f"⚠ Forecaster deferred loading: {e}")

    yield

    logger.info(f"Shutting down {settings.APP_NAME}...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Unified Climate Intelligence Platform combining Hyperlocal Air Quality Forecasting, "
        "SHAP-Powered Explainability with cited WHO guidelines, Climate Adaptation Risk Simulation, "
        "and Machine Learning Carbon Footprint Audit."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include Domain Routers ────────────────────────────────────────────────────
app.include_router(aqi_router)
app.include_router(forecast_router)
app.include_router(simulator_router)
app.include_router(carbon_router)
app.include_router(dashboard_router)


# ── Global Root & Health Endpoints ───────────────────────────────────────────
@app.get("/", tags=["System"])
def root():
    return {
        "title": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "documentation": "/docs",
        "interactive_redoc": "/redoc",
        "modules": {
            "aqi_explainability": "/api/aqi",
            "hyperlocal_forecaster": "/api/forecast",
            "climate_simulator": "/api/climate-simulator",
            "carbon_footprint": "/api/carbon",
            "dashboard": "/api/dashboard",
        },
    }


@app.get("/health", tags=["System"])
def health_check():
    try:
        import torch
        torch_found = True
    except ImportError:
        torch_found = False

    return {
        "status": "healthy",
        "environment": "production" if not settings.DEBUG else "debug",
        "models": {
            "aqi_model": {
                "loaded": aqi_service._is_loaded,
                "file": str(settings.AQI_MODEL_PATH.name),
                "exists": settings.AQI_MODEL_PATH.exists(),
            },
            "forecaster_lstm": {
                "pytorch_available": torch_found,
                "model_weights": str(settings.FORECAST_MODEL_PATH.name),
                "exists": settings.FORECAST_MODEL_PATH.exists(),
            },
            "climate_simulator": {
                "status": "ready",
                "rules_engine": "deterministic_cpcb_ncap",
            },
            "carbon_clustering": {
                "status": "ready",
                "algorithm": "kmeans_unsupervised",
            },
            "groq_llm": {
                "configured": bool(settings.GROQ_API_KEY),
                "model": settings.GROQ_MODEL,
            },
        },
    }


# ── Global Exception Handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "error": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
