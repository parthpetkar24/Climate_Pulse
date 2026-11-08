"""
================================================================================
  ClimatePulse Backend  —  Core Configuration
================================================================================
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List
from pydantic import BaseModel

# Base paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Module paths
AQI_MODULE_DIR = PROJECT_ROOT / "aqi_explainability"
FORECASTER_MODULE_DIR = PROJECT_ROOT / "hyperlocal_forecaster"
SIMULATOR_MODULE_DIR = PROJECT_ROOT / "climate_risk_simulator"
CARBON_MODULE_DIR = PROJECT_ROOT / "carbonfootprintmodel"

# Load .env file from aqi_explainability/.env or PROJECT_ROOT/.env if present
def _load_env_file():
    for env_path in [PROJECT_ROOT / ".env", AQI_MODULE_DIR / ".env"]:
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass

_load_env_file()


class Settings(BaseModel):
    APP_NAME: str = "ClimatePulse Unified API"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # CORS origins
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ]

    # Model Artefact Paths
    AQI_MODEL_PATH: Path = AQI_MODULE_DIR / "models" / "best_aqi_model.joblib"
    AQI_MEDIANS_PATH: Path = AQI_MODULE_DIR / "models" / "feature_medians.joblib"
    AQI_FEATURES_PATH: Path = AQI_MODULE_DIR / "models" / "feature_names.json"
    AQI_BACKGROUND_PATH: Path = AQI_MODULE_DIR / "models" / "background_sample.joblib"
    AQI_WHO_PDF_PATH: Path = AQI_MODULE_DIR / "data" / "raw" / "9789240034433-eng.pdf"
    AQI_GLOBAL_SHAP_PATH: Path = AQI_MODULE_DIR / "outputs" / "shap" / "global_feature_importance.csv"

    # Forecaster Paths
    FORECAST_MODEL_PATH: Path = FORECASTER_MODULE_DIR / "outputs" / "best_model.pt"
    FORECAST_CONFIG_PATH: Path = FORECASTER_MODULE_DIR / "outputs" / "model_config.json"
    FORECAST_SCALER_PATH: Path = FORECASTER_MODULE_DIR / "data" / "processed" / "scaler_standard.pkl"
    FORECAST_DATA_PATH: Path = FORECASTER_MODULE_DIR / "data" / "raw" / "Dataset Daily Air Quality Index and Pollutant Conc" / "all_stations_combined.csv"

    # Groq API
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


settings = Settings()

