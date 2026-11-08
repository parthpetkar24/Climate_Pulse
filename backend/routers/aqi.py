"""
================================================================================
  Router  —  AQI Prediction & Explainability
================================================================================
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from backend.schemas.aqi import (
    AQIPredictRequest,
    AQIPredictResponse,
    GlobalImportanceResponse,
    LiveAQIResponse,
)
from backend.services.aqi_service import aqi_service

router = APIRouter(prefix="/api/aqi", tags=["AQI Explainability & Prediction"])


@router.post(
    "/predict",
    response_model=AQIPredictResponse,
    summary="Predict AQI with Local SHAP & Cited WHO Health Evidence",
)
def predict_aqi(req: AQIPredictRequest) -> AQIPredictResponse:
    try:
        return aqi_service.predict_and_explain(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get(
    "/explainability/global",
    response_model=GlobalImportanceResponse,
    summary="Global Feature Importance across Dataset (SHAP values)",
)
def get_global_explainability() -> GlobalImportanceResponse:
    try:
        return aqi_service.get_global_importance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability error: {str(e)}")


@router.get(
    "/live",
    response_model=LiveAQIResponse,
    summary="Fetch live ambient pollutant concentrations for a city or coordinates",
)
def get_live_aqi(
    city: str = Query("Pune", description="City name (e.g. Pune, Mumbai, Delhi)"),
    lat: Optional[float] = Query(None, description="Optional latitude coordinate"),
    lon: Optional[float] = Query(None, description="Optional longitude coordinate"),
) -> LiveAQIResponse:
    try:
        return aqi_service.fetch_live_aqi(city=city, lat=lat, lon=lon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Live AQI fetch error: {str(e)}")


