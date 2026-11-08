"""
================================================================================
  Router  —  Hyperlocal AQI Forecaster (7-Day LSTM)
================================================================================
"""
from fastapi import APIRouter, HTTPException
from backend.schemas.forecast import ForecastRequest, ForecastResponse
from backend.services.forecast_service import forecast_service

router = APIRouter(prefix="/api/forecast", tags=["Hyperlocal Forecaster (LSTM)"])


@router.post(
    "/7day",
    response_model=ForecastResponse,
    summary="Predict multi-step 7-day AQI trajectory",
)
def predict_7day_forecast(req: ForecastRequest) -> ForecastResponse:
    try:
        return forecast_service.predict_7day(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")


@router.get(
    "/latest",
    response_model=ForecastResponse,
    summary="Get current 7-day forecast for Pune monitoring network",
)
def get_latest_forecast() -> ForecastResponse:
    try:
        req = ForecastRequest()
        return forecast_service.predict_7day(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")

