"""
================================================================================
  Schemas  —  Hyperlocal AQI Forecaster (LSTM)
================================================================================
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ForecastStep(BaseModel):
    step: int
    day: str
    date: str
    predicted_aqi: float
    category: str
    color: str


class ForecastRequest(BaseModel):
    station_name: Optional[str] = Field("Revenue Colony-Shivajinagar, Pune - IITM", description="Station to forecast for")
    horizon_days: int = Field(7, ge=1, le=14, description="Forecast horizon in days")
    recent_aqi_values: Optional[List[float]] = Field(None, description="Optional custom recent 14-day sequence")


class ForecastResponse(BaseModel):
    station_name: str
    horizon_days: int
    forecast: List[ForecastStep]
    model_type: str
    is_live_model: bool
    status_message: Optional[str] = None

