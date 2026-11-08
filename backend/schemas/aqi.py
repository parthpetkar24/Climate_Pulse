"""
================================================================================
  Schemas  —  AQI Prediction & Explainability
================================================================================
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class AQIPredictRequest(BaseModel):
    pm25: float = Field(..., ge=0, le=1000, description="PM2.5 concentration in µg/m³", example=45.0)
    pm10: float = Field(..., ge=0, le=1000, description="PM10 concentration in µg/m³", example=95.0)
    no2: Optional[float] = Field(None, ge=0, le=1000, description="NO₂ concentration in µg/m³", example=25.0)
    o3: Optional[float] = Field(None, ge=0, le=1000, description="Ozone concentration in µg/m³", example=30.0)
    co: Optional[float] = Field(None, ge=0, le=100, description="CO concentration in mg/m³", example=1.0)
    so2: Optional[float] = Field(None, ge=0, le=1000, description="SO₂ concentration in µg/m³", example=8.0)
    date: Optional[str] = Field(None, description="Observation date in YYYY-MM-DD format", example="2026-11-07")
    use_groq: bool = Field(True, description="Whether to query Groq LLM for cited health explanation")


class SHAPFeature(BaseModel):
    feature: str
    value: float
    shap: float
    direction: str  # "increases" or "decreases"


class WHOEvidenceItem(BaseModel):
    citation: str
    text: str
    score: float


class AQIPredictResponse(BaseModel):
    aqi: float
    category: str
    color: str
    advice: str
    shap_base_value: float
    shap_features: List[SHAPFeature]
    who_evidence: List[WHOEvidenceItem]
    health_explanation: Optional[str] = None
    groq_called: bool = False
    narrative: Optional[str] = None


class GlobalImportanceItem(BaseModel):
    feature: str
    importance: float
    color: str


class GlobalImportanceResponse(BaseModel):
    features: List[GlobalImportanceItem]
    total_features: int


class LiveAQIResponse(BaseModel):
    city: str
    latitude: float
    longitude: float
    timestamp: str
    pm25: float
    pm10: float
    no2: Optional[float] = None
    o3: Optional[float] = None
    co: Optional[float] = None
    so2: Optional[float] = None
    source: str
    is_live: bool


