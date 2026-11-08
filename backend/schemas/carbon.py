"""
================================================================================
  Schemas  —  Carbon Footprint & ML Clustering
================================================================================
"""
from __future__ import annotations

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class CarbonActivityInput(BaseModel):
    category: str = Field(..., description="Activity category (e.g. electricity_india, diesel_fuel_oil, passenger_car, natural_gas_scf, air_travel_long)", example="electricity_india")
    consumption: float = Field(..., ge=0, description="Consumption value in standard activity units", example=1500.0)
    label: Optional[str] = Field(None, description="Optional custom source label", example="Main Pune Campus Grid")


class CarbonCalculateRequest(BaseModel):
    activities: List[CarbonActivityInput]


class CarbonRecordOutput(BaseModel):
    activity_source: str
    consumption: float
    factor: float
    footprint_mt: float
    cluster_id: int
    classification: str
    risk_level: str  # "low", "medium", "critical"
    color: str


class CarbonCalculateResponse(BaseModel):
    records: List[CarbonRecordOutput]
    total_footprint_mt: float
    summary_by_risk: Dict[str, float]
    recommendations: List[str]

