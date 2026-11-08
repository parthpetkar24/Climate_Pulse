"""
================================================================================
  Schemas  —  Dashboard Overview
================================================================================
"""
from __future__ import annotations

from typing import List, Dict, Any
from pydantic import BaseModel


class WeeklyPoint(BaseModel):
    day: str
    aqi: float
    pm25: float
    pm10: float
    no2: float


class MonthlyPoint(BaseModel):
    month: str
    aqi: float


class CategoryDistPoint(BaseModel):
    name: str
    value: int
    color: str


class PollutantItem(BaseModel):
    name: str
    value: float
    unit: str
    limit: float


class DashboardOverviewResponse(BaseModel):
    city: str
    current_aqi: float
    category: str
    color: str
    weekly_avg: float
    best_day: str
    best_aqi: float
    weekly_trend: List[WeeklyPoint]
    monthly_trend: List[MonthlyPoint]
    category_distribution: List[CategoryDistPoint]
    pollutant_breakdown: List[PollutantItem]

