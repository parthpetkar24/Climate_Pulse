"""
================================================================================
  Router  —  Dashboard Analytics & Overview
================================================================================
"""
from fastapi import APIRouter, HTTPException
from backend.schemas.dashboard import DashboardOverviewResponse
from backend.services.dashboard_service import dashboard_service

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard Overview"])


@router.get(
    "/overview",
    response_model=DashboardOverviewResponse,
    summary="Get regional air quality analytics, weekly/monthly trends, and pollutant status",
)
def get_dashboard_overview() -> DashboardOverviewResponse:
    try:
        return dashboard_service.get_overview()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")

