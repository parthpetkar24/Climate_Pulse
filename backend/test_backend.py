"""
================================================================================
  ClimatePulse Backend  —  End-to-End Integration Tests
================================================================================
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas.aqi import AQIPredictRequest
from backend.schemas.forecast import ForecastRequest
from backend.schemas.carbon import CarbonCalculateRequest, CarbonActivityInput
from backend.services.aqi_service import aqi_service
from backend.services.forecast_service import forecast_service
from backend.services.carbon_service import carbon_service
from backend.services.dashboard_service import dashboard_service


def test_aqi_service_predict():
    req = AQIPredictRequest(
        pm25=45.0,
        pm10=95.0,
        no2=25.0,
        o3=30.0,
        co=1.0,
        so2=8.0,
        date="2026-11-07",
        use_groq=False,
    )
    res = aqi_service.predict_and_explain(req)
    assert res.aqi > 0
    assert res.category in ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    assert len(res.shap_features) >= 3
    assert res.shap_base_value > 0


def test_aqi_service_global_importance():
    res = aqi_service.get_global_importance()
    assert len(res.features) >= 5
    assert res.total_features >= 5


def test_forecast_service_7day():
    req = ForecastRequest(horizon_days=7)
    res = forecast_service.predict_7day(req)
    assert res.horizon_days == 7
    assert len(res.forecast) == 7
    assert res.forecast[0].predicted_aqi > 0


def test_carbon_service_calculate():
    req = CarbonCalculateRequest(
        activities=[
            CarbonActivityInput(category="electricity_india", consumption=1000.0, label="Grid"),
            CarbonActivityInput(category="diesel_fuel_oil", consumption=500.0, label="Generators"),
            CarbonActivityInput(category="passenger_car", consumption=2000.0, label="Commute"),
        ]
    )
    res = carbon_service.calculate(req)
    assert res.total_footprint_mt > 0
    assert len(res.records) == 3
    assert "low" in res.summary_by_risk or "medium" in res.summary_by_risk or "critical" in res.summary_by_risk


def test_dashboard_service():
    res = dashboard_service.get_overview()
    assert res.city == "Pune"
    assert res.current_aqi > 0
    assert len(res.weekly_trend) == 7
    assert len(res.pollutant_breakdown) == 6


def test_aqi_service_live_fetch():
    res = aqi_service.fetch_live_aqi(city="Pune")
    assert res.city == "Pune"
    assert res.pm25 > 0
    assert res.pm10 > 0
    assert res.source is not None


if __name__ == "__main__":
    print("Running ClimatePulse Backend Service Integration Suite...")
    test_aqi_service_predict()
    print("✓ AQI Predict & SHAP Test passed")
    test_aqi_service_global_importance()
    print("✓ Global Explainability Test passed")
    test_aqi_service_live_fetch()
    print("✓ Live AQI Fetch & Fallback Test passed")
    test_forecast_service_7day()
    print("✓ Forecaster 7-Day Trajectory Test passed")
    test_carbon_service_calculate()
    print("✓ Carbon Footprint & KMeans ML Clustering Test passed")
    test_dashboard_service()
    print("✓ Dashboard Analytics Overview Test passed")
    print("\n🎉 ALL 6 BACKEND SERVICE TESTS PASSED SUCCESSFULLY!")
