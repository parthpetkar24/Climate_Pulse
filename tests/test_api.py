from fastapi import FastAPI
from fastapi.testclient import TestClient

from climate_risk_simulator import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)
URL = "/api/climate-simulator"

GOOD = {
    "baseline": {"temperature_c": 28, "ndvi": 0.3, "aqi": 180},
    "interventions": [
        {"type": "farm_ponds", "intensity": 2},
        {"type": "traffic_restrictions", "intensity": 25},
        {"type": "green_cover", "intensity": 6},
    ],
    "duration_years": 8,
}


def test_defaults_endpoint():
    r = client.get(f"{URL}/defaults")
    assert r.status_code == 200
    body = r.json()
    assert "NOT scientifically validated" in body["disclaimer"]
    assert body["example_request"]["interventions"]
    assert "green_cover" in body["intervention_ranges"]


def test_simulate_ok_structure():
    r = client.post(f"{URL}/simulate", json=GOOD)
    assert r.status_code == 200
    b = r.json()
    for key in ("meta", "years", "baseline", "intervention", "intervention_low", "intervention_high",
                "delta", "summary", "attribution", "scoring", "explanations"):
        assert key in b
    assert len(b["years"]) == 9
    assert b["scoring"]["intervention_score"] > b["scoring"]["baseline_score"]


def test_validation_errors_are_422():
    bad = dict(GOOD, duration_years=500)
    assert client.post(f"{URL}/simulate", json=bad).status_code == 422
    assert client.post(f"{URL}/simulate", json={"interventions": []}).status_code == 422
    dup = dict(GOOD, interventions=[{"type": "green_cover", "intensity": 3}] * 2)
    assert client.post(f"{URL}/simulate", json=dup).status_code == 422


def test_zero_weights_returns_400():
    bad = dict(GOOD, targets={"weight_aqi": 0, "weight_ncap": 0, "weight_green_cover": 0, "weight_temperature": 0})
    r = client.post(f"{URL}/simulate", json=bad)
    assert r.status_code == 400


def test_unknown_route_does_not_exist():
    assert client.get(f"{URL}/nope").status_code == 404
