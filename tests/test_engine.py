import pytest
from pydantic import ValidationError

from climate_risk_simulator.engine import run_simulation, run_trajectory
from climate_risk_simulator.config import DEFAULT_COEFFICIENTS
from climate_risk_simulator.schemas import BaselineParameters, SimulationRequest


def make(interventions, **kw):
    return SimulationRequest(interventions=interventions, **kw)


ALL = [
    {"type": "farm_ponds", "intensity": 2},
    {"type": "traffic_restrictions", "intensity": 20},
    {"type": "green_cover", "intensity": 5},
]


def test_deterministic_and_reproducible():
    a = run_simulation(make(ALL, duration_years=10)).model_dump()
    b = run_simulation(make(ALL, duration_years=10)).model_dump()
    assert a == b


def test_series_lengths_and_years():
    r = run_simulation(make(ALL, duration_years=7, start_year=2030))
    assert r.years == list(range(2030, 2038))
    for series in (r.baseline, r.intervention, r.intervention_low, r.intervention_high, r.delta):
        assert len(series) == 8


def test_year_zero_matches_baseline():
    r = run_simulation(make(ALL, duration_years=5))
    assert r.delta[0].temperature_c == 0 and r.delta[0].aqi == 0 and r.delta[0].ndvi == 0


def test_interventions_improve_all_metrics():
    r = run_simulation(make(ALL, duration_years=10))
    d = r.delta[-1]
    assert d.temperature_c < 0 and d.aqi < 0 and d.ndvi > 0 and d.green_cover_pct > 0
    assert r.scoring.intervention_score > r.scoring.baseline_score


def test_traffic_only_affects_aqi_and_not_vegetation():
    r = run_simulation(make([{"type": "traffic_restrictions", "intensity": 30}], duration_years=5))
    assert r.delta[-1].aqi < 0
    assert r.delta[-1].ndvi == 0 and r.delta[-1].green_cover_pct == 0


def test_more_intensity_gives_bigger_effect():
    lo = run_simulation(make([{"type": "green_cover", "intensity": 2}], duration_years=10)).delta[-1]
    hi = run_simulation(make([{"type": "green_cover", "intensity": 20}], duration_years=10)).delta[-1]
    assert hi.ndvi > lo.ndvi and hi.temperature_c < lo.temperature_c


def test_ramp_up_is_gradual():
    r = run_simulation(make([{"type": "green_cover", "intensity": 10}], duration_years=10))
    ndvi = [d.ndvi for d in r.delta]
    assert ndvi == sorted(ndvi) and ndvi[1] < ndvi[5]


def test_uncertainty_band_ordering():
    r = run_simulation(make(ALL, duration_years=10))
    assert r.intervention_low[-1].aqi >= r.intervention[-1].aqi >= r.intervention_high[-1].aqi
    assert r.intervention_low[-1].ndvi <= r.intervention[-1].ndvi <= r.intervention_high[-1].ndvi


def test_bounds_respected():
    b = BaselineParameters(ndvi=0.89, aqi=495, aqi_trend_pct_per_year=10)
    pts = run_trajectory(b, [], DEFAULT_COEFFICIENTS, 30, 2025)
    assert all(p.aqi <= 500 and p.ndvi <= 0.9 for p in pts)


def test_attribution_present_for_each_intervention():
    r = run_simulation(make(ALL, duration_years=10))
    assert {a.intervention.value for a in r.attribution} == {"farm_ponds", "traffic_restrictions", "green_cover"}


def test_estimated_pond_count():
    r = run_simulation(make([{"type": "farm_ponds", "intensity": 2}], area_km2=50))
    assert r.meta.estimated_total_ponds == 100


def test_disclaimer_and_assumptions_present():
    r = run_simulation(make(ALL))
    assert "NOT scientifically validated" in r.meta.disclaimer
    assert len(r.meta.assumptions) > 5 and r.explanations


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interventions": []},
        {"interventions": [{"type": "farm_ponds", "intensity": 2}, {"type": "farm_ponds", "intensity": 3}]},
        {"interventions": [{"type": "farm_ponds", "intensity": 999}]},
        {"interventions": [{"type": "unknown", "intensity": 1}]},
        {"interventions": ALL, "duration_years": 0},
        {"interventions": ALL, "duration_years": 100},
        {"interventions": ALL, "baseline": {"aqi": 900}},
        {"interventions": [{"type": "green_cover", "intensity": 30}], "baseline": {"green_cover_pct": 80}},
    ],
)
def test_invalid_inputs_rejected(kwargs):
    with pytest.raises(ValidationError):
        SimulationRequest(**kwargs)
