from climate_risk_simulator.config import aqi_category
from climate_risk_simulator.engine import run_simulation
from climate_risk_simulator.schemas import SimulationRequest
from climate_risk_simulator.scoring import aqi_progress, reduction_progress, warming_progress


def test_aqi_category_bands():
    assert aqi_category(50) == "Good" and aqi_category(100) == "Satisfactory"
    assert aqi_category(101) == "Moderate" and aqi_category(450) == "Severe"


def test_progress_functions():
    assert aqi_progress(200, 90, 100) == 1.0
    assert aqi_progress(200, 150, 100) == 0.5
    assert aqi_progress(200, 250, 100) == 0.0
    assert reduction_progress(200, 120, 40) == 1.0
    assert reduction_progress(200, 160, 40) == 0.5
    assert warming_progress(1.0, 1.5) == 1.0 and warming_progress(3.0, 1.5) == 0.5


def test_weights_normalised_and_labels():
    r = run_simulation(SimulationRequest(interventions=[{"type": "green_cover", "intensity": 5}]))
    assert abs(sum(t.weight for t in r.scoring.targets) - 1.0) < 1e-3
    kinds = {t.id: t.kind for t in r.scoring.targets}
    assert kinds["warming_limit"] == "illustrative" and kinds["aqi_cpcb_satisfactory"] == "official"
    assert 0 <= r.scoring.intervention_score <= 100


def test_target_override_changes_score():
    base = {"interventions": [{"type": "traffic_restrictions", "intensity": 40}]}
    easy = run_simulation(SimulationRequest(**base, targets={"aqi_target": 200})).scoring.intervention_score
    hard = run_simulation(SimulationRequest(**base, targets={"aqi_target": 60})).scoring.intervention_score
    assert easy >= hard
