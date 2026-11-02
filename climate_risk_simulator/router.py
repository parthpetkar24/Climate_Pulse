"""FastAPI router for the Climate Risk & Adaptation Simulator.

Mount with:  app.include_router(climate_risk_simulator.router)
Adjust ``prefix`` below if the repo uses a different API naming convention.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from .config import (
    ASSUMPTIONS,
    DEFAULT_COEFFICIENTS,
    DEFAULT_TARGETS,
    DISCLAIMER,
    INTERVENTION_RANGES,
    RANGES,
    TARGET_SOURCES,
)
from .engine import SimulationError, run_simulation
from .schemas import (
    DefaultsResponse,
    Intervention,
    InterventionType,
    SimulationRequest,
    SimulationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/climate-simulator",
    tags=["Climate Risk & Adaptation Simulator"],
)


def _example_request() -> SimulationRequest:
    return SimulationRequest(
        interventions=[
            Intervention(type=InterventionType.farm_ponds, intensity=2.0),
            Intervention(type=InterventionType.traffic_restrictions, intensity=20.0),
            Intervention(type=InterventionType.green_cover, intensity=5.0),
        ],
        duration_years=10,
    )


@router.get(
    "/defaults",
    response_model=DefaultsResponse,
    summary="Parameter ranges, default coefficients, targets, assumptions and an example request",
)
def get_defaults() -> DefaultsResponse:
    return DefaultsResponse(
        disclaimer=DISCLAIMER,
        ranges={k: [float(v[0]), float(v[1])] for k, v in RANGES.items()},
        intervention_ranges={
            k: {"min": v[0], "max": v[1], "unit": v[2]} for k, v in INTERVENTION_RANGES.items()
        },
        default_coefficients=DEFAULT_COEFFICIENTS,
        default_targets=DEFAULT_TARGETS,
        target_sources=TARGET_SOURCES,
        assumptions=ASSUMPTIONS,
        example_request=_example_request(),
    )


@router.post(
    "/simulate",
    response_model=SimulationResponse,
    summary="Simulate baseline vs intervention scenario and score against Indian targets",
    description=(
        "Runs a deterministic, simplified scenario model. Results are illustrative and are "
        "NOT validated climate forecasts. See `meta.assumptions` in the response."
    ),
)
def simulate(request: SimulationRequest) -> SimulationResponse:
    try:
        return run_simulation(request)
    except SimulationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected simulator failure")
        raise HTTPException(status_code=500, detail="Simulation failed unexpectedly.") from exc
