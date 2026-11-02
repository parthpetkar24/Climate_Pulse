"""Transparent, modular scenario engine (SIMPLIFIED MODEL - not a validated forecast).

No reinforcement learning or trained ML is used: there is no observational dataset
in scope to train against, so a fitted model would only dress up made-up numbers.
Instead the engine is a small, readable dose-response model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import (
    AQI_MAX,
    AQI_MIN,
    ASSUMPTIONS,
    DEFAULT_COEFFICIENTS,
    DEFAULT_TARGETS,
    DISCLAIMER,
    MODEL_NAME,
    MODEL_VERSION,
    NDVI_MAX,
    ModelCoefficients,
    aqi_category,
)
from .schemas import (
    Attribution,
    BaselineParameters,
    DeltaPoint,
    Intervention,
    InterventionType,
    Meta,
    MetricSummary,
    SimulationRequest,
    SimulationResponse,
    YearPoint,
)
from .scoring import build_explanations, score_scenarios


class SimulationError(ValueError):
    """Raised for inputs that pass schema validation but cannot be simulated."""


@dataclass
class _Effects:
    d_temp: float = 0.0       # deg C change (negative = cooling)
    d_ndvi: float = 0.0
    aqi_factor: float = 1.0   # multiplicative
    d_green_pct: float = 0.0


def _sat(dose: float, scale: float) -> float:
    return 1.0 - math.exp(-dose / scale) if dose > 0 else 0.0


def _maturity(t: int, ramp_years: float) -> float:
    if ramp_years <= 0:
        return 1.0 if t > 0 else 0.0
    return max(0.0, min(1.0, t / ramp_years))


def _effects_at(interventions: List[Intervention], c: ModelCoefficients, t: int, mult: float) -> _Effects:
    e = _Effects()
    for iv in interventions:
        dose = iv.intensity
        if iv.type == InterventionType.green_cover:
            m = _maturity(t, c.green_ramp_years)
            s = _sat(dose, c.green_dose_scale_pct)
            e.d_ndvi += mult * c.green_max_ndvi_gain * s * m
            e.d_temp -= mult * c.green_max_cooling_c * s * m
            e.aqi_factor *= 1.0 - min(0.9, mult * c.green_max_aqi_reduction_frac * s * m)
            e.d_green_pct += dose * m
        elif iv.type == InterventionType.farm_ponds:
            m = _maturity(t, c.pond_ramp_years)
            s = _sat(dose, c.pond_dose_scale_per_km2)
            e.d_ndvi += mult * c.pond_max_ndvi_gain * s * m
            e.d_temp -= mult * c.pond_max_cooling_c * s * m
            e.aqi_factor *= 1.0 - min(0.9, mult * c.pond_max_aqi_reduction_frac * s * m)
        elif iv.type == InterventionType.traffic_restrictions:
            m = _maturity(t, c.traffic_ramp_years)
            red = c.traffic_transport_share * c.traffic_compliance * (dose / 100.0)
            e.aqi_factor *= 1.0 - min(0.9, mult * red * m)
            e.d_temp -= mult * c.traffic_cooling_c_per_pct * dose * m
    return e


def _baseline_values(b: BaselineParameters, t: int):
    temp = b.temperature_c + b.warming_rate_c_per_year * t
    ndvi = min(NDVI_MAX, max(0.0, b.ndvi + b.ndvi_trend_per_year * t))
    aqi = min(AQI_MAX, max(AQI_MIN, b.aqi * (1.0 + b.aqi_trend_pct_per_year / 100.0) ** t))
    return temp, ndvi, aqi


def _point(year0: int, t: int, temp: float, ndvi: float, aqi: float, green: float) -> YearPoint:
    return YearPoint(
        year=year0 + t,
        year_index=t,
        temperature_c=round(temp, 4),
        ndvi=round(ndvi, 4),
        aqi=round(aqi, 2),
        aqi_category=aqi_category(aqi),
        green_cover_pct=round(green, 3),
    )


def run_trajectory(
    baseline: BaselineParameters,
    interventions: List[Intervention],
    coeffs: ModelCoefficients,
    duration_years: int,
    start_year: int,
    effect_multiplier: float = 1.0,
) -> List[YearPoint]:
    """Year-by-year trajectory. With an empty intervention list this is the baseline."""
    points: List[YearPoint] = []
    for t in range(duration_years + 1):
        temp, ndvi, aqi = _baseline_values(baseline, t)
        e = _effects_at(interventions, coeffs, t, effect_multiplier)
        temp += e.d_temp
        ndvi = min(NDVI_MAX, max(0.0, ndvi + e.d_ndvi))
        aqi = min(AQI_MAX, max(AQI_MIN, aqi * e.aqi_factor))
        green = min(100.0, baseline.green_cover_pct + e.d_green_pct)
        points.append(_point(start_year, t, temp, ndvi, aqi, green))
    return points


def _summaries(base: List[YearPoint], scen: List[YearPoint]) -> List[MetricSummary]:
    spec = [
        ("temperature_c", "deg C", "lower"),
        ("ndvi", "NDVI (0-1)", "higher"),
        ("aqi", "AQI (CPCB 0-500)", "lower"),
        ("green_cover_pct", "% of area", "higher"),
    ]
    out: List[MetricSummary] = []
    for metric, unit, direction in spec:
        b_final = getattr(base[-1], metric)
        s_final = getattr(scen[-1], metric)
        change = s_final - b_final
        pct = (change / b_final * 100.0) if b_final not in (0, 0.0) else None
        diffs = [getattr(s, metric) - getattr(b, metric) for b, s in zip(base[1:], scen[1:])]
        mean_change = sum(diffs) / len(diffs) if diffs else 0.0
        favourable = change < 0 if direction == "lower" else change > 0
        out.append(
            MetricSummary(
                metric=metric,
                unit=unit,
                initial=getattr(base[0], metric),
                baseline_final=b_final,
                intervention_final=s_final,
                change_vs_baseline=round(change, 4),
                pct_change_vs_baseline=None if pct is None else round(pct, 2),
                mean_change_vs_baseline=round(mean_change, 4),
                favourable_direction=direction,
                favourable=favourable,
            )
        )
    return out


def _attribution(req: SimulationRequest, coeffs: ModelCoefficients, base: List[YearPoint]) -> List[Attribution]:
    out: List[Attribution] = []
    for iv in req.interventions:
        solo = run_trajectory(req.baseline, [iv], coeffs, req.duration_years, req.start_year)[-1]
        b = base[-1]
        out.append(
            Attribution(
                intervention=iv.type,
                final_year_effect={
                    "temperature_c": round(solo.temperature_c - b.temperature_c, 4),
                    "ndvi": round(solo.ndvi - b.ndvi, 4),
                    "aqi": round(solo.aqi - b.aqi, 2),
                    "green_cover_pct": round(solo.green_cover_pct - b.green_cover_pct, 3),
                },
            )
        )
    return out


def run_simulation(req: SimulationRequest) -> SimulationResponse:
    coeffs = req.coefficients or DEFAULT_COEFFICIENTS
    targets = req.targets or DEFAULT_TARGETS

    if (targets.weight_aqi + targets.weight_ncap + targets.weight_green_cover + targets.weight_temperature) <= 0:
        raise SimulationError("At least one target weight must be greater than zero.")

    n, y0 = req.duration_years, req.start_year
    base = run_trajectory(req.baseline, [], coeffs, n, y0)
    scen = run_trajectory(req.baseline, req.interventions, coeffs, n, y0)
    low = run_trajectory(req.baseline, req.interventions, coeffs, n, y0, coeffs.effect_low_multiplier)
    high = run_trajectory(req.baseline, req.interventions, coeffs, n, y0, coeffs.effect_high_multiplier)

    delta = [
        DeltaPoint(
            year=s.year,
            year_index=s.year_index,
            temperature_c=round(s.temperature_c - b.temperature_c, 4),
            ndvi=round(s.ndvi - b.ndvi, 4),
            aqi=round(s.aqi - b.aqi, 2),
            green_cover_pct=round(s.green_cover_pct - b.green_cover_pct, 3),
        )
        for b, s in zip(base, scen)
    ]

    summary = _summaries(base, scen)
    scoring = score_scenarios(base, scen, targets)
    attribution = _attribution(req, coeffs, base)

    pond = next((i for i in req.interventions if i.type == InterventionType.farm_ponds), None)
    total_ponds: Optional[int] = int(round(pond.intensity * req.area_km2)) if pond else None

    meta = Meta(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        model_type="simplified dose-response scenario model (no ML / no RL)",
        disclaimer=DISCLAIMER,
        deterministic=True,
        assumptions=ASSUMPTIONS,
        coefficients_used=coeffs,
        targets_used=targets,
        area_km2=req.area_km2,
        estimated_total_ponds=total_ponds,
    )

    return SimulationResponse(
        meta=meta,
        years=[p.year for p in base],
        baseline=base,
        intervention=scen,
        intervention_low=low,
        intervention_high=high,
        delta=delta,
        summary=summary,
        attribution=attribution,
        scoring=scoring,
        explanations=build_explanations(req, base, scen, summary, scoring, attribution),
    )
