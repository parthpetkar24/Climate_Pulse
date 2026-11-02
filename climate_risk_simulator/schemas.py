"""Pydantic (v2) request/response schemas for the simulator."""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import (
    INTERVENTION_RANGES,
    RANGES,
    ModelCoefficients,
    TargetConfig,
)


def _f(name: str, default, description: str):
    lo, hi = RANGES[name]
    return Field(default, ge=lo, le=hi, description=description)


class InterventionType(str, Enum):
    farm_ponds = "farm_ponds"
    traffic_restrictions = "traffic_restrictions"
    green_cover = "green_cover"


class Intervention(BaseModel):
    type: InterventionType
    intensity: float = Field(
        ...,
        description=(
            "Dose of the intervention. Units depend on type: farm_ponds = ponds per km2 "
            "(0.1-20); traffic_restrictions = % reduction in vehicle activity (1-60); "
            "green_cover = percentage points of area newly greened (0.5-30)."
        ),
    )

    @model_validator(mode="after")
    def _check_range(self) -> "Intervention":
        lo, hi, unit = INTERVENTION_RANGES[self.type.value]
        if not (lo <= self.intensity <= hi):
            raise ValueError(
                f"intensity for '{self.type.value}' must be between {lo} and {hi} ({unit}); got {self.intensity}"
            )
        return self


class BaselineParameters(BaseModel):
    temperature_c: float = _f("baseline_temperature_c", 27.5, "Annual mean air temperature at start (deg C).")
    ndvi: float = _f("baseline_ndvi", 0.35, "Annual mean NDVI at start (0-0.9).")
    aqi: float = _f("baseline_aqi", 150.0, "Annual mean AQI at start (CPCB 0-500 scale).")
    green_cover_pct: float = _f(
        "baseline_green_cover_pct", 25.0,
        "Existing forest/tree cover as % of area. Default is an approximate national-scale figure; supply a local value.",
    )
    warming_rate_c_per_year: float = _f(
        "warming_rate_c_per_year", 0.03, "Background warming trend without intervention (deg C / year)."
    )
    ndvi_trend_per_year: float = _f(
        "ndvi_trend_per_year", -0.002, "Background NDVI trend without intervention (per year)."
    )
    aqi_trend_pct_per_year: float = _f(
        "aqi_trend_pct_per_year", 1.0, "Background AQI growth without intervention (% per year, compound)."
    )


class SimulationRequest(BaseModel):
    baseline: BaselineParameters = Field(default_factory=BaselineParameters)
    interventions: List[Intervention] = Field(
        ..., min_length=1, max_length=3, description="One entry per intervention type (no duplicates)."
    )
    duration_years: int = _f("duration_years", 10, "Simulation length in years (1-30).")
    start_year: int = _f("start_year", 2025, "Calendar year of year 0 (labels only).")
    area_km2: float = _f("area_km2", 100.0, "Study area in km2 (reported; informs ponds count).")
    coefficients: Optional[ModelCoefficients] = Field(
        None, description="Override the illustrative effect coefficients."
    )
    targets: Optional[TargetConfig] = Field(None, description="Override the scoring targets/weights.")

    @field_validator("interventions")
    @classmethod
    def _no_duplicates(cls, v: List[Intervention]) -> List[Intervention]:
        types = [i.type for i in v]
        if len(set(types)) != len(types):
            raise ValueError("Each intervention type may appear at most once.")
        return v

    @model_validator(mode="after")
    def _check_green_cover_total(self) -> "SimulationRequest":
        added = sum(i.intensity for i in self.interventions if i.type == InterventionType.green_cover)
        if self.baseline.green_cover_pct + added > 100.0:
            raise ValueError("baseline.green_cover_pct + green_cover intensity cannot exceed 100%.")
        return self


# ----------------------------- Response models -----------------------------
class YearPoint(BaseModel):
    year: int
    year_index: int
    temperature_c: float
    ndvi: float
    aqi: float
    aqi_category: str
    green_cover_pct: float


class DeltaPoint(BaseModel):
    year: int
    year_index: int
    temperature_c: float
    ndvi: float
    aqi: float
    green_cover_pct: float


class MetricSummary(BaseModel):
    metric: Literal["temperature_c", "ndvi", "aqi", "green_cover_pct"]
    unit: str
    initial: float
    baseline_final: float
    intervention_final: float
    change_vs_baseline: float = Field(..., description="intervention_final - baseline_final")
    pct_change_vs_baseline: Optional[float]
    mean_change_vs_baseline: float = Field(..., description="Mean yearly difference over years 1..N")
    favourable_direction: Literal["lower", "higher"]
    favourable: bool


class TargetResult(BaseModel):
    id: str
    name: str
    metric: str
    kind: Literal["official", "official-proxy", "illustrative"]
    source: str
    threshold: float
    unit: str
    baseline_value: float
    intervention_value: float
    baseline_progress: float = Field(..., description="0-1 progress toward target")
    intervention_progress: float
    baseline_met: bool
    intervention_met: bool
    weight: float
    notes: str


class Scoring(BaseModel):
    baseline_score: float = Field(..., description="0-100 weighted progress, baseline scenario")
    intervention_score: float
    score_improvement: float
    targets: List[TargetResult]
    method: str


class Attribution(BaseModel):
    intervention: InterventionType
    final_year_effect: Dict[str, float] = Field(
        ..., description="Final-year change vs baseline if this intervention were applied alone."
    )


class Meta(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    disclaimer: str
    deterministic: bool
    assumptions: List[str]
    coefficients_used: ModelCoefficients
    targets_used: TargetConfig
    area_km2: float
    estimated_total_ponds: Optional[int]


class SimulationResponse(BaseModel):
    meta: Meta
    years: List[int]
    baseline: List[YearPoint]
    intervention: List[YearPoint]
    intervention_low: List[YearPoint] = Field(..., description="Low-effect uncertainty band")
    intervention_high: List[YearPoint] = Field(..., description="High-effect uncertainty band")
    delta: List[DeltaPoint] = Field(..., description="intervention - baseline per year")
    summary: List[MetricSummary]
    attribution: List[Attribution]
    scoring: Scoring
    explanations: List[str]


class DefaultsResponse(BaseModel):
    disclaimer: str
    ranges: Dict[str, List[float]]
    intervention_ranges: Dict[str, Dict[str, object]]
    default_coefficients: ModelCoefficients
    default_targets: TargetConfig
    target_sources: Dict[str, str]
    assumptions: List[str]
    example_request: SimulationRequest
