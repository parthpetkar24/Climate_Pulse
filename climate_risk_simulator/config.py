"""Configuration for the Climate Risk & Adaptation Simulator.

Everything in this file is either
  (a) a published Indian standard/policy value (see ``TargetConfig`` docstrings), or
  (b) an ILLUSTRATIVE coefficient chosen to give plausible orders of magnitude.

The coefficients in ``ModelCoefficients`` are NOT calibrated against observed data
and are NOT scientifically validated. They are exposed as request-level overrides so
a domain expert can replace them with locally calibrated values.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

MODEL_NAME = "climate-risk-adaptation-simplified-scenario-model"
MODEL_VERSION = "0.1.0"

DISCLAIMER = (
    "SIMPLIFIED ILLUSTRATIVE MODEL. Outputs are scenario explorations driven by "
    "transparent, uncalibrated coefficients. They are NOT scientifically validated "
    "forecasts and must not be used for policy, engineering or investment decisions "
    "without local calibration and expert review."
)

# ---------------------------------------------------------------------------
# Input ranges (single source of truth for validation + /defaults endpoint)
# ---------------------------------------------------------------------------
RANGES = {
    "baseline_temperature_c": (5.0, 40.0),        # annual mean air temp, deg C
    "baseline_ndvi": (0.0, 0.9),                  # annual mean NDVI
    "baseline_aqi": (0.0, 500.0),                 # CPCB AQI scale is 0-500
    "baseline_green_cover_pct": (0.0, 100.0),     # % of area under forest/tree cover
    "warming_rate_c_per_year": (0.0, 0.1),        # background trend
    "ndvi_trend_per_year": (-0.02, 0.02),
    "aqi_trend_pct_per_year": (-10.0, 10.0),
    "area_km2": (1.0, 100000.0),
    "duration_years": (1, 30),
    "start_year": (2000, 2100),
}

# Valid intensity range per intervention type: (min, max, unit)
INTERVENTION_RANGES = {
    "farm_ponds": (0.1, 20.0, "ponds per km2"),
    "traffic_restrictions": (1.0, 60.0, "% reduction in vehicle activity"),
    "green_cover": (0.5, 30.0, "percentage points of area newly greened"),
}

NDVI_MAX = 0.9
AQI_MIN = 0.0
AQI_MAX = 500.0


class ModelCoefficients(BaseModel):
    """Illustrative effect sizes. All are overridable per request."""

    # --- Green cover: saturating response  effect = max * (1 - exp(-dose/scale)) ---
    green_max_ndvi_gain: float = Field(0.15, ge=0, le=0.5)
    green_max_cooling_c: float = Field(1.0, ge=0, le=5.0)
    green_max_aqi_reduction_frac: float = Field(0.08, ge=0, le=0.5)
    green_dose_scale_pct: float = Field(10.0, gt=0, le=100)
    green_ramp_years: float = Field(5.0, ge=0, le=30)   # trees mature slowly

    # --- Farm ponds: dose = ponds per km2 ---
    pond_max_ndvi_gain: float = Field(0.05, ge=0, le=0.5)
    pond_max_cooling_c: float = Field(0.5, ge=0, le=5.0)
    pond_max_aqi_reduction_frac: float = Field(0.03, ge=0, le=0.5)  # dust suppression
    pond_dose_scale_per_km2: float = Field(3.0, gt=0, le=100)
    pond_ramp_years: float = Field(2.0, ge=0, le=30)

    # --- Traffic restrictions: linear in % vehicle-activity reduction ---
    # AQI reduction fraction = transport_share * compliance * (reduction_pct / 100)
    traffic_transport_share: float = Field(0.25, ge=0, le=1.0)  # vehicular share of ambient PM
    traffic_compliance: float = Field(0.8, ge=0, le=1.0)
    traffic_cooling_c_per_pct: float = Field(0.002, ge=0, le=0.05)  # waste-heat effect, minor
    traffic_ramp_years: float = Field(1.0, ge=0, le=30)

    # --- Uncertainty band: scale intervention effect magnitudes ---
    effect_low_multiplier: float = Field(0.5, ge=0, le=1.0)
    effect_high_multiplier: float = Field(1.5, ge=1.0, le=3.0)


class TargetConfig(BaseModel):
    """Targets used for scoring.

    ``kind`` tells the frontend how much to trust each one:
      * "official"       - a published Indian standard/policy value, used as stated.
      * "official-proxy" - an official target applied to a proxy metric (explained in notes).
      * "illustrative"   - NOT an official Indian target; a configurable placeholder.
    """

    # CPCB National AQI: 'Satisfactory' band is 51-100. Using <=100 as the target.
    aqi_target: float = Field(100.0, gt=0, le=500)
    # NCAP (MoEFCC, 2019; revised 2022): 40% reduction in PM10 by 2025-26 vs 2017 base.
    # Applied here to simulated AQI relative to the simulation start (proxy).
    ncap_reduction_pct: float = Field(40.0, gt=0, le=100)
    # National Forest Policy 1988: 33% of geographical area under forest/tree cover.
    green_cover_target_pct: float = Field(33.0, gt=0, le=100)
    # NOT an official Indian target. Paris-style 1.5 C figure used as a placeholder for
    # "acceptable local warming over the simulated period".
    max_warming_c: float = Field(1.5, gt=0, le=10)

    weight_aqi: float = Field(0.30, ge=0)
    weight_ncap: float = Field(0.20, ge=0)
    weight_green_cover: float = Field(0.25, ge=0)
    weight_temperature: float = Field(0.25, ge=0)


DEFAULT_COEFFICIENTS = ModelCoefficients()
DEFAULT_TARGETS = TargetConfig()

ASSUMPTIONS = [
    "Deterministic annual-timestep model; no stochastic weather, seasonality or extreme events.",
    "Baseline scenario = linear temperature trend, compound AQI trend and linear NDVI trend supplied by the user.",
    "Intervention effects follow saturating (green cover, farm ponds) or linear (traffic) dose-response curves with illustrative coefficients.",
    "Effects ramp up linearly to full strength over a per-intervention maturity period (trees: 5 y, ponds: 2 y, traffic: 1 y by default).",
    "Temperature and NDVI effects are additive; AQI reductions combine multiplicatively across interventions.",
    "No feedbacks between variables (e.g. NDVI does not further change AQI), no spatial dispersion, no transboundary pollution.",
    "Annual-mean AQI is treated as a single number on the CPCB 0-500 scale; real AQI is driven by daily/seasonal peaks.",
    "Traffic AQI effect = vehicular share of ambient PM x compliance x % reduction; the 25% share is a placeholder (city-specific source apportionment studies differ widely).",
    "Green-cover score uses baseline_green_cover_pct + newly greened % at maturity; farm ponds do not add forest/tree cover.",
    "The NCAP target is defined on PM10 vs a 2017 base year; here it is applied to simulated AQI relative to the simulation start year as a proxy only.",
    "max_warming_c is an illustrative placeholder, not an official Indian target.",
    "Coefficients are uncalibrated. Replace them via the `coefficients` request field with locally validated values.",
]

TARGET_SOURCES = {
    "aqi_cpcb_satisfactory": "CPCB National Air Quality Index (AQI bands: Good 0-50, Satisfactory 51-100, Moderate 101-200, Poor 201-300, Very Poor 301-400, Severe 401-500).",
    "ncap_reduction": "National Clean Air Programme (MoEFCC): 40% PM10 reduction by 2025-26 vs 2017 (revised 2022). Applied to AQI as a proxy.",
    "green_cover_33": "National Forest Policy 1988: 33% of geographical area under forest/tree cover.",
    "warming_limit": "Illustrative placeholder (not an official Indian target).",
}

CPCB_AQI_BANDS = [
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (500, "Severe"),
]


def aqi_category(aqi: float) -> str:
    for upper, name in CPCB_AQI_BANDS:
        if aqi <= upper:
            return name
    return "Severe"
