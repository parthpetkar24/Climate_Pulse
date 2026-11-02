"""Scoring against Indian climate/air-quality targets + plain-language explanations.

Only targets with a published source are marked "official"/"official-proxy".
The temperature limit is an illustrative placeholder and is labelled as such.
"""
from __future__ import annotations

from typing import List

from .config import TARGET_SOURCES, TargetConfig, aqi_category


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def aqi_progress(initial: float, final: float, target: float) -> float:
    """1.0 once final <= target; otherwise how much of the gap initial->target is closed."""
    if final <= target:
        return 1.0
    if initial <= target:  # was compliant, now is not
        return _clamp01(target / final)
    return _clamp01(1.0 - (final - target) / (initial - target))


def reduction_progress(initial: float, final: float, target_pct: float) -> float:
    if initial <= 0:
        return 1.0
    reduction = (initial - final) / initial * 100.0
    return _clamp01(reduction / target_pct)


def level_progress(value: float, target: float) -> float:
    return _clamp01(value / target)


def warming_progress(warming: float, limit: float) -> float:
    if warming <= limit:
        return 1.0
    return _clamp01(limit / warming)


def score_scenarios(base, scen, t: TargetConfig):
    from .schemas import Scoring, TargetResult  # local import avoids cycle

    b0, b1, s1 = base[0], base[-1], scen[-1]
    b_warm, s_warm = b1.temperature_c - b0.temperature_c, s1.temperature_c - b0.temperature_c
    b_red = (b0.aqi - b1.aqi) / b0.aqi * 100.0 if b0.aqi > 0 else 0.0
    s_red = (b0.aqi - s1.aqi) / b0.aqi * 100.0 if b0.aqi > 0 else 0.0

    total_w = t.weight_aqi + t.weight_ncap + t.weight_green_cover + t.weight_temperature
    w = lambda x: round(x / total_w, 4)  # noqa: E731

    rows = [
        TargetResult(
            id="aqi_cpcb_satisfactory",
            name="Final-year AQI within CPCB 'Satisfactory' band",
            metric="aqi",
            kind="official",
            source=TARGET_SOURCES["aqi_cpcb_satisfactory"],
            threshold=t.aqi_target,
            unit="AQI (lower is better)",
            baseline_value=b1.aqi,
            intervention_value=s1.aqi,
            baseline_progress=round(aqi_progress(b0.aqi, b1.aqi, t.aqi_target), 4),
            intervention_progress=round(aqi_progress(b0.aqi, s1.aqi, t.aqi_target), 4),
            baseline_met=b1.aqi <= t.aqi_target,
            intervention_met=s1.aqi <= t.aqi_target,
            weight=w(t.weight_aqi),
            notes="Annual-mean AQI is compared with the CPCB band limit; real compliance is judged on daily values.",
        ),
        TargetResult(
            id="ncap_reduction",
            name="AQI reduction vs simulation start (NCAP-style)",
            metric="aqi",
            kind="official-proxy",
            source=TARGET_SOURCES["ncap_reduction"],
            threshold=t.ncap_reduction_pct,
            unit="% reduction (higher is better)",
            baseline_value=round(b_red, 2),
            intervention_value=round(s_red, 2),
            baseline_progress=round(reduction_progress(b0.aqi, b1.aqi, t.ncap_reduction_pct), 4),
            intervention_progress=round(reduction_progress(b0.aqi, s1.aqi, t.ncap_reduction_pct), 4),
            baseline_met=b_red >= t.ncap_reduction_pct,
            intervention_met=s_red >= t.ncap_reduction_pct,
            weight=w(t.weight_ncap),
            notes="NCAP targets PM10 vs a 2017 base year; this applies the % figure to simulated AQI vs the start year as a proxy.",
        ),
        TargetResult(
            id="green_cover_33",
            name="Forest/tree cover vs 33% national goal",
            metric="green_cover_pct",
            kind="official",
            source=TARGET_SOURCES["green_cover_33"],
            threshold=t.green_cover_target_pct,
            unit="% of area (higher is better)",
            baseline_value=b1.green_cover_pct,
            intervention_value=s1.green_cover_pct,
            baseline_progress=round(level_progress(b1.green_cover_pct, t.green_cover_target_pct), 4),
            intervention_progress=round(level_progress(s1.green_cover_pct, t.green_cover_target_pct), 4),
            baseline_met=b1.green_cover_pct >= t.green_cover_target_pct,
            intervention_met=s1.green_cover_pct >= t.green_cover_target_pct,
            weight=w(t.weight_green_cover),
            notes="The national figure is a country-wide goal; applying it to a local area is a simplification.",
        ),
        TargetResult(
            id="warming_limit",
            name="Local warming over the period within configured limit",
            metric="temperature_c",
            kind="illustrative",
            source=TARGET_SOURCES["warming_limit"],
            threshold=t.max_warming_c,
            unit="deg C of warming vs start (lower is better)",
            baseline_value=round(b_warm, 3),
            intervention_value=round(s_warm, 3),
            baseline_progress=round(warming_progress(b_warm, t.max_warming_c), 4),
            intervention_progress=round(warming_progress(s_warm, t.max_warming_c), 4),
            baseline_met=b_warm <= t.max_warming_c,
            intervention_met=s_warm <= t.max_warming_c,
            weight=w(t.weight_temperature),
            notes="Not an official Indian target. Adjust `targets.max_warming_c` to your own threshold.",
        ),
    ]

    base_score = sum(r.baseline_progress * r.weight for r in rows) * 100.0
    scen_score = sum(r.intervention_progress * r.weight for r in rows) * 100.0
    return Scoring(
        baseline_score=round(base_score, 2),
        intervention_score=round(scen_score, 2),
        score_improvement=round(scen_score - base_score, 2),
        targets=rows,
        method=(
            "Score = 100 x weighted mean of per-target progress (0-1). Progress is 1.0 when the target is met "
            "and scales linearly toward it otherwise. Weights are normalised to sum to 1."
        ),
    )


def build_explanations(req, base, scen, summary, scoring, attribution) -> List[str]:
    by = {s.metric: s for s in summary}
    n = req.duration_years
    out: List[str] = []

    a, t, v = by["aqi"], by["temperature_c"], by["ndvi"]
    out.append(
        f"Over {n} years the selected interventions change final-year AQI from {a.baseline_final:.1f} "
        f"({aqi_category(a.baseline_final)}) to {a.intervention_final:.1f} ({aqi_category(a.intervention_final)}), "
        f"a change of {a.change_vs_baseline:+.1f} AQI points."
    )
    out.append(
        f"Final-year temperature differs by {t.change_vs_baseline:+.2f} deg C and NDVI by {v.change_vs_baseline:+.3f} "
        f"versus the no-intervention baseline."
    )
    if attribution:
        top = max(attribution, key=lambda x: abs(x.final_year_effect["aqi"]))
        if abs(top.final_year_effect["aqi"]) > 0:
            out.append(
                f"The largest standalone AQI effect comes from '{top.intervention.value}' "
                f"({top.final_year_effect['aqi']:+.1f} AQI points in the final year when applied alone)."
            )
    met = [r.name for r in scoring.targets if r.intervention_met and not r.baseline_met]
    if met:
        out.append("Targets newly met because of the interventions: " + "; ".join(met) + ".")
    unmet = [r.name for r in scoring.targets if not r.intervention_met]
    if unmet:
        out.append("Targets still not met: " + "; ".join(unmet) + ".")
    out.append(
        f"Overall target score moves from {scoring.baseline_score:.1f} to {scoring.intervention_score:.1f} "
        f"({scoring.score_improvement:+.1f})."
    )
    out.append(
        "Effects ramp up gradually (trees take longest to mature), so year-wise gains are smaller early on."
    )
    out.append(
        "CAUTION: this is a simplified model with uncalibrated illustrative coefficients, not a validated forecast. "
        "The low/high bands only scale intervention effect sizes (0.5x-1.5x by default) and do not capture all uncertainty."
    )
    return out
