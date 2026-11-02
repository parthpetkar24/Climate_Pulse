# Climate Risk & Adaptation Simulator (MVP)

Simplified, transparent scenario model: pick interventions (farm ponds, traffic restrictions, green cover),
give baseline conditions + duration, get baseline vs intervention trajectories for temperature, NDVI and AQI,
scored against Indian targets.

> **Not a validated forecast.** Coefficients are illustrative and uncalibrated. No RL / ML is used because
> there is no training dataset in scope; a fitted model would only disguise invented numbers.

## Run
```bash
pip install -r requirements-simulator.txt
python -m pytest -q                 # 29 tests
uvicorn demo_app:app --reload       # http://127.0.0.1:8000/docs
```

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/climate-simulator/defaults` | ranges, default coefficients/targets, sources, assumptions, example request |
| POST | `/api/climate-simulator/simulate` | run simulation |

## Intervention units
| type | intensity unit | range |
|---|---|---|
| farm_ponds | ponds per km² | 0.1–20 |
| traffic_restrictions | % reduction in vehicle activity | 1–60 |
| green_cover | percentage points of area newly greened | 0.5–30 |

## Response (for charts)
`years`, `baseline[]`, `intervention[]`, `intervention_low[]`/`intervention_high[]` (uncertainty band),
`delta[]`, `summary[]` (per-metric change), `attribution[]` (each intervention alone),
`scoring` (per-target progress + overall 0–100), `explanations[]`, `meta` (assumptions, coefficients, disclaimer).

## Targets used
| id | kind | basis |
|---|---|---|
| aqi_cpcb_satisfactory | official | CPCB AQI "Satisfactory" band ≤ 100 |
| ncap_reduction | official-proxy | NCAP 40% PM10 reduction, applied to AQI vs start year |
| green_cover_33 | official | National Forest Policy 1988, 33% forest/tree cover |
| warming_limit | **illustrative** | placeholder 1.5 °C, NOT an official Indian target |

Please double-check the official figures against current MoEFCC/CPCB publications before relying on them.

## Limitations
Annual timestep, deterministic, no feedbacks/spatial/seasonal effects, annual-mean AQI, local area scored
against national goals, default baseline green cover (25%) is only an approximate national-scale figure.
See `meta.assumptions` in every response.
