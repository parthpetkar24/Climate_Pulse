"""
================================================================================
  Service  —  Carbon Footprint Calculator & ML Risk Clustering
================================================================================
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from backend.schemas.carbon import (
    CarbonCalculateRequest,
    CarbonCalculateResponse,
    CarbonRecordOutput,
)

logger = logging.getLogger("climatepulse.carbon_service")

# Global emission factors (kg CO2e or tCO2/MWh per unit)
EMISSION_FACTORS = {
    "electricity_india": 0.727,   # tCO2/MWh
    "electricity_uk": 0.13096,    # kg CO2e/kWh
    "anthracite_coal": 2602.0,    # kg CO2/short ton
    "bituminous_coal": 2325.0,    # kg CO2/short ton
    "natural_gas_scf": 0.05444,   # kg CO2/scf
    "diesel_fuel_oil": 10.21,     # kg CO2/gallon
    "passenger_car": 0.297,       # kg CO2/mile
    "light_duty_truck": 0.394,    # kg CO2/mile
    "air_travel_short": 0.207,    # kg CO2/mile
    "air_travel_long": 0.163,     # kg CO2/mile
}


class CarbonService:
    def calculate(self, req: CarbonCalculateRequest) -> CarbonCalculateResponse:
        records_raw = []

        for item in req.activities:
            cat = item.category.strip().lower()
            factor = EMISSION_FACTORS.get(cat, 0.5)
            consumption = float(item.consumption)

            # Compute metric tons of CO2e
            if cat == "electricity_india":
                total_mt = consumption * factor
            elif cat in ["anthracite_coal", "bituminous_coal"]:
                total_mt = (consumption * factor) / 1000.0
            else:
                total_mt = (consumption * factor) / 1000.0

            source_name = item.label or cat.replace("_", " ").title()
            records_raw.append({
                "activity_source": source_name,
                "consumption": consumption,
                "factor": factor,
                "footprint_mt": round(total_mt, 4),
            })

        if not records_raw:
            # Fallback default items if empty
            records_raw = [
                {"activity_source": "Grid Electricity (Pune)", "consumption": 1655.0, "factor": 0.727, "footprint_mt": 1203.18},
                {"activity_source": "Industrial Coal (Bituminous)", "consumption": 1200.0, "factor": 2325.0, "footprint_mt": 2790.0},
                {"activity_source": "Fleet Diesel Logistics", "consumption": 4500.0, "factor": 10.21, "footprint_mt": 45.945},
                {"activity_source": "Corporate Air Travel", "consumption": 15000.0, "factor": 0.163, "footprint_mt": 2.445},
            ]

        df = pd.DataFrame(records_raw)

        # Unsupervised ML KMeans Clustering
        X = df[["factor", "footprint_mt"]].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = min(3, len(df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["cluster_id"] = kmeans.fit_predict(X_scaled)

        # Map cluster means to ranked risk labels
        cluster_means = df.groupby("cluster_id")["footprint_mt"].mean().sort_values()
        rankings = [
            ("Low Emission Profile", "low", "#5e8a5e"),
            ("Medium Emission Profile", "medium", "#c4a65a"),
            ("High Emission Profile (Critical Red)", "critical", "#8a3d3d"),
        ]

        intensity_map = {}
        for i, cluster_idx in enumerate(cluster_means.index):
            intensity_map[cluster_idx] = rankings[min(i, len(rankings) - 1)]

        out_records = []
        summary = {"low": 0.0, "medium": 0.0, "critical": 0.0}

        for _, row in df.iterrows():
            c_label, r_level, color = intensity_map[row["cluster_id"]]
            summary[r_level] += row["footprint_mt"]
            out_records.append(
                CarbonRecordOutput(
                    activity_source=row["activity_source"],
                    consumption=row["consumption"],
                    factor=row["factor"],
                    footprint_mt=round(row["footprint_mt"], 3),
                    cluster_id=int(row["cluster_id"]),
                    classification=c_label,
                    risk_level=r_level,
                    color=color,
                )
            )

        total_mt = float(df["footprint_mt"].sum())

        recommendations = []
        if summary["critical"] > 0:
            recommendations.append("Priority 1: Transition high-impact fossil fuel or coal units to renewable PPA or natural gas.")
        if summary["medium"] > 0:
            recommendations.append("Priority 2: Implement route optimization for logistics and switch light commercial fleets to EV.")
        recommendations.append("Priority 3: Establish annual energy efficiency audits to reduce baseline electricity intensity.")

        return CarbonCalculateResponse(
            records=out_records,
            total_footprint_mt=round(total_mt, 3),
            summary_by_risk={k: round(v, 3) for k, v in summary.items()},
            recommendations=recommendations,
        )


carbon_service = CarbonService()

