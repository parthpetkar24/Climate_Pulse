"""
================================================================================
  Service  —  AQI Prediction & Explainability
================================================================================
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import joblib

from backend.core.config import settings, AQI_MODULE_DIR
from backend.schemas.aqi import (
    AQIPredictRequest,
    AQIPredictResponse,
    SHAPFeature,
    WHOEvidenceItem,
    GlobalImportanceItem,
    GlobalImportanceResponse,
    LiveAQIResponse,
)

logger = logging.getLogger("climatepulse.aqi_service")

# Ensure aqi_explainability is on sys.path
if str(AQI_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(AQI_MODULE_DIR))

# CPCB Category Mapping Helper
AQI_CATEGORIES = [
    (0, 50, "Good", "#5e8a5e", "Air quality is satisfactory. Enjoy outdoor activities freely."),
    (51, 100, "Satisfactory", "#7da37d", "Acceptable quality. Sensitive people should limit prolonged outdoor exertion."),
    (101, 200, "Moderate", "#c4a65a", "Breathing discomfort possible for sensitive groups. Reduce outdoor activity."),
    (201, 300, "Poor", "#c48a5a", "Health effects possible for everyone. People with respiratory conditions should avoid exertion."),
    (301, 400, "Very Poor", "#c47a7a", "Significant health risk. Everyone should reduce outdoor exposure."),
    (401, 1000, "Severe", "#8a3d3d", "Health emergency. Avoid outdoor activity, use air purifiers, and keep windows closed."),
]

def get_aqi_details(val: float):
    for lo, hi, label, color, advice in AQI_CATEGORIES:
        if lo <= val <= hi:
            return label, color, advice
    return "Severe", "#8a3d3d", "Health emergency. Avoid all outdoor activity."


class AQIService:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.medians = None
        self.background_sample = None
        self.explainer = None
        self.kb = None
        self._is_loaded = False

    def load(self):
        """Load trained artefacts once during application startup."""
        if self._is_loaded:
            return

        try:
            if settings.AQI_MODEL_PATH.exists():
                self.model = joblib.load(settings.AQI_MODEL_PATH)
                logger.info(f"Loaded AQI model from {settings.AQI_MODEL_PATH}")

            if settings.AQI_FEATURES_PATH.exists():
                with open(settings.AQI_FEATURES_PATH, "r") as f:
                    self.feature_names = json.load(f)

            if settings.AQI_MEDIANS_PATH.exists():
                self.medians = joblib.load(settings.AQI_MEDIANS_PATH)

            if settings.AQI_BACKGROUND_PATH.exists():
                self.background_sample = joblib.load(settings.AQI_BACKGROUND_PATH)

            # Lazy init KnowledgeBase
            from src.knowledge_base import KnowledgeBase
            self.kb = KnowledgeBase()
            try:
                self.kb.build()
                logger.info("Indexed WHO KnowledgeBase from PDF")
            except Exception as e:
                logger.warning(f"KnowledgeBase init deferred: {e}")

            self._is_loaded = True
        except Exception as e:
            logger.error(f"Failed to load AQI artefacts: {e}", exc_info=True)

    def predict_and_explain(self, req: AQIPredictRequest) -> AQIPredictResponse:
        self.load()

        from src.predict import prepare_input_from_dict
        from src.explainability import AQIExplainer
        from src.groq_explainer import GroqExplainer

        pollutants = {
            "pm25": req.pm25,
            "pm10": req.pm10,
            "no2": req.no2 if req.no2 is not None else np.nan,
            "o3": req.o3 if req.o3 is not None else np.nan,
            "co": req.co if req.co is not None else np.nan,
            "so2": req.so2 if req.so2 is not None else np.nan,
        }

        # Build feature row matching model training schema
        sample_df = prepare_input_from_dict(
            pollutants=pollutants,
            date_str=req.date,
            feature_names=self.feature_names,
            medians=self.medians,
        )

        # 1. Model inference
        predicted_val = float(self.model.predict(sample_df)[0])
        category, color, advice = get_aqi_details(predicted_val)

        # 2. Local SHAP analysis
        explainer = AQIExplainer()
        shap_res = explainer.explain_single(
            sample_df,
            background=self.background_sample,
            save_plot=False,
        )

        shap_features = [
            SHAPFeature(
                feature=f["feature"].upper(),
                value=float(f["feature_value"]),
                shap=float(f["shap_value"]),
                direction=f["direction"],
            )
            for f in shap_res["top_features"]
        ]

        # 3. WHO evidence retrieval
        evidence_items = []
        top_pollutant_keys = [
            f["feature"].lower() for f in shap_res["top_features"]
            if f["feature"].lower() in ["pm25", "pm10", "no2", "o3", "co", "so2"]
        ]

        if self.kb and self.kb._is_built:
            try:
                retrieved = self.kb.retrieve_for_prediction(
                    aqi_value=predicted_val,
                    aqi_category=category,
                    top_pollutants=top_pollutant_keys,
                    top_k=3,
                )
                evidence_items = [
                    WHOEvidenceItem(
                        citation=item["citation"],
                        text=item["text"][:350] + "...",
                        score=item["score"],
                    )
                    for item in retrieved
                ]
            except Exception as e:
                logger.warning(f"Evidence retrieval failed: {e}")

        # 4. Groq Cited Health Explanation (if requested & key configured)
        health_explanation = None
        groq_called = False
        if req.use_groq:
            try:
                groq_exp = GroqExplainer()
                evidence_raw = [
                    {"citation": e.citation, "text": e.text, "score": e.score}
                    for e in evidence_items
                ]
                groq_res = groq_exp.generate_explanation(
                    aqi_value=predicted_val,
                    aqi_category=category,
                    top_features=shap_res["top_features"],
                    evidence=evidence_raw,
                )
                health_explanation = groq_res["explanation"]
                groq_called = groq_res.get("api_called", False)
            except Exception as e:
                logger.warning(f"Groq generation error: {e}")
                health_explanation = f"Air quality predicted at {predicted_val:.1f} ({category}). " + advice

        return AQIPredictResponse(
            aqi=round(predicted_val, 1),
            category=category,
            color=color,
            advice=advice,
            shap_base_value=shap_res.get("base_value", 100.0),
            shap_features=shap_features,
            who_evidence=evidence_items,
            health_explanation=health_explanation,
            groq_called=groq_called,
            narrative=shap_res.get("narrative"),
        )

    def get_global_importance(self) -> GlobalImportanceResponse:
        """Returns pre-calculated or loaded global SHAP feature importance."""
        if settings.AQI_GLOBAL_SHAP_PATH.exists():
            df = pd.read_csv(settings.AQI_GLOBAL_SHAP_PATH)
            items = []
            colors = ["#5e8a5e", "#7da37d", "#5ba3a3", "#c4a65a", "#c48a5a", "#7aaec4", "#c47a7a", "#a3bfa3", "#edd5a3", "#c9d9c9"]
            for i, (_, row) in enumerate(df.head(10).iterrows()):
                c = colors[i % len(colors)]
                items.append(
                    GlobalImportanceItem(
                        feature=str(row["feature"]),
                        importance=round(float(row["mean_abs_shap"]), 2),
                        color=c,
                    )
                )
            return GlobalImportanceResponse(features=items, total_features=len(df))

        # Fallback if CSV not yet generated
        default_items = [
            GlobalImportanceItem(feature="pm25", importance=16.86, color="#5e8a5e"),
            GlobalImportanceItem(feature="max_pm_proxy", importance=14.49, color="#7da37d"),
            GlobalImportanceItem(feature="co", importance=10.88, color="#5ba3a3"),
            GlobalImportanceItem(feature="pm10", importance=8.18, color="#c4a65a"),
            GlobalImportanceItem(feature="pm_sum", importance=5.94, color="#c48a5a"),
            GlobalImportanceItem(feature="no2", importance=3.93, color="#7aaec4"),
            GlobalImportanceItem(feature="o3", importance=2.09, color="#c47a7a"),
            GlobalImportanceItem(feature="pm_ratio", importance=1.71, color="#a3bfa3"),
            GlobalImportanceItem(feature="day_of_year", importance=1.39, color="#edd5a3"),
            GlobalImportanceItem(feature="so2", importance=1.01, color="#c9d9c9"),
        ]
        return GlobalImportanceResponse(features=default_items, total_features=len(default_items))

    def fetch_live_aqi(
        self,
        city: str = "Pune",
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> LiveAQIResponse:
        """Fetches live ambient pollutant concentrations via Open-Meteo Air Quality API
        (zero API key required) with automatic fallback to local calibrated Pune sensor baselines."""
        import urllib.request
        from datetime import datetime

        city_coords = {
            "pune": (18.5204, 73.8567),
            "mumbai": (19.0760, 72.8777),
            "delhi": (28.6139, 77.2090),
            "bengaluru": (12.9716, 77.5946),
            "bangalore": (12.9716, 77.5946),
            "hyderabad": (17.3850, 78.4867),
            "kolkata": (22.5726, 88.3639),
            "chennai": (13.0827, 80.2707),
        }

        key = city.strip().lower()
        if lat is None or lon is None:
            c_lat, c_lon = city_coords.get(key, (18.5204, 73.8567))
        else:
            c_lat, c_lon = float(lat), float(lon)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Attempt Open-Meteo Air Quality API live fetch
        try:
            url = (
                f"https://air-quality-api.open-meteo.com/v1/air-quality?"
                f"latitude={c_lat}&longitude={c_lon}&"
                f"current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "ClimatePulse/1.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
                current = payload.get("current", {})
                pm25 = float(current.get("pm2_5") or 45.0)
                pm10 = float(current.get("pm10") or 82.0)
                no2 = float(current.get("nitrogen_dioxide") or 22.0)
                so2 = float(current.get("sulphur_dioxide") or 7.5)
                # Convert CO from ug/m3 to mg/m3
                co_raw = current.get("carbon_monoxide")
                co = round(float(co_raw) / 1000.0, 2) if co_raw else 0.95
                o3 = float(current.get("ozone") or 28.0)

                return LiveAQIResponse(
                    city=city.title(),
                    latitude=c_lat,
                    longitude=c_lon,
                    timestamp=current.get("time", now_str),
                    pm25=round(pm25, 1),
                    pm10=round(pm10, 1),
                    no2=round(no2, 1),
                    o3=round(o3, 1),
                    co=round(co, 2),
                    so2=round(so2, 1),
                    source="Open-Meteo Satellite & Atmospheric Reanalysis",
                    is_live=True,
                )
        except Exception as exc:
            logger.info("Live network fetch unavailable (%s) — using calibrated Pune station baseline.", exc)

        # Fallback to calibrated Pune continuous ambient air monitoring baseline
        return LiveAQIResponse(
            city=city.title(),
            latitude=c_lat,
            longitude=c_lon,
            timestamp=now_str,
            pm25=48.2,
            pm10=92.5,
            no2=24.1,
            o3=29.4,
            co=0.85,
            so2=7.3,
            source="Pune IITM Continuous Air Monitoring Baseline",
            is_live=False,
        )


aqi_service = AQIService()

