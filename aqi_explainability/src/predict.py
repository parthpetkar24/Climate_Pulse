"""
================================================================================
  predict.py  —  Unified AQI Prediction + Explainability Pipeline
  ClimatePulse | PCCOE Hackathon  —  MVP 1
================================================================================

Combines all four modules into a single prediction pipeline:
  1. Load model and preprocess input.
  2. Predict AQI.
  3. Generate SHAP explanation (local).
  4. Retrieve WHO evidence from the knowledge base.
  5. Generate a cited health explanation via Groq (if API key available).

Usage
-----
    # From aqi_explainability/ directory:
    python -m src.predict

    # Or run a prediction on a specific date from the dataset:
    python -m src.predict --date 2025-06-15
================================================================================
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

log = logging.getLogger("aqi_explainability.predict")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent         # aqi_explainability/
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    """Load the trained model, feature names, and medians."""
    model_path    = MODEL_DIR / "best_aqi_model.joblib"
    features_path = MODEL_DIR / "feature_names.json"
    medians_path  = MODEL_DIR / "feature_medians.joblib"

    for p in [model_path, features_path, medians_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run train_model.py first to generate model artefacts."
            )

    model = joblib.load(model_path)
    with open(features_path) as f:
        feature_names = json.load(f)
    medians = joblib.load(medians_path)

    log.info("Model loaded from %s", model_path)
    return model, feature_names, medians


def prepare_input_from_dict(
    pollutants: dict,
    date_str: str | None = None,
    feature_names: list | None = None,
    medians: dict | None = None,
) -> pd.DataFrame:
    """
    Construct a 1-row DataFrame formatted with all 22 required features
    from user-provided pollutant concentrations and an optional date.

    Parameters
    ----------
    pollutants : dict
        Dict with keys like 'pm25', 'pm10', 'no2', 'o3', 'co', 'so2'.
    date_str : str, optional
        Date in 'YYYY-MM-DD' format (default: current date).
    feature_names : list, optional
        Target list of features. If None, loaded from disk.
    medians : dict, optional
        Median values for imputation. If None, loaded from disk.

    Returns
    -------
    pd.DataFrame
        1-row DataFrame with all required features in the correct order.
    """
    if feature_names is None or medians is None:
        _, feature_names, medians = load_model()

    ts = pd.to_datetime(date_str) if date_str else pd.Timestamp.now()

    season_map = {
        1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON", 12: "DJF",
    }
    month = ts.month
    day_of_week = ts.dayofweek
    day_of_year = ts.dayofyear
    quarter = ts.quarter
    season = season_map.get(month, "DJF")

    pm25_v = float(pollutants.get("pm25", np.nan)) if not pd.isna(pollutants.get("pm25", np.nan)) else np.nan
    pm10_v = float(pollutants.get("pm10", np.nan)) if not pd.isna(pollutants.get("pm10", np.nan)) else np.nan

    row = {
        "pm25": pm25_v,
        "pm10": pm10_v,
        "no2":  pollutants.get("no2",  np.nan),
        "o3":   pollutants.get("o3",   np.nan),
        "co":   pollutants.get("co",   np.nan),
        "so2":  pollutants.get("so2",  np.nan),
        "pm_ratio": pm25_v / (pm10_v + 1.0) if not pd.isna(pm25_v) and not pd.isna(pm10_v) else np.nan,
        "pm_sum":   pm25_v + pm10_v if not pd.isna(pm25_v) and not pd.isna(pm10_v) else np.nan,
        "max_pm_proxy": max(pm25_v * 1.67, pm10_v) if not pd.isna(pm25_v) and not pd.isna(pm10_v) else np.nan,
        "year": ts.year,
        "month": month,
        "day_of_week": day_of_week,
        "day_of_year": day_of_year,
        "quarter": quarter,
        "is_weekend": 1 if day_of_week >= 5 else 0,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "dow_sin":   np.sin(2 * np.pi * day_of_week / 7),
        "dow_cos":   np.cos(2 * np.pi * day_of_week / 7),
        "doy_sin":   np.sin(2 * np.pi * day_of_year / 365),
        "doy_cos":   np.cos(2 * np.pi * day_of_year / 365),
        "season_DJF": 1 if season == "DJF" else 0,
        "season_JJA": 1 if season == "JJA" else 0,
        "season_MAM": 1 if season == "MAM" else 0,
        "season_SON": 1 if season == "SON" else 0,
    }

    df = pd.DataFrame([row])
    for col in feature_names:
        if col not in df.columns:
            df[col] = medians.get(col, 0.0)

    df = df[feature_names].fillna(medians).fillna(0.0)
    return df


def predict_and_explain(
    sample: pd.DataFrame,
    model=None,
    feature_names: list = None,
    medians: dict = None,
    use_groq: bool = True,
    background_data: pd.DataFrame = None,
) -> dict:
    """
    Full prediction + explanation pipeline for one sample row.

    Parameters
    ----------
    sample : pd.DataFrame
        Single-row DataFrame with the feature columns.
    model, feature_names, medians : optional
        Pre-loaded artefacts.  If None, loaded from disk.
    use_groq : bool
        Whether to attempt Groq explanation generation.
    background_data : pd.DataFrame, optional
        Background samples for SHAP explainer.

    Returns
    -------
    dict with prediction, SHAP explanation, evidence, and LLM explanation.
    """
    # ── 1. Load model if needed ──────────────────────────────────────────
    if model is None:
        model, feature_names, medians = load_model()

    # ── 2. Prepare input ─────────────────────────────────────────────────
    X = sample[feature_names].copy()
    X = X.fillna(medians)

    # ── 3. Predict ───────────────────────────────────────────────────────
    prediction = float(model.predict(X)[0])
    log.info("Predicted AQI: %.2f", prediction)

    # ── 4. SHAP explanation ──────────────────────────────────────────────
    from src.explainability import AQIExplainer, aqi_category

    category = aqi_category(prediction)
    explainer = AQIExplainer()

    if background_data is not None:
        bg = background_data
    elif (MODEL_DIR / "background_sample.joblib").exists():
        bg = joblib.load(MODEL_DIR / "background_sample.joblib")
    else:
        # Fallback to test set sample
        from src.data_preprocessing import load_and_prepare
        data = load_and_prepare()
        bg = data["X_test"].sample(n=min(100, len(data["X_test"])), random_state=42)

    shap_result = explainer.explain_single(
        X,
        background=bg,
        save_plot=True,
        plot_prefix="prediction",
    )

    # Identify top pollutant features (for WHO retrieval)
    pollutant_features = ["pm25", "pm10", "no2", "o3", "co", "so2"]
    top_pollutants = [
        f["feature"]
        for f in shap_result["top_features"]
        if f["feature"] in pollutant_features
    ]

    # ── 5. WHO evidence retrieval ────────────────────────────────────────
    from src.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    try:
        kb.build()
        evidence = kb.retrieve_for_prediction(
            aqi_value=prediction,
            aqi_category=category,
            top_pollutants=top_pollutants,
            top_k=5,
        )
        log.info("Retrieved %d WHO evidence chunks", len(evidence))
    except (FileNotFoundError, ImportError) as e:
        log.warning("WHO evidence retrieval failed: %s", e)
        evidence = []

    # ── 6. Groq LLM explanation ──────────────────────────────────────────
    groq_result = None
    if use_groq:
        from src.groq_explainer import GroqExplainer

        groq = GroqExplainer()
        groq_result = groq.generate_explanation(
            aqi_value=prediction,
            aqi_category=category,
            top_features=shap_result["top_features"],
            evidence=evidence,
        )
        log.info("Groq explanation generated (api_called=%s)", groq_result["api_called"])

    # ── 7. Assemble result ───────────────────────────────────────────────
    result = {
        "prediction": {
            "aqi_value":    round(prediction, 2),
            "aqi_category": category,
        },
        "shap_explanation": {
            "base_value":    shap_result["base_value"],
            "top_features":  shap_result["top_features"],
            "narrative":     shap_result["narrative"],
        },
        "who_evidence": [
            {"citation": e["citation"], "score": e["score"], "text": e["text"][:300]}
            for e in evidence
        ],
        "health_explanation": groq_result["explanation"] if groq_result else None,
        "groq_api_called": groq_result["api_called"] if groq_result else False,
    }

    # Save result
    result_path = OUTPUT_DIR / "latest_prediction.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info("Full result saved → %s", result_path)

    return result


def _print_result(result: dict):
    """Pretty-print a prediction result to stdout."""
    print("\n" + "=" * 70)
    print("  AQI PREDICTION + EXPLAINABILITY RESULT")
    print("=" * 70)

    pred = result["prediction"]
    print(f"\n  Predicted AQI : {pred['aqi_value']}")
    print(f"  AQI Category  : {pred['aqi_category']}")

    print("\n" + result["shap_explanation"]["narrative"])

    if result["who_evidence"]:
        print("\n--- WHO Evidence ---")
        for i, e in enumerate(result["who_evidence"], 1):
            print(f"  [{i}] {e['citation']} (score: {e['score']})")
            print(f"      {e['text'][:150]}...")

    if result["health_explanation"]:
        print("\n--- Health Explanation ---")
        print(result["health_explanation"])

    print("\n" + "=" * 70)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AQI Prediction + Explainability")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Predict for a specific date (YYYY-MM-DD) from the dataset.",
    )
    parser.add_argument(
        "--no-groq", action="store_true",
        help="Skip Groq LLM explanation.",
    )
    parser.add_argument(
        "--sample-index", type=int, default=0,
        help="Index of the test sample to use (default: 0).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load data
    from src.data_preprocessing import load_and_prepare
    data = load_and_prepare()

    if args.date:
        # Find the closest matching row in the full dataset
        from src.data_preprocessing import DATASET_PATH
        full_df = pd.read_csv(DATASET_PATH)
        full_df["date"] = pd.to_datetime(full_df["date"])
        target_date = pd.to_datetime(args.date)

        match = full_df[full_df["date"] == target_date]
        if match.empty:
            print(f"No data found for date {args.date}")
            sys.exit(1)

        # Re-preprocess just this row
        from src.data_preprocessing import _add_temporal_features, POLLUTANT_COLS, LEAKAGE_COLS
        row = match.iloc[[0]].copy()
        for col in LEAKAGE_COLS:
            if col in row.columns:
                row = row.drop(columns=[col])
        for col in ["city", "station_name", "station_id"]:
            if col in row.columns:
                row = row.drop(columns=[col])
        # Coerce pollutant columns to numeric
        for col in POLLUTANT_COLS:
            if col in row.columns:
                row[col] = pd.to_numeric(row[col], errors="coerce")
                row[col] = row[col].clip(lower=0)

        # Domain interaction features (must match data_preprocessing.py)
        row["pm_ratio"] = row["pm25"] / (row["pm10"] + 1.0)
        row["pm_sum"] = row["pm25"] + row["pm10"]
        row["max_pm_proxy"] = np.maximum(row["pm25"] * 1.67, row["pm10"])

        row = _add_temporal_features(row)
        row = pd.get_dummies(row, columns=["season"], prefix="season", drop_first=False)
        for c in row.columns:
            if row[c].dtype == bool:
                row[c] = row[c].astype(int)

        sample = row
    else:
        sample = data["X_test"].iloc[[args.sample_index]]

    # Background data for SHAP
    bg = data["X_test"].sample(n=min(100, len(data["X_test"])), random_state=42)

    result = predict_and_explain(
        sample=sample,
        background_data=bg,
        use_groq=not args.no_groq,
    )

    _print_result(result)


if __name__ == "__main__":
    main()

