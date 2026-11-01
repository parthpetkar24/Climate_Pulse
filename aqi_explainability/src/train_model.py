"""
================================================================================
  train_model.py  —  AQI Prediction Model Training
  ClimatePulse | PCCOE Hackathon  —  MVP 1
================================================================================

Trains and compares a Random Forest Regressor and an XGBoost Regressor for AQI
prediction, selects the better model on the validation set, evaluates on the
test set, and saves all artefacts (model, scaler, metrics, config).

Usage
-----
    python -m src.train_model

    Or from the project root:
    python aqi_explainability/src/train_model.py
================================================================================
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import load_and_prepare, PROCESSED_DIR

log = logging.getLogger("aqi_explainability.train_model")

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = PROCESSED_DIR.parent.parent / "models"       # aqi_explainability/models/
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────────────

def _evaluate(model, X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    """Return MAE, RMSE, R² for a fitted model on (X, y)."""
    y_pred = model.predict(X)
    mae  = mean_absolute_error(y, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    r2   = r2_score(y, y_pred)
    log.info("  [%s]  MAE=%.4f  RMSE=%.4f  R²=%.4f", label, mae, rmse, r2)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def train_random_forest(X_train, y_train, X_val, y_val) -> tuple:
    """Train a Random Forest Regressor and return (model, val_metrics)."""
    log.info("Training Random Forest Regressor ...")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    train_metrics = _evaluate(rf, X_train, y_train, "RF-Train")
    val_metrics   = _evaluate(rf, X_val,   y_val,   "RF-Val")

    # Overfitting check
    if train_metrics["r2"] - val_metrics["r2"] > 0.10:
        log.warning(
            "  ⚠ Possible overfitting: train R²=%.4f vs val R²=%.4f (gap=%.4f)",
            train_metrics["r2"], val_metrics["r2"],
            train_metrics["r2"] - val_metrics["r2"],
        )

    return rf, {"train": train_metrics, "val": val_metrics}


def train_xgboost(X_train, y_train, X_val, y_val) -> tuple | None:
    """Train XGBoost if available. Returns (model, val_metrics) or None."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        log.warning("XGBoost not installed — skipping. Install with: pip install xgboost")
        return None

    log.info("Training XGBoost Regressor ...")
    xgb = XGBRegressor(
        n_estimators=350,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=3,
        gamma=0.05,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=40,
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    train_metrics = _evaluate(xgb, X_train, y_train, "XGB-Train")
    val_metrics   = _evaluate(xgb, X_val,   y_val,   "XGB-Val")

    if train_metrics["r2"] - val_metrics["r2"] > 0.10:
        log.warning(
            "  ⚠ Possible overfitting: train R²=%.4f vs val R²=%.4f",
            train_metrics["r2"], val_metrics["r2"],
        )

    return xgb, {"train": train_metrics, "val": val_metrics}


def train_and_select(data: dict | None = None) -> dict:
    """
    Full training pipeline: preprocess → train → compare → save best model.

    Parameters
    ----------
    data : dict, optional
        Preprocessed dataset dict from load_and_prepare(). If None, loaded automatically.

    Returns a dict summarising the results.
    """
    # ── 1. Load preprocessed data ────────────────────────────────────────
    if data is None:
        data = load_and_prepare()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    feature_names    = data["feature_names"]
    medians          = data["medians"]

    results = {}

    # ── 2. Train Random Forest ───────────────────────────────────────────
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    results["random_forest"] = rf_metrics

    # ── 3. Train XGBoost ─────────────────────────────────────────────────
    xgb_result = train_xgboost(X_train, y_train, X_val, y_val)
    if xgb_result is not None:
        xgb_model, xgb_metrics = xgb_result
        results["xgboost"] = xgb_metrics
    else:
        xgb_model, xgb_metrics = None, None

    # ── 4. Select best model on validation MAE ───────────────────────────
    candidates = {"random_forest": (rf_model, rf_metrics)}
    if xgb_model is not None:
        candidates["xgboost"] = (xgb_model, xgb_metrics)

    best_name = min(candidates, key=lambda k: candidates[k][1]["val"]["mae"])
    best_model = candidates[best_name][0]
    log.info("✓ Selected model: %s (lowest validation MAE)", best_name)

    # ── 5. Final evaluation on test set ──────────────────────────────────
    log.info("Evaluating %s on TEST set ...", best_name)
    test_metrics = _evaluate(best_model, X_test, y_test, f"{best_name}-Test")
    results["selected_model"] = best_name
    results["test_metrics"]   = test_metrics

    # ── 6. Save artefacts ────────────────────────────────────────────────
    # 6a. Model
    model_path = MODEL_DIR / "best_aqi_model.joblib"
    joblib.dump(best_model, model_path)
    log.info("Model saved → %s", model_path)

    # 6b. Feature medians (for inference-time imputation)
    medians_path = MODEL_DIR / "feature_medians.joblib"
    joblib.dump(medians, medians_path)

    # 6c. Background sample for SHAP inference
    bg_sample = X_train.sample(n=min(100, len(X_train)), random_state=RANDOM_SEED)
    bg_path = MODEL_DIR / "background_sample.joblib"
    joblib.dump(bg_sample, bg_path)
    log.info("Background sample saved → %s", bg_path)

    # 6d. Feature names
    features_path = MODEL_DIR / "feature_names.json"
    with open(features_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    # 6e. Full metrics
    metrics_path = MODEL_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Metrics saved → %s", metrics_path)

    # 6f. Model config
    config = {
        "model_type":     best_name,
        "model_params":   best_model.get_params(),
        "random_seed":    RANDOM_SEED,
        "n_features":     len(feature_names),
        "feature_names":  feature_names,
        "target":         "aqi",
        "split_strategy": "chronological_70_15_15",
    }
    config_path = MODEL_DIR / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    log.info("Config saved → %s", config_path)

    # Summary printout
    log.info("")
    log.info("=" * 60)
    log.info("  TRAINING SUMMARY")
    log.info("=" * 60)
    log.info("  Selected model : %s", best_name)
    log.info("  Test MAE       : %.4f", test_metrics["mae"])
    log.info("  Test RMSE      : %.4f", test_metrics["rmse"])
    log.info("  Test R²        : %.4f", test_metrics["r2"])
    log.info("  Model file     : %s", model_path)
    log.info("=" * 60)

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    warnings.filterwarnings("ignore")
    train_and_select()

