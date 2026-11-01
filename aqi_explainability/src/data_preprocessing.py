"""
================================================================================
  data_preprocessing.py  —  AQI Explainability Pipeline
  ClimatePulse | PCCOE Hackathon  —  MVP 1
================================================================================

Loads the teammate-provided city_averaged.csv dataset, performs leakage-safe
feature engineering, and produces a train/validation/test split for AQI
prediction.

Key design decisions
--------------------
* **Target column**: `aqi` (pre-calculated daily AQI value from CPCB).
* **Dropped columns** (leakage): `sub_pm25`, `sub_pm10`, `sub_no2`, `sub_o3`,
  `sub_co`, `sub_so2`, and `dominant_pollutant` are all derived *from* AQI and
  would leak the answer.
* **Input features**: raw pollutant concentrations (pm25, pm10, no2, o3, co, so2)
  plus temporal features (month, day_of_week, season, etc.).
* **Splitting**: chronological (time-based) because the data has a meaningful
  time dimension — 70 % train / 15 % validation / 15 % test.
* **Missing values**: imputed with column medians fitted on the training set only.

NOTE: Since AQI is *calculated from* the same pollutant concentrations used as
features, the model captures a near-deterministic mapping.  This is acceptable
when the goal is **explainability** (explaining which pollutant drove AQI on a
given day), not out-of-distribution forecasting.
================================================================================
"""

import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # aqi_explainability/
DATA_DIR = BASE_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"

# The teammate's preprocessed dataset lives inside hyperlocal_forecaster.
# Prefer all_stations_combined.csv (3,753 rows) to prevent overfitting and
# provide dense multi-station coverage; fallback to city_averaged.csv if needed.
PROJECT_ROOT   = BASE_DIR.parent                           # Climate_Pulse/
ALL_STATIONS_PATH = (
    PROJECT_ROOT / "hyperlocal_forecaster" / "data" / "raw"
    / "Dataset Daily Air Quality Index and Pollutant Conc"
    / "all_stations_combined.csv"
)
CITY_AVERAGED_PATH = (
    PROJECT_ROOT / "hyperlocal_forecaster" / "data" / "raw"
    / "Dataset Daily Air Quality Index and Pollutant Conc"
    / "city_averaged.csv"
)
DATASET_PATH = ALL_STATIONS_PATH if ALL_STATIONS_PATH.exists() else CITY_AVERAGED_PATH

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("aqi_explainability.preprocessing")

# ── Constants ────────────────────────────────────────────────────────────────
POLLUTANT_COLS   = ["pm25", "pm10", "no2", "o3", "co", "so2"]
LEAKAGE_COLS     = [
    "sub_pm25", "sub_pm10", "sub_no2", "sub_o3", "sub_co", "sub_so2",
    "dominant_pollutant",
]
TARGET_COL       = "aqi"
RANDOM_SEED      = 42

SEASON_MAP = {
    1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON", 12: "DJF",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical + categorical temporal features from the date column."""
    ts = pd.to_datetime(df["date"])
    df = df.copy()
    df["year"]        = ts.dt.year
    df["month"]       = ts.dt.month
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_year"] = ts.dt.dayofyear
    df["quarter"]     = ts.dt.quarter
    df["is_weekend"]  = (ts.dt.dayofweek >= 5).astype(int)
    df["season"]      = df["month"].map(SEASON_MAP)

    # Cyclical encoding
    df["month_sin"]   = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"]       / 12)
    df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["doy_sin"]     = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]     = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df


def load_and_prepare(dataset_path: str | Path | None = None) -> dict:
    """
    Full preprocessing pipeline.

    Returns
    -------
    dict with keys:
        X_train, X_val, X_test   — feature DataFrames
        y_train, y_val, y_test   — target Series
        feature_names            — ordered list of feature column names
        medians                  — dict of column medians (fitted on train)
        metadata                 — dict with dataset summary info
    """
    path = Path(dataset_path) if dataset_path else DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Ensure the teammate's city_averaged.csv is in place."
        )

    log.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    log.info("Raw shape: %s", df.shape)

    # ── 1. Basic validation ──────────────────────────────────────────────
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

    # ── 2. Parse date ────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_bad_dates = df["date"].isna().sum()
    if n_bad_dates:
        log.warning("Dropping %d rows with unparsable dates", n_bad_dates)
        df = df.dropna(subset=["date"])

    # ── 3. Drop duplicates ───────────────────────────────────────────────
    n_dup = df.duplicated().sum()
    if n_dup:
        log.info("Dropping %d duplicate rows", n_dup)
        df = df.drop_duplicates()

    # ── 4. Drop rows where target is missing ─────────────────────────────
    n_na_target = df[TARGET_COL].isna().sum()
    if n_na_target:
        log.warning("Dropping %d rows with missing AQI target", n_na_target)
        df = df.dropna(subset=[TARGET_COL])

    # ── 4b. Filter to continuous multi-pollutant monitoring period ────────
    # In Pune, continuous CAAQMS stations measuring all key pollutants (PM2.5, PM10,
    # NO2, CO, etc.) were commissioned in November 2020. Data prior to 2020-11-13
    # lacked PM10, CO, O3, and SO2, creating severe covariate shift (historical sensor
    # rollout mismatch) where earlier training data was blind to pollutants that dominate
    # modern AQI. Filtering to >= 2020-11-13 ensures all splits share active sensors.
    n_before = len(df)
    df = df[df["date"] >= "2020-11-13"].copy()
    log.info("Filtered pre-CAAQMS era (< 2020-11-13): %d rows kept (dropped %d pre-sensor rows)", len(df), n_before - len(df))

    # ── 5. Remove leakage columns ────────────────────────────────────────
    present_leakage = [c for c in LEAKAGE_COLS if c in df.columns]
    if present_leakage:
        log.info("Removing leakage columns: %s", present_leakage)
        df = df.drop(columns=present_leakage)

    # ── 6. Coerce pollutant columns to numeric + clean ──────────────────
    for col in POLLUTANT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Clip negative values to 0 (instrument calibration errors)
            neg_count = (df[col] < 0).sum()
            if neg_count:
                log.info("Clipping %d negative values in '%s' to 0", neg_count, col)
                df[col] = df[col].clip(lower=0)

    # Sort before forward fill — group by station if available to avoid cross-station bleeding
    if "station_name" in df.columns:
        df = df.sort_values(["station_name", "date"]).reset_index(drop=True)
        for col in POLLUTANT_COLS:
            if col in df.columns:
                df[col] = df.groupby("station_name")[col].ffill(limit=3)
    else:
        df = df.sort_values("date").reset_index(drop=True)
        for col in POLLUTANT_COLS:
            if col in df.columns:
                df[col] = df[col].ffill(limit=3)

    # Drop non-feature metadata columns after forward-fill
    drop_meta = [c for c in ["city", "station_name", "station_id"] if c in df.columns]
    if drop_meta:
        df = df.drop(columns=drop_meta)

    # Domain interaction features (strictly from inputs, no target leakage)
    df["pm_ratio"] = df["pm25"] / (df["pm10"] + 1.0)
    df["pm_sum"] = df["pm25"] + df["pm10"]
    # Relative severity proxy based on CPCB breakpoint scales (PM2.5=60 vs PM10=100)
    df["max_pm_proxy"] = np.maximum(df["pm25"] * 1.67, df["pm10"])

    # ── 7. Add temporal features ─────────────────────────────────────────
    df = _add_temporal_features(df)

    # One-hot encode season
    df = pd.get_dummies(df, columns=["season"], prefix="season", drop_first=False)
    # Ensure bool columns become int
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)

    # ── 8. Sort chronologically for time-based split ─────────────────────
    df = df.sort_values("date").reset_index(drop=True)
    date_range = (df["date"].min(), df["date"].max())
    log.info("Date range: %s → %s (%d rows)", date_range[0].date(), date_range[1].date(), len(df))

    # ── 9. Define feature columns ────────────────────────────────────────
    exclude = {"date", TARGET_COL}
    feature_names = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, np.int32, np.uint8, np.float32]]
    log.info("Feature columns (%d): %s", len(feature_names), feature_names)

    X = df[feature_names]
    y = df[TARGET_COL]

    # ── 10. Chronological split: 70% train / 15% val / 15% test ─────────
    n = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]
    X_val   = X.iloc[n_train:n_train + n_val]
    y_val   = y.iloc[n_train:n_train + n_val]
    X_test  = X.iloc[n_train + n_val:]
    y_test  = y.iloc[n_train + n_val:]

    log.info("Split sizes — train: %d | val: %d | test: %d", len(X_train), len(X_val), len(X_test))

    # ── 11. Impute missing values with training medians ──────────────────
    medians = X_train.median().to_dict()
    # If a column is entirely NaN in the training set (e.g. sparse pollutants
    # in early years), the median is NaN — replace with 0.0 so fillna works.
    for k, v in medians.items():
        if pd.isna(v):
            medians[k] = 0.0
            log.info("Column '%s' is all-NaN in training set — imputing with 0.0", k)

    X_train = X_train.fillna(medians)
    X_val   = X_val.fillna(medians)
    X_test  = X_test.fillna(medians)

    remaining_nans = X_train.isna().sum().sum()
    if remaining_nans > 0:
        log.warning("Still %d NaN values after imputation — filling remainder with 0", remaining_nans)
        X_train = X_train.fillna(0)
        X_val   = X_val.fillna(0)
        X_test  = X_test.fillna(0)

    # ── 12. Build metadata ───────────────────────────────────────────────
    metadata = {
        "dataset_path":   str(path),
        "original_rows":  int(n),
        "date_range":     [str(date_range[0].date()), str(date_range[1].date())],
        "n_features":     len(feature_names),
        "feature_names":  feature_names,
        "target_col":     TARGET_COL,
        "train_size":     len(X_train),
        "val_size":       len(X_val),
        "test_size":      len(X_test),
        "leakage_cols_removed": present_leakage,
        "leakage_note": (
            "AQI is calculated from the same pollutant concentrations used as "
            "input features. The sub-index columns (sub_pm25 etc.) and "
            "dominant_pollutant were removed to prevent direct leakage. "
            "The model learns the CPCB AQI formula mapping, which is acceptable "
            "for explainability purposes."
        ),
    }

    # Save metadata
    meta_path = PROCESSED_DIR / "preprocessing_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Metadata saved → %s", meta_path)

    return {
        "X_train": X_train,
        "X_val":   X_val,
        "X_test":  X_test,
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
        "feature_names": feature_names,
        "medians": medians,
        "metadata": metadata,
    }


# ── CLI entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    result = load_and_prepare()
    print("\n✓ Preprocessing complete.")
    print(f"  Train: {result['metadata']['train_size']}")
    print(f"  Val:   {result['metadata']['val_size']}")
    print(f"  Test:  {result['metadata']['test_size']}")
    print(f"  Features ({result['metadata']['n_features']}): {result['feature_names']}")

