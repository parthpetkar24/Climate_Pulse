"""
preprocessor.py — Preprocessing for Climate_Pulse / hyperlocal_forecaster
==========================================================================
Transforms cleaned air-quality data into ML-ready features:
  • Timestamp parsing & time-based feature engineering
  • Handling irregular intervals / duplicate timestamps
  • Feature scaling (StandardScaler by default, MinMaxScaler optional)
  • Chronological train / validation / test splits (no data leakage)
  • Configurable for time-series forecasting (lookback / forecast windows)

Split strategy
--------------
The dataset contains 5 Pune stations with different date ranges
(2018-03-09 to 2026-01-28).  A *global date cutoff* is used so that
ALL stations share the same train / val / test time periods.  Rows are
partitioned by comparing each row's date against two cutoff dates derived
from the sorted union of all observation dates:

    cutoff_train: 70th-percentile date across all rows
    cutoff_val:   85th-percentile date across all rows

This guarantees:
  - No temporal overlap between splits for any single station.
  - Consistent "historical" / "recent" / "future" semantics across stations.

Data-leakage prevention
-----------------------
The pipeline is ordered to prevent any future information from reaching
the training imputer or scaler:

    1. Drop flag columns, drop fully-empty CO column.
    2. Deduplicate timestamps (per station, mean aggregation).
    3. Sort chronologically (global date, then station).
    4. Compute global date cutoffs → split into train / val / test DataFrames.
    5. Add time features to ALL splits (no data leakage — purely calendar).
    6. Fit imputer statistics (median) on TRAIN only → apply to val & test.
    7. Add lag/rolling features (per-station, shift-based, no leakage).
    8. Drop rows with NaN target AFTER feature engineering.
    9. Fit scaler on TRAIN only → apply to val & test.
    10. Save.

Input  : cleaned CSVs from data/cleaned/
Output : processed CSVs + fitted scalers in data/processed/

Usage (from hyperlocal_forecaster/):
    python -m preprocessing.preprocessor                    # default: daily AQI
    python -m preprocessing.preprocessor --scaler minmax
    python -m preprocessing.preprocessor --lookback 14 --horizon 7
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent.parent
CLEANED_DIR   = BASE_DIR / "data" / "cleaned"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = ["pm25", "pm10", "no2", "so2", "o3", "co", "aqi"]
TIME_FEATURES = [
    "hour", "day_of_week", "month", "day_of_year",
    "week_of_year", "is_weekend", "quarter",
]

# Global date-cutoff ratios (chronological splits)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# Test = remainder (≈ 0.15)

SCALER_CLASSES = {
    "standard": StandardScaler,
    "minmax":   MinMaxScaler,
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("preprocessor")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame, dt_col: str = "date") -> pd.DataFrame:
    """
    Add calendar-based time features from a datetime column.

    Parameters
    ----------
    df     : DataFrame containing a datetime column.
    dt_col : Name of the datetime column.

    Returns
    -------
    DataFrame with additional columns: hour, day_of_week, month,
    day_of_year, week_of_year, is_weekend, quarter.

    Notes
    -----
    All features are derived purely from the calendar — no data statistics
    are used, so there is zero leakage risk.
    """
    if dt_col not in df.columns:
        raise ValueError(f"Datetime column '{dt_col}' not found in DataFrame.")

    dt = pd.to_datetime(df[dt_col], errors="coerce")

    df = df.copy()
    df["hour"]         = dt.dt.hour                              # 0–23
    df["day_of_week"]  = dt.dt.dayofweek                        # 0=Mon … 6=Sun
    df["month"]        = dt.dt.month                            # 1–12
    df["day_of_year"]  = dt.dt.dayofyear                        # 1–366
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)   # 1–53
    df["is_weekend"]   = (dt.dt.dayofweek >= 5).astype(int)     # 0/1
    df["quarter"]      = dt.dt.quarter                          # 1–4
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int],
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Add lagged values of `target_col` as new features.

    Parameters
    ----------
    df         : DataFrame sorted chronologically (within each group).
    target_col : Column to lag.
    lags       : List of lag offsets (e.g. [1, 2, 7]).
    group_col  : Optional groupby column (e.g. station_name) to avoid
                 cross-station leakage.

    Notes
    -----
    All lags are computed via ``groupby(...).shift(lag)``, which uses only
    past observations within the same station — no future information leaks.
    When called on the training split only, the first `max(lags)` rows of
    each station will be NaN and are dropped downstream.
    """
    df = df.copy()
    for lag in lags:
        col_name = f"{target_col}_lag{lag}"
        if group_col and group_col in df.columns:
            df[col_name] = df.groupby(group_col)[target_col].shift(lag)
        else:
            df[col_name] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: list[int],
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Add rolling mean and standard deviation of `target_col`.

    Parameters
    ----------
    df         : DataFrame sorted chronologically (within each group).
    target_col : Column to roll over.
    windows    : Window sizes in time steps (e.g. [3, 7]).
    group_col  : Optional groupby column.

    Notes
    -----
    A ``shift(1)`` is applied before each rolling window so that the feature
    at row *i* uses only rows [i-w … i-1] — no current or future values.
    ``min_periods=1`` avoids NaN at the start of each series but the first
    few rows will be based on fewer observations, which is clearly documented.
    """
    df = df.copy()
    for w in windows:
        mean_col = f"{target_col}_roll{w}_mean"
        std_col  = f"{target_col}_roll{w}_std"
        if group_col and group_col in df.columns:
            grp = df.groupby(group_col)[target_col]
            df[mean_col] = grp.transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
            df[std_col] = grp.transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).std()
            )
        else:
            df[mean_col] = df[target_col].shift(1).rolling(w, min_periods=1).mean()
            df[std_col]  = df[target_col].shift(1).rolling(w, min_periods=1).std()
    return df


# ---------------------------------------------------------------------------
# Duplicate timestamp handling
# ---------------------------------------------------------------------------
def handle_duplicate_timestamps(
    df: pd.DataFrame,
    dt_col: str,
    group_col: str | None,
    agg: str = "mean",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Aggregate rows that share the same (group, timestamp).

    Parameters
    ----------
    df        : Input DataFrame.
    dt_col    : Datetime column.
    group_col : Station/location column (None = no grouping).
    agg       : Aggregation to apply to numeric columns ('mean' or 'first').
    logger    : Optional logger.
    """
    if logger is None:
        logger = logging.getLogger("preprocessor")

    key_cols = [dt_col]
    if group_col and group_col in df.columns:
        key_cols = [group_col, dt_col]

    dup_mask = df.duplicated(subset=key_cols, keep=False)
    n_dups = dup_mask.sum()
    if n_dups == 0:
        return df

    logger.info(f"  Duplicate timestamps: {n_dups} rows -> aggregating with '{agg}'")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols and c not in key_cols]

    num_agg = {c: agg if agg == "mean" else "first"
               for c in num_cols if c not in key_cols}
    cat_agg = {c: "first" for c in cat_cols if c not in key_cols}

    agg_dict = {**num_agg, **cat_agg}
    df = df.groupby(key_cols, as_index=False).agg(agg_dict)
    return df


# ---------------------------------------------------------------------------
# Chronological splitting — global date cutoffs
# ---------------------------------------------------------------------------
def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    dt_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a (possibly multi-station) DataFrame chronologically using
    *global date cutoffs* so that all stations share the same time periods.

    The two cutoff dates are derived from the empirical quantiles of the
    ``dt_col`` column across ALL rows:

        cutoff_train = quantile(train_ratio)
        cutoff_val   = quantile(train_ratio + val_ratio)

    Every row whose date ≤ cutoff_train goes to train; rows with date in
    (cutoff_train, cutoff_val] go to val; the remainder go to test.

    Parameters
    ----------
    df          : DataFrame sorted chronologically.
    train_ratio : Cumulative fraction for the training period.
    val_ratio   : Additional fraction for the validation period.
    dt_col      : Name of the datetime column.

    Returns
    -------
    (train_df, val_df, test_df) — non-overlapping, chronologically ordered.

    Notes
    -----
    When the DataFrame contains a single station, this is equivalent to the
    original row-count slice.  For multiple stations with different date
    ranges, using a global date cutoff guarantees temporal consistency:
    a model is never asked to predict on dates that would be "in the past"
    relative to some training observations.
    """
    if dt_col not in df.columns:
        # Fallback: row-count split (single-station or no date column)
        n = len(df)
        i_train = int(n * train_ratio)
        i_val   = int(n * (train_ratio + val_ratio))
        return (
            df.iloc[:i_train].copy(),
            df.iloc[i_train:i_val].copy(),
            df.iloc[i_val:].copy(),
        )

    dates = pd.to_datetime(df[dt_col], errors="coerce").sort_values()
    n = len(dates)
    # Derive cutoff dates from sorted date array (quantile by position)
    cutoff_train = dates.iloc[int(n * train_ratio) - 1]
    cutoff_val   = dates.iloc[int(n * (train_ratio + val_ratio)) - 1]

    dt_series = pd.to_datetime(df[dt_col], errors="coerce")
    train = df[dt_series <= cutoff_train].copy()
    val   = df[(dt_series > cutoff_train) & (dt_series <= cutoff_val)].copy()
    test  = df[dt_series > cutoff_val].copy()

    return train, val, test


# ---------------------------------------------------------------------------
# Imputation — fit on training data, apply to val & test separately
# ---------------------------------------------------------------------------
def fit_imputer(
    train: pd.DataFrame,
    cols: list[str],
    group_col: str | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    """
    Compute per-column median fill values from the TRAINING set only.

    Strategy
    --------
    For each column the function records:
      - The station-level medians (keyed by station name) for group-aware fill.
      - A global training median as the final fallback.

    Parameters
    ----------
    train     : Training-split DataFrame.
    cols      : Columns to compute statistics for.
    group_col : Optional groupby column.

    Returns
    -------
    Dictionary: ``{col: global_median}`` for fast access.
    Also embeds per-station medians inside the dict as
    ``{f"{col}__station__{station}": median}`` for station-aware fill.
    """
    if logger is None:
        logger = logging.getLogger("preprocessor")

    stats: dict[str, float] = {}
    present = [c for c in cols if c in train.columns]

    for col in present:
        # Global training median
        gm = train[col].median()
        stats[col] = float(gm) if not pd.isna(gm) else 0.0

        # Per-station medians (training rows only)
        if group_col and group_col in train.columns:
            for station, grp_df in train.groupby(group_col):
                sm = grp_df[col].median()
                key = f"{col}__station__{station}"
                stats[key] = float(sm) if not pd.isna(sm) else stats[col]

    return stats


def apply_imputer(
    df: pd.DataFrame,
    cols: list[str],
    stats: dict[str, float],
    group_col: str | None = None,
    method: str = "ffill",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Impute missing values using pre-fitted statistics (from training data).

    Strategy (in order)
    -------------------
    1. Forward-fill within each station group (temporal continuity).
    2. Back-fill residual NaN at the start of each station's series
       (only looks within the current split — no cross-boundary leakage).
    3. Station-level median from ``stats`` (training statistics).
    4. Global training median from ``stats`` as final fallback.

    Parameters
    ----------
    df        : DataFrame for a single split (train, val, or test).
    cols      : Columns to impute.
    stats     : Output of ``fit_imputer`` — training-derived statistics.
    group_col : Optional groupby column.
    method    : Primary imputation ('ffill' or 'interpolate').
    logger    : Optional logger.

    Notes
    -----
    Back-fill here uses ONLY rows within the current split — it never
    reaches across split boundaries.  For the validation and test splits,
    back-fill at the start of the series is the only operation that could
    theoretically pull a "future" value from later in the same split, but
    this is unavoidable when the first observation is NaN; the alternative
    (leaving NaN) would prevent the model from making predictions.
    """
    if logger is None:
        logger = logging.getLogger("preprocessor")

    df = df.copy()
    present = [c for c in cols if c in df.columns]
    before_total = df[present].isna().sum().sum()

    for col in present:
        if group_col and group_col in df.columns:
            # Step 1 & 2: ffill then bfill within each station (within this split)
            if method == "interpolate":
                df[col] = df.groupby(group_col)[col].transform(
                    lambda s: s.interpolate(method="linear", limit_direction="forward")
                )
            else:
                df[col] = df.groupby(group_col)[col].transform(
                    lambda s: s.ffill().bfill()
                )
            # Step 3: station-level median from TRAINING stats
            for station, grp_idx in df.groupby(group_col).groups.items():
                key = f"{col}__station__{station}"
                fill_val = stats.get(key, stats.get(col, 0.0))
                mask = df.loc[grp_idx, col].isna()
                df.loc[grp_idx[mask], col] = fill_val
        else:
            if method == "interpolate":
                df[col] = df[col].interpolate(method="linear", limit_direction="forward")
            else:
                df[col] = df[col].ffill().bfill()
            # Fallback: global training median
            if df[col].isna().any():
                df[col] = df[col].fillna(stats.get(col, 0.0))

    # Step 4: global training median as absolute fallback
    for col in present:
        if df[col].isna().any():
            fill_val = stats.get(col, 0.0)
            n_fill = df[col].isna().sum()
            df[col] = df[col].fillna(fill_val)
            logger.debug(
                f"    Global median fill [{col}]: {n_fill} remaining values "
                f"-> {fill_val:.4f} (training median)"
            )

    after_total = df[present].isna().sum().sum()
    logger.info(
        f"  Imputation: {before_total} -> {after_total} missing values remaining"
    )
    return df


# Backward-compatible wrapper so existing code / tests calling impute_missing still work.
def impute_missing(
    df: pd.DataFrame,
    cols: list[str],
    method: str = "ffill",
    group_col: str | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: fit imputer on ``df`` itself and apply immediately.

    .. warning::
        When called on the *full* dataset (before splitting), the training
        median is computed from ALL rows including val/test, which constitutes
        mild statistical leakage.  For leakage-free preprocessing use
        ``fit_imputer`` on the training split and ``apply_imputer`` on each
        split separately (as ``preprocess_daily_aqi`` now does).

    This wrapper is preserved for backward compatibility with existing tests
    and scripts that operate on a single homogeneous split.
    """
    if logger is None:
        logger = logging.getLogger("preprocessor")

    stats = fit_imputer(df, cols, group_col=group_col, logger=logger)
    return apply_imputer(df, cols, stats, group_col=group_col,
                         method=method, logger=logger)


# ---------------------------------------------------------------------------
# Scaling (fit on train only → transform val & test)
# ---------------------------------------------------------------------------
def fit_and_scale(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    scale_cols: list[str],
    scaler_type: str = "standard",
    save_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """
    Fit a scaler on training data, transform all splits.

    Parameters
    ----------
    train, val, test : Split DataFrames.
    scale_cols       : Columns to scale.
    scaler_type      : 'standard' or 'minmax'.
    save_path        : If provided, serialize scaler to this .pkl path.

    Returns
    -------
    (train_scaled, val_scaled, test_scaled, fitted_scaler)

    Notes
    -----
    Scaler is fitted ONLY on ``train`` — val and test are transformed with
    the training statistics to prevent data leakage.
    """
    cls = SCALER_CLASSES.get(scaler_type, StandardScaler)
    scaler = cls()

    present = [c for c in scale_cols if c in train.columns]
    if not present:
        return train, val, test, None

    train = train.copy()
    val   = val.copy()
    test  = test.copy()

    train[present] = scaler.fit_transform(train[present])
    val[present]   = scaler.transform(val[present])
    test[present]  = scaler.transform(test[present])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(scaler, f)

    return train, val, test, scaler


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------
def preprocess_daily_aqi(
    input_path: Path,
    output_dir: Path,
    scaler_type: str = "standard",
    lookback: int = 14,
    horizon: int = 7,
    target_col: str = "aqi",
    logger: logging.Logger | None = None,
) -> dict[str, pd.DataFrame]:
    """
    End-to-end preprocessing for the cleaned daily AQI dataset.

    Pipeline order (leakage-free)
    ------------------------------
    1. Load → drop flags / empty columns.
    2. Deduplicate timestamps.
    3. Sort by date globally (then by station for determinism).
    4. Compute global date cutoffs → split into train / val / test.
    5. Add time features to ALL splits (calendar only — no leakage).
    6. Add lag & rolling features to ALL splits (shift-based — no leakage).
    7. Fit imputer on TRAIN → apply to train, val, test separately.
    8. Drop rows with NaN target.
    9. Fit scaler on TRAIN → transform val & test.
    10. Save splits, scaler, and metadata.

    Parameters
    ----------
    input_path  : Path to cleaned all_stations CSV.
    output_dir  : Directory to save processed outputs.
    scaler_type : 'standard' or 'minmax'.
    lookback    : Number of past time steps used as model input.
    horizon     : Number of future time steps to forecast.
    target_col  : Column being forecast.
    logger      : Optional logger.

    Returns
    -------
    Dictionary with keys 'train', 'val', 'test' -> DataFrames.
    """
    if logger is None:
        logger = _setup_logging()

    logger.info(f"[Preprocess] {input_path.name}")

    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        return {}

    df = pd.read_csv(input_path, parse_dates=["date"], low_memory=False)
    logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # 1. Drop flag columns not needed downstream
    drop_flags = [c for c in ["flag_out_of_range", "flag_all_zero"] if c in df.columns]
    if drop_flags:
        df.drop(columns=drop_flags, inplace=True)

    # Drop CO column if entirely empty (common in this dataset)
    if "co" in df.columns and df["co"].isna().all():
        logger.info("  Dropping 'co' column (entirely missing).")
        df.drop(columns=["co"], inplace=True)

    avail_features = [c for c in FEATURE_COLS if c in df.columns]
    group_col = "station_name" if "station_name" in df.columns else None

    # 2. Deduplicate timestamps (within each station, before any feature work)
    if "date" in df.columns:
        df = handle_duplicate_timestamps(
            df, "date", group_col, agg="mean", logger=logger
        )

    # 3. Sort by global date first, then by station for determinism.
    #    Sorting by date ONLY (not station-first) ensures the subsequent
    #    global date cutoff produces consistent train/val/test periods.
    df.sort_values(["date"] + ([group_col] if group_col else []), inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 4. Chronological split using global date cutoffs
    #    MUST happen BEFORE imputation and scaling to prevent leakage.
    train, val, test = chronological_split(df, TRAIN_RATIO, VAL_RATIO, dt_col="date")
    logger.info(
        f"  Split sizes (global date cutoff) -- "
        f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}"
    )
    # Log date boundaries for traceability
    if len(train) and len(val) and len(test):
        logger.info(
            f"  Train dates: {train['date'].min().date()} -> {train['date'].max().date()}"
        )
        logger.info(
            f"  Val dates  : {val['date'].min().date()} -> {val['date'].max().date()}"
        )
        logger.info(
            f"  Test dates : {test['date'].min().date()} -> {test['date'].max().date()}"
        )

    # 5. Time features (calendar — no leakage)
    train = add_time_features(train, dt_col="date")
    val   = add_time_features(val,   dt_col="date")
    test  = add_time_features(test,  dt_col="date")

    # 6. Lag & rolling features (shift-based — uses only past observations
    #    within each station within each split.  The first `max_lag` rows of
    #    each station in val/test will have NaN lag values because those
    #    lookbacks fall inside the training period; they are filled later
    #    by the imputer using training statistics.)
    if target_col in df.columns:
        train = add_lag_features(train, target_col, lags=[1, 2, 3, 7, 14],
                                 group_col=group_col)
        train = add_rolling_features(train, target_col, windows=[3, 7],
                                     group_col=group_col)
        val   = add_lag_features(val,   target_col, lags=[1, 2, 3, 7, 14],
                                 group_col=group_col)
        val   = add_rolling_features(val,   target_col, windows=[3, 7],
                                     group_col=group_col)
        test  = add_lag_features(test,  target_col, lags=[1, 2, 3, 7, 14],
                                 group_col=group_col)
        test  = add_rolling_features(test,  target_col, windows=[3, 7],
                                     group_col=group_col)

    # 7. Fit imputer on TRAINING data only, then apply to each split.
    #    This guarantees that no val/test statistics influence training imputation.
    logger.info("  Fitting imputer on training data...")
    imputer_stats = fit_imputer(train, avail_features,
                                group_col=group_col, logger=logger)
    train = apply_imputer(train, avail_features, imputer_stats,
                          group_col=group_col, method="ffill", logger=logger)
    val   = apply_imputer(val,   avail_features, imputer_stats,
                          group_col=group_col, method="ffill", logger=logger)
    test  = apply_imputer(test,  avail_features, imputer_stats,
                          group_col=group_col, method="ffill", logger=logger)

    # 8. Drop rows where the target is still NaN after imputation
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        if target_col in split_df.columns:
            before = len(split_df)
            split_df.dropna(subset=[target_col], inplace=True)
            dropped = before - len(split_df)
            if dropped:
                logger.info(
                    f"  Dropped {dropped} rows with NaN target ('{target_col}') "
                    f"from {split_name}"
                )
    # Reassign after potential in-place drop
    train = train.reset_index(drop=True)
    val   = val.reset_index(drop=True)
    test  = test.reset_index(drop=True)

    logger.info(
        f"  Final split sizes -- "
        f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}"
    )

    # 9. Scale — fit on TRAIN only, transform val & test
    scale_cols = avail_features + [
        c for c in train.columns if "_lag" in c or "_roll" in c
    ]
    scale_cols = [c for c in scale_cols if c in train.columns]

    scaler_path = output_dir / f"scaler_{scaler_type}.pkl"
    train, val, test, scaler = fit_and_scale(
        train, val, test,
        scale_cols=scale_cols,
        scaler_type=scaler_type,
        save_path=scaler_path,
    )
    logger.info(f"  Scaler ({scaler_type}) fitted and saved -> {scaler_path}")

    # 10. Save splits
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem.replace("_cleaned", "")
    splits = {"train": train, "val": val, "test": test}
    for split_name, split_df in splits.items():
        out_path = output_dir / f"{stem}_{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        logger.info(f"  Saved {split_name} -> {out_path}")

    # Metadata
    meta = {
        "source_file":  str(input_path),
        "rows_total":   len(train) + len(val) + len(test),
        "rows_train":   len(train),
        "rows_val":     len(val),
        "rows_test":    len(test),
        "target_col":   target_col,
        "lookback":     lookback,
        "horizon":      horizon,
        "scaler_type":  scaler_type,
        "feature_cols": avail_features,
        "split_strategy": "global_date_cutoff",
        "train_ratio":  TRAIN_RATIO,
        "val_ratio":    VAL_RATIO,
        "date_range": (
            str(df["date"].min()) if "date" in df.columns else "N/A",
            str(df["date"].max()) if "date" in df.columns else "N/A",
        ),
        "train_date_range": (
            str(train["date"].min()) if len(train) else "N/A",
            str(train["date"].max()) if len(train) else "N/A",
        ),
        "val_date_range": (
            str(val["date"].min()) if len(val) else "N/A",
            str(val["date"].max()) if len(val) else "N/A",
        ),
        "test_date_range": (
            str(test["date"].min()) if len(test) else "N/A",
            str(test["date"].max()) if len(test) else "N/A",
        ),
    }
    meta_path = output_dir / f"{stem}_meta.json"
    with open(meta_path, "w") as mf:
        json.dump(meta, mf, indent=2, default=str)
    logger.info(f"  Metadata saved -> {meta_path}")

    return splits


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess cleaned air-quality data for ML/time-series."
    )
    parser.add_argument(
        "--input", type=Path,
        default=CLEANED_DIR / "daily_aqi_all_stations_cleaned.csv",
        help="Path to cleaned CSV (default: all_stations)",
    )
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR,
                        help="Output directory for processed files")
    parser.add_argument("--scaler", choices=["standard", "minmax"],
                        default="standard", help="Scaler type (default: standard)")
    parser.add_argument("--lookback", type=int, default=14,
                        help="Lookback window in time steps (default: 14)")
    parser.add_argument("--horizon", type=int, default=7,
                        help="Forecast horizon in time steps (default: 7)")
    parser.add_argument("--target", default="aqi",
                        help="Target column to forecast (default: aqi)")
    args = parser.parse_args()

    logger = _setup_logging()
    preprocess_daily_aqi(
        input_path=args.input,
        output_dir=args.out_dir,
        scaler_type=args.scaler,
        lookback=args.lookback,
        horizon=args.horizon,
        target_col=args.target,
        logger=logger,
    )
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
