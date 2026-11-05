"""
augmentor.py -- Data Augmentation for Climate_Pulse / hyperlocal_forecaster
===========================================================================

Augmentation strategy for air-quality time-series
--------------------------------------------------
Image-style augmentation (flips, rotations, colour jitter) is physically
meaningless for pollution data and is NOT used here.

Two scientifically justified techniques are implemented:

1. **Sliding-window sequence generation** (always appropriate)
   Converts a sorted time-series into overlapping (X, y) window pairs for
   sequence models (LSTM, Transformer, CNN-based forecasters).
   - X shape: (n_samples, lookback, n_features)
   - y shape: (n_samples, horizon)

2. **Gaussian noise injection** (training windows only, opt-in)
   Adds small, physically plausible noise (sigma = noise_std x local_std) to
   training X windows to improve model generalisation.
   - Applied ONLY to training windows.
   - Noise magnitude is a configurable fraction of each feature's standard
     deviation (default sigma_fraction = 0.01 -> 1 %).
   - Augmented samples double the training set size.
   - Preserves relative pollutant relationships.

Validation and test windows are NEVER augmented.

Usage (from hyperlocal_forecaster/):
    python -m augmentation.augmentor                            # default
    python -m augmentation.augmentor --lookback 14 --horizon 7
    python -m augmentation.augmentor --no-noise                 # windows only
    python -m augmentation.augmentor --noise-std 0.02           # 2 % noise
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
AUGMENTED_DIR = BASE_DIR / "data" / "augmented"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("augmentor")
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
# Sliding-window generation
# ---------------------------------------------------------------------------
def sliding_windows(
    values: np.ndarray,
    lookback: int,
    horizon: int,
    target_idx: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate sliding-window (X, y) pairs from a 2-D array.

    Parameters
    ----------
    values     : 2-D array of shape (T, n_features), sorted chronologically.
    lookback   : Number of past time steps to use as model input.
    horizon    : Number of future time steps to predict.
    target_idx : Column index of the target variable in `values`.
    stride     : Step between consecutive windows (default 1).

    Returns
    -------
    X : ndarray of shape (n_windows, lookback, n_features)
    y : ndarray of shape (n_windows, horizon)
    """
    T, F = values.shape
    n_windows = max(0, (T - lookback - horizon) // stride + 1)

    X = np.empty((n_windows, lookback, F), dtype=np.float32)
    y = np.empty((n_windows, horizon),     dtype=np.float32)

    for i in range(n_windows):
        start = i * stride
        X[i] = values[start : start + lookback]
        y[i] = values[start + lookback : start + lookback + horizon, target_idx]

    return X, y


def build_windows_for_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookback: int,
    horizon: int,
    group_col: str | None = None,
    stride: int = 1,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window arrays from a (possibly multi-station) DataFrame.

    When `group_col` is provided, windows are generated per-station to avoid
    cross-station temporal leakage at boundaries.

    Parameters
    ----------
    df           : Preprocessed, sorted DataFrame.
    feature_cols : Ordered list of feature columns to include in X.
    target_col   : Column to predict (must be in feature_cols or df.columns).
    lookback     : Past steps in input window.
    horizon      : Future steps to predict.
    group_col    : Groupby column (e.g. 'station_name').
    stride       : Step between consecutive windows.
    logger       : Optional logger.

    Returns
    -------
    X : (n_windows, lookback, n_features)
    y : (n_windows, horizon)
    """
    if logger is None:
        logger = logging.getLogger("augmentor")

    present_features = [c for c in feature_cols if c in df.columns]
    if target_col not in present_features and target_col in df.columns:
        present_features.append(target_col)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    target_idx = present_features.index(target_col)

    Xs, ys = [], []

    if group_col and group_col in df.columns:
        groups = df[group_col].unique()
        for grp in sorted(groups):
            sub = df[df[group_col] == grp][present_features].copy()
            # Drop rows where the target itself is NaN; forward-fill everything else
            sub = sub[sub[target_col].notna()].ffill().bfill()
            # Drop any remaining all-NaN rows
            sub = sub.dropna(how="any").values.astype(np.float32)
            if len(sub) < lookback + horizon:
                logger.debug(f"  Skipping '{grp}': only {len(sub)} rows < {lookback+horizon}")
                continue
            Xi, yi = sliding_windows(sub, lookback, horizon, target_idx, stride)
            Xs.append(Xi)
            ys.append(yi)
    else:
        sub = df[present_features].copy()
        sub = sub[sub[target_col].notna()].ffill().bfill().dropna(how="any")
        vals = sub.values.astype(np.float32)
        X, y = sliding_windows(vals, lookback, horizon, target_idx, stride)
        Xs.append(X)
        ys.append(y)

    if not Xs:
        logger.warning("  No windows generated (insufficient data).")
        return np.empty((0, lookback, len(present_features)), dtype=np.float32), \
               np.empty((0, horizon), dtype=np.float32)

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    logger.info(f"  Windows: X={X_all.shape}, y={y_all.shape}")
    return X_all, y_all


# ---------------------------------------------------------------------------
# Noise injection (training only)
# ---------------------------------------------------------------------------
def gaussian_noise_augment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    noise_std_fraction: float = 0.01,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment training windows with additive Gaussian noise.

    Noise is scaled per-feature: sigma_feature = noise_std_fraction x std(feature).
    This preserves the relative scale and physical plausibility of each
    pollutant measurement.

    Parameters
    ----------
    X_train            : Training input windows, shape (N, lookback, F).
    y_train            : Training targets, shape (N, horizon).
    noise_std_fraction : Fraction of feature std to use as noise sigma (default 0.01).
    rng_seed           : Random seed for reproducibility.

    Returns
    -------
    X_aug : Concatenation of original + noisy windows, shape (2N, lookback, F).
    y_aug : Duplicated targets (noise not added to y), shape (2N, horizon).

    Notes
    -----
    - Only input X is noised; targets y are duplicated unchanged.
    - Never call this on validation or test data.
    """
    if noise_std_fraction <= 0:
        return X_train, y_train

    rng = np.random.default_rng(rng_seed)

    # Compute per-feature std across all (samples, time-steps)
    # X shape: (N, T, F)
    feat_std = X_train.std(axis=(0, 1), keepdims=True)  # (1, 1, F)
    noise = rng.normal(0.0, noise_std_fraction * feat_std, size=X_train.shape).astype(np.float32)
    X_noisy = X_train + noise

    X_aug = np.concatenate([X_train, X_noisy], axis=0)
    y_aug = np.concatenate([y_train, y_train], axis=0)

    return X_aug, y_aug


# ---------------------------------------------------------------------------
# Save windows as compressed numpy archives
# ---------------------------------------------------------------------------
def save_windows(
    X: np.ndarray,
    y: np.ndarray,
    path: Path,
    logger: logging.Logger | None = None,
) -> None:
    """Save (X, y) window arrays as a compressed .npz file."""
    if logger is None:
        logger = logging.getLogger("augmentor")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    logger.info(f"  Saved windows -> {path}  (X={X.shape}, y={y.shape})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate augmented sliding-window datasets for time-series forecasting."
    )
    parser.add_argument("--train-file", type=Path,
                        default=PROCESSED_DIR / "daily_aqi_all_stations_train.csv")
    parser.add_argument("--val-file",   type=Path,
                        default=PROCESSED_DIR / "daily_aqi_all_stations_val.csv")
    parser.add_argument("--test-file",  type=Path,
                        default=PROCESSED_DIR / "daily_aqi_all_stations_test.csv")
    parser.add_argument("--out-dir",    type=Path, default=AUGMENTED_DIR)
    parser.add_argument("--lookback",   type=int,  default=14)
    parser.add_argument("--horizon",    type=int,  default=7)
    parser.add_argument("--target",     default="aqi")
    parser.add_argument("--stride",     type=int,  default=1)
    parser.add_argument("--noise-std",  type=float, default=0.01,
                        help="Fraction of feature std used as noise sigma (default 0.01)")
    parser.add_argument("--no-noise",   action="store_true",
                        help="Disable Gaussian noise augmentation (windows only)")
    args = parser.parse_args()

    logger = _setup_logging()
    logger.info("=" * 60)
    logger.info("Climate_Pulse -- Data Augmentation")
    logger.info(f"Lookback : {args.lookback}, Horizon : {args.horizon}")
    logger.info(f"Noise    : {'disabled' if args.no_noise else f'std_fraction={args.noise_std}'}")
    logger.info("=" * 60)

    def _load(path: Path) -> pd.DataFrame:
        if not path.exists():
            logger.error(f"File not found: {path}")
            return pd.DataFrame()
        return pd.read_csv(path, low_memory=False)

    train_df = _load(args.train_file)
    val_df   = _load(args.val_file)
    test_df  = _load(args.test_file)

    if train_df.empty:
        logger.error("Training data is empty -- aborting.")
        return

    # Infer feature columns (exclude metadata / string cols)
    exclude = {"date", "city", "station_name", "station_id", "dominant_pollutant",
               "source_file", "state_parsed", "city_parsed"}
    feature_cols = [
        c for c in train_df.columns
        if c not in exclude and train_df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
    ]
    group_col = "station_name" if "station_name" in train_df.columns else None

    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:8]}{'...' if len(feature_cols)>8 else ''}")

    # --- Training windows ---
    logger.info("[Train] Building windows...")
    X_train, y_train = build_windows_for_split(
        train_df, feature_cols, args.target,
        args.lookback, args.horizon, group_col, args.stride, logger,
    )

    if not args.no_noise and len(X_train) > 0:
        logger.info(f"[Train] Applying Gaussian noise (sigma_fraction={args.noise_std})...")
        X_train, y_train = gaussian_noise_augment(X_train, y_train, args.noise_std)
        logger.info(f"  Augmented train set: X={X_train.shape}, y={y_train.shape}")

    save_windows(X_train, y_train, args.out_dir / "train_windows.npz", logger)

    # --- Validation windows (no augmentation) ---
    # Default empty arrays; overwritten below if val data is available.
    X_val: np.ndarray = np.empty((0, args.lookback, 0), dtype=np.float32)
    y_val: np.ndarray = np.empty((0, args.horizon), dtype=np.float32)
    if not val_df.empty:
        logger.info("[Val] Building windows (NO augmentation)...")
        X_val, y_val = build_windows_for_split(
            val_df, feature_cols, args.target,
            args.lookback, args.horizon, group_col, args.stride, logger,
        )
        save_windows(X_val, y_val, args.out_dir / "val_windows.npz", logger)

    # --- Test windows (no augmentation) ---
    # Default empty arrays; overwritten below if test data is available.
    X_test: np.ndarray = np.empty((0, args.lookback, 0), dtype=np.float32)
    y_test: np.ndarray = np.empty((0, args.horizon), dtype=np.float32)
    if not test_df.empty:
        logger.info("[Test] Building windows (NO augmentation)...")
        X_test, y_test = build_windows_for_split(
            test_df, feature_cols, args.target,
            args.lookback, args.horizon, group_col, args.stride, logger,
        )
        save_windows(X_test, y_test, args.out_dir / "test_windows.npz", logger)

    # Save augmentation metadata
    meta = {
        "lookback":          args.lookback,
        "horizon":           args.horizon,
        "target_col":        args.target,
        "stride":            args.stride,
        "noise_std_fraction": None if args.no_noise else args.noise_std,
        "feature_cols":      feature_cols,
        "train_windows":     int(len(X_train)),
        "val_windows":       int(len(X_val)) if not val_df.empty else 0,
        "test_windows":      int(len(X_test)) if not test_df.empty else 0,
    }
    meta_path = args.out_dir / "augmentation_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as mf:
        json.dump(meta, mf, indent=2)
    logger.info(f"  Metadata saved -> {meta_path}")
    logger.info("Augmentation complete.")


if __name__ == "__main__":
    main()


