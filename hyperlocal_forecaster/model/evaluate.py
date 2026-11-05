"""
model/evaluate.py -- Evaluation script for the trained AQIForecastLSTM
=======================================================================
Loads the best checkpoint, runs inference on the held-out test set, and
compares against a naive persistence baseline.

Outputs (all in outputs/)
--------------------------
  test_metrics.json           -- all metrics in original AQI units
  actual_vs_predicted.png     -- time-series overlay (actual / LSTM / baseline)
  horizon_metrics.png         -- per-horizon-step MAE and RMSE bar chart

Usage (from hyperlocal_forecaster/)
------------------------------------
  python -m model.evaluate
  python -m model.evaluate --out outputs --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path

# Suppress noisy but harmless warnings before any imports that trigger them
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".mplconfig"))
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import matplotlib
matplotlib.use("Agg")  # headless backend -- no display required
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure hyperlocal_forecaster/ is on PYTHONPATH when run as a script
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

# Import from lstm_model directly to avoid the RuntimeWarning from circular
# import when this module is loaded as part of the 'model' package and also
# executed as __main__ via -m model.evaluate.
from model.lstm_model import AQIForecastLSTM, ModelConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_windows(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a .npz window file and return (X, y) float32 tensors."""
    data = np.load(path)
    X = torch.from_numpy(data["X"].astype(np.float32))
    y = torch.from_numpy(data["y"].astype(np.float32))
    return X, y


def inverse_transform_aqi(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Convert standardised AQI values back to original units."""
    return values * std + mean


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute MAE, RMSE, MSE, and MAPE (where target != 0) for arrays of any
    shape; also compute per-horizon-step metrics assuming last axis = horizon.
    Both arrays must be in the same scale (original AQI units).
    """
    residuals = y_pred - y_true
    mae   = float(np.mean(np.abs(residuals)))
    mse   = float(np.mean(residuals ** 2))
    rmse  = float(np.sqrt(mse))

    # MAPE -- guard against zero targets
    nz = y_true != 0
    mape = float(np.mean(np.abs(residuals[nz] / y_true[nz])) * 100) if nz.any() else float("nan")

    # Per-horizon-step (last dim)
    horizon = y_true.shape[-1]
    per_step_mae  = [float(np.mean(np.abs(y_pred[:, h] - y_true[:, h]))) for h in range(horizon)]
    per_step_rmse = [float(np.sqrt(np.mean((y_pred[:, h] - y_true[:, h]) ** 2))) for h in range(horizon)]

    return {
        "mae":           mae,
        "rmse":          rmse,
        "mse":           mse,
        "mape":          mape,
        "per_step_mae":  per_step_mae,
        "per_step_rmse": per_step_rmse,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_lstm: np.ndarray,
    y_base: np.ndarray,
    output_dir: Path,
    split: str,
    n_samples: int = 200,
) -> None:
    """
    Plot actual AQI vs LSTM prediction vs persistence baseline.
    Uses the 1-day-ahead forecast (horizon step 0) for clarity.
    Shows up to `n_samples` test windows.
    """
    idx = min(n_samples, len(y_true))
    t = np.arange(idx)
    step = 0  # 1-day-ahead

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, y_true[:idx, step],  label="Actual AQI",       color="steelblue",  linewidth=1.5)
    ax.plot(t, y_lstm[:idx, step],  label="LSTM (1-day ahead)", color="darkorange", linewidth=1.2, alpha=0.9)
    ax.plot(t, y_base[:idx, step],  label="Persistence",       color="green",      linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_title(f"Actual vs Predicted AQI — {split} set (1-day ahead, first {idx} samples)")
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("AQI")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / "actual_vs_predicted.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved actual-vs-predicted plot -> %s", out)


def plot_horizon_metrics(
    lstm_metrics: dict,
    base_metrics: dict,
    output_dir: Path,
) -> None:
    """Bar chart of per-step MAE and RMSE for LSTM vs persistence baseline."""
    horizon = len(lstm_metrics["per_step_mae"])
    x = np.arange(1, horizon + 1)
    width = 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # MAE
    ax1.bar(x - width / 2, lstm_metrics["per_step_mae"], width, label="LSTM",        color="darkorange", alpha=0.85)
    ax1.bar(x + width / 2, base_metrics["per_step_mae"], width, label="Persistence", color="green",      alpha=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Day+{i}" for i in x])
    ax1.set_title("MAE per forecast horizon")
    ax1.set_ylabel("MAE (AQI units)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # RMSE
    ax2.bar(x - width / 2, lstm_metrics["per_step_rmse"], width, label="LSTM",        color="darkorange", alpha=0.85)
    ax2.bar(x + width / 2, base_metrics["per_step_rmse"], width, label="Persistence", color="green",      alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Day+{i}" for i in x])
    ax2.set_title("RMSE per forecast horizon")
    ax2.set_ylabel("RMSE (AQI units)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = output_dir / "horizon_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved horizon metrics plot -> %s", out)


def plot_loss_curve(history: dict, output_dir: Path) -> None:
    """Plot training and validation loss curves from training_history.json.

    Handles both old format (single val_loss) and new format that also has
    ext_val_loss (external val_windows.npz loss logged for reference).
    """
    train_loss   = history.get("train_loss",   [])
    val_loss     = history.get("val_loss",     [])   # internal ES val (month-stratified)
    ext_val_loss = history.get("ext_val_loss", [])   # external val_windows.npz (optional)
    best_epoch   = history.get("best_epoch", None)

    if not train_loss:
        logger.warning("No loss history found; skipping loss curve plot.")
        return

    epochs = np.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(epochs, train_loss, label="Train MSE",    color="steelblue",  linewidth=1.5)
    ax.plot(epochs, val_loss,   label="ES-Val MSE",   color="darkorange", linewidth=1.5)
    if ext_val_loss:
        ax.plot(epochs, ext_val_loss, label="Ext-Val MSE (ref)",
                color="purple", linewidth=1.0, linestyle="--", alpha=0.7)
    if best_epoch:
        ax.axvline(best_epoch, color="red", linestyle="--", linewidth=1.0,
                   label=f"Best epoch ({best_epoch})")
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (standardised)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")  # log scale often more informative for loss curves
    fig.tight_layout()
    out = output_dir / "loss_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved loss curve plot -> %s", out)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------
def evaluate(output_dir_str: str = "outputs", split: str = "test") -> dict:
    """
    Load best checkpoint, run inference, compute metrics, save plots.

    Parameters
    ----------
    output_dir_str : str
        Directory containing best_model.pt and model_config.json.
    split : str
        Which data split to evaluate on: 'test', 'val', or 'train'.

    Returns
    -------
    Dict with LSTM metrics, baseline metrics, and metadata.
    """
    output_dir = Path(output_dir_str)
    aug_dir    = _BASE / "data" / "augmented"
    proc_dir   = _BASE / "data" / "processed"

    # --- Load config ---
    cfg_path = output_dir / "model_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"model_config.json not found at {cfg_path}. "
            "Run model.train first."
        )
    cfg = ModelConfig.from_json(cfg_path)
    logger.info("Loaded config from %s", cfg_path)
    logger.info("  n_features=%d  lookback=%d  horizon=%d  target_idx=%d",
                cfg.n_features, cfg.lookback, cfg.horizon, cfg.target_idx)

    # --- Load model ---
    ckpt_path = output_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"best_model.pt not found at {ckpt_path}. "
            "Run model.train first."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = AQIForecastLSTM(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded checkpoint from %s  (params=%s)",
                ckpt_path, f"{model.count_parameters():,}")

    # --- Load windows ---
    split_map = {
        "test":  "test_windows.npz",
        "val":   "val_windows.npz",
        "train": "train_windows.npz",
    }
    if split not in split_map:
        raise ValueError(f"split must be one of {list(split_map)}; got {split!r}")

    npz_path = aug_dir / split_map[split]
    X, y_true_scaled = load_windows(npz_path)
    logger.info("Loaded %s windows: X=%s  y=%s", split,
                tuple(X.shape), tuple(y_true_scaled.shape))

    # --- Inference ---
    criterion = nn.MSELoss()
    with torch.no_grad():
        X_dev = X.to(device)
        y_pred_scaled = model(X_dev).cpu()
        mse_scaled = float(criterion(y_pred_scaled, y_true_scaled).item())

    y_pred_scaled_np = y_pred_scaled.numpy()      # (N, horizon)
    y_true_scaled_np = y_true_scaled.numpy()      # (N, horizon)

    # --- Persistence baseline (last known AQI value repeated for all steps) ---
    # X[:, -1, target_idx] is the last time-step's AQI feature (standardised)
    last_aqi = X[:, -1, cfg.target_idx].numpy()          # (N,)
    y_base_scaled_np = np.tile(last_aqi[:, None], (1, cfg.horizon))  # (N, horizon)

    # --- Inverse-transform ---
    mean, std = cfg.aqi_mean, cfg.aqi_std
    y_true = inverse_transform_aqi(y_true_scaled_np, mean, std)
    y_pred = inverse_transform_aqi(y_pred_scaled_np, mean, std)
    y_base = inverse_transform_aqi(y_base_scaled_np, mean, std)

    # --- Metrics ---
    lstm_metrics = compute_metrics(y_true, y_pred)
    base_metrics = compute_metrics(y_true, y_base)
    mse_base_scaled = float(np.mean((y_base_scaled_np - y_true_scaled_np) ** 2))

    logger.info("=" * 60)
    logger.info("%-30s  %8s  %8s", "Metric", "LSTM", "Persistence")
    logger.info("-" * 60)
    logger.info("%-30s  %8.2f  %8.2f", "MAE (AQI units)",  lstm_metrics["mae"],  base_metrics["mae"])
    logger.info("%-30s  %8.2f  %8.2f", "RMSE (AQI units)", lstm_metrics["rmse"], base_metrics["rmse"])
    logger.info("%-30s  %8.4f  %8.4f", "MSE (scaled)",     mse_scaled,           mse_base_scaled)
    logger.info("%-30s  %7.2f%%  %7.2f%%", "MAPE",         lstm_metrics["mape"], base_metrics["mape"])
    logger.info("=" * 60)
    logger.info("Per-step MAE (LSTM):")
    for i, v in enumerate(lstm_metrics["per_step_mae"], 1):
        logger.info("  Day+%d: %.2f", i, v)

    # --- Plots ---
    plot_actual_vs_predicted(y_true, y_pred, y_base, output_dir, split)
    plot_horizon_metrics(lstm_metrics, base_metrics, output_dir)

    # Loss curve (if training history exists)
    hist_path = output_dir / "training_history.json"
    if hist_path.exists():
        history = json.loads(hist_path.read_text())
        plot_loss_curve(history, output_dir)

    # --- Save metrics JSON ---
    result = {
        "split": split,
        "n_samples": int(len(y_true)),
        "aqi_mean_original": mean,
        "aqi_std_original":  std,
        "lstm": {
            "mse_scaled": mse_scaled,
            **lstm_metrics,
        },
        "persistence_baseline": {
            "mse_scaled": mse_base_scaled,
            **base_metrics,
        },
    }
    metrics_path = output_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2))
    logger.info("Metrics saved -> %s", metrics_path)

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained AQIForecastLSTM on a data split."
    )
    parser.add_argument("--out",   type=str, default="outputs",
                        help="Directory with best_model.pt and model_config.json")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "val", "train"],
                        help="Which split to evaluate (default: test)")
    args = parser.parse_args()
    evaluate(args.out, args.split)


if __name__ == "__main__":
    main()

