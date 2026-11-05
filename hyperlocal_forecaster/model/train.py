"""
model/train.py -- Training script for AQIForecastLSTM
======================================================
Loads augmented training windows (from data/augmented/), trains the
LSTM with early stopping on validation loss, saves the best checkpoint,
and writes training history (JSON + loss plot).

Usage (from hyperlocal_forecaster/)
------------------------------------
  python -m model.train                              # defaults
  python -m model.train --epochs 200 --hidden 256   # custom
  python -m model.train --help                       # all options

Outputs (all in outputs/)
--------------------------
  best_model.pt          -- PyTorch state dict of the best checkpoint
  model_config.json      -- ModelConfig used to build the model
  training_history.json  -- epoch-by-epoch train/val loss
  loss_curve.png         -- plot of training and validation loss
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress noisy but harmless warnings
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".mplconfig"))
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Ensure hyperlocal_forecaster/ is on PYTHONPATH when run as a script
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from model.lstm_model import AQIForecastLSTM, ModelConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Ensure reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_windows(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a .npz window file and return (X, y) float32 tensors."""
    data = np.load(path)
    X = torch.from_numpy(data["X"].astype(np.float32))
    y = torch.from_numpy(data["y"].astype(np.float32))
    return X, y


def make_dataloader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=False, num_workers=0)


def month_stratified_split(
    X: torch.Tensor,
    y: torch.Tensor,
    month_feat_idx: int,
    holdout_frac: float = 0.20,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split training windows into fit-train and an internal early-stopping val set
    that is balanced across all months present in the data.

    The month feature at the last time step (X[:, -1, month_feat_idx]) is used
    to identify which calendar month each window belongs to.  Windows whose
    month values round to 0 or >12 are assigned to the chronological tail.

    This avoids early stopping on a pure out-of-sample val split (Apr–Sep 2025)
    that has a completely different seasonal distribution from training, which
    would cause misleading val-loss signals and premature stopping.

    Parameters
    ----------
    X, y          : full training window tensors
    month_feat_idx: index of the 'month' feature in the feature dimension
    holdout_frac  : fraction of windows to hold out for early-stopping val
    seed          : random seed for per-stratum sampling

    Returns
    -------
    X_fit, y_fit, X_es_val, y_es_val
    """
    rng = np.random.default_rng(seed)
    months_raw = X[:, -1, month_feat_idx].numpy()          # (N,)
    months     = np.round(months_raw).astype(int)
    months     = np.clip(months, 1, 12)                    # guard against edge values

    es_idx: list[int] = []
    for m in range(1, 13):
        m_idx = np.where(months == m)[0]
        if len(m_idx) == 0:
            continue
        n_hold = max(1, int(round(len(m_idx) * holdout_frac)))
        chosen = rng.choice(m_idx, size=n_hold, replace=False)
        es_idx.extend(chosen.tolist())

    es_mask  = np.zeros(len(X), dtype=bool)
    es_mask[es_idx] = True
    fit_mask = ~es_mask

    fit_idx = np.where(fit_mask)[0]
    es_idx_arr = np.array(es_idx)

    X_fit   = X[fit_idx]
    y_fit   = y[fit_idx]
    X_es    = X[es_idx_arr]
    y_es    = y[es_idx_arr]

    return X_fit, y_fit, X_es, y_es


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(cfg: ModelConfig) -> dict:
    """
    Full training routine.

    Returns
    -------
    history dict with keys 'train_loss', 'val_loss' (lists of floats)
    and 'best_epoch', 'best_val_loss'.
    """
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("  GPU: %s", torch.cuda.get_device_name(0))

    # --- Data ---
    aug_dir = _BASE / "data" / "augmented"
    logger.info("Loading windows from %s", aug_dir)

    X_train_all, y_train_all = load_windows(aug_dir / "train_windows.npz")
    X_ext_val,   y_ext_val   = load_windows(aug_dir / "val_windows.npz")

    logger.info("  Train (all): X=%s  y=%s",
                tuple(X_train_all.shape), tuple(y_train_all.shape))
    logger.info("  Ext-Val    : X=%s  y=%s",
                tuple(X_ext_val.shape),   tuple(y_ext_val.shape))

    # Validate shapes match config
    assert X_train_all.shape[1] == cfg.lookback,   "lookback mismatch"
    assert X_train_all.shape[2] == cfg.n_features, "n_features mismatch"
    assert y_train_all.shape[1] == cfg.horizon,    "horizon mismatch"

    # --- Month-stratified internal early-stopping split ---
    # Early stopping on the external val set (Apr–Sep 2025) is unreliable because
    # it covers only summer months — a very different distribution from training.
    # Instead we carve out a balanced 20 % of training windows (all 12 months
    # represented) as the early-stopping signal.  The checkpoint saved is still
    # the one with the best internal-val loss, then we also report external-val
    # loss each epoch for transparency.
    if cfg.month_stratified_es and cfg.month_feat_idx >= 0:
        logger.info(
            "Using month-stratified internal early-stopping split "
            "(holdout_frac=%.2f, month_feat_idx=%d)",
            cfg.es_holdout_frac, cfg.month_feat_idx,
        )
        X_train, y_train, X_es_val, y_es_val = month_stratified_split(
            X_train_all, y_train_all,
            month_feat_idx=cfg.month_feat_idx,
            holdout_frac=cfg.es_holdout_frac,
            seed=cfg.seed,
        )
        logger.info("  Fit-train  : X=%s  y=%s", tuple(X_train.shape), tuple(y_train.shape))
        logger.info("  ES-val     : X=%s  y=%s", tuple(X_es_val.shape), tuple(y_es_val.shape))
    else:
        X_train, y_train = X_train_all, y_train_all
        X_es_val, y_es_val = X_ext_val, y_ext_val
        logger.info("Using external val_windows.npz for early stopping (no stratification)")

    train_loader  = make_dataloader(X_train,  y_train,  cfg.batch_size, shuffle=True)
    es_val_loader = make_dataloader(X_es_val, y_es_val, cfg.batch_size, shuffle=False)
    ext_val_loader= make_dataloader(X_ext_val,y_ext_val,cfg.batch_size, shuffle=False)

    # --- Model ---
    model = AQIForecastLSTM(cfg).to(device)
    logger.info("Model parameters: %s", f"{model.count_parameters():,}")

    # --- Optimiser & Loss ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # ReduceLROnPlateau: halve LR if val loss doesn't improve for 5 epochs.
    # More adaptive than CosineAnnealing for datasets with distribution shift
    # between train and validation splits.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=cfg.lr * 0.001,
    )
    criterion = nn.MSELoss()

    # --- Training loop ---
    history: dict = {
        "train_loss": [],
        "val_loss":   [],       # internal early-stopping val (month-stratified if enabled)
        "ext_val_loss": [],     # external val_windows.npz loss (reported but not used for ES)
        "best_epoch":     0,
        "best_val_loss":  float("inf"),
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt_path = output_dir / "best_model.pt"

    logger.info("Starting training for up to %d epochs (patience=%d)",
                cfg.max_epochs, cfg.patience)
    t0 = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg.grad_clip
        )
        es_val_loss  = evaluate(model, es_val_loader,  criterion, device)
        ext_val_loss = evaluate(model, ext_val_loader, criterion, device)
        scheduler.step(es_val_loss)  # ReduceLROnPlateau uses internal (balanced) val

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(es_val_loss))
        history["ext_val_loss"].append(float(ext_val_loss))

        improved = es_val_loss < best_val_loss
        if improved:
            best_val_loss = es_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
            history["best_epoch"]    = epoch
            history["best_val_loss"] = float(best_val_loss)

        if epoch % 10 == 0 or epoch <= 5 or improved:
            lr_now = optimizer.param_groups[0]["lr"]
            tag = " <-- best" if improved else ""
            logger.info(
                "Epoch %3d/%d  train=%.4f  es_val=%.4f  ext_val=%.4f  lr=%.2e%s",
                epoch, cfg.max_epochs, train_loss, es_val_loss, ext_val_loss, lr_now, tag,
            )

        patience_counter += (0 if improved else 1)
        if patience_counter >= cfg.patience:
            logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch, cfg.patience)
            break

    elapsed = time.time() - t0
    logger.info(
        "Training complete in %.1f s  |  best val MSE=%.6f at epoch %d",
        elapsed, best_val_loss, history["best_epoch"],
    )

    # --- Save history ---
    hist_path = output_dir / "training_history.json"
    hist_path.write_text(json.dumps(history, indent=2))
    logger.info("History saved -> %s", hist_path)

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train AQIForecastLSTM on augmented windows."
    )
    # Architecture
    parser.add_argument("--hidden",   type=int,   default=64,
                        help="LSTM hidden units (default 64; smaller reduces memorisation)")
    parser.add_argument("--layers",   type=int,   default=2)
    parser.add_argument("--dropout",  type=float, default=0.3,
                        help="Dropout between LSTM layers (default 0.3)")
    # Training
    parser.add_argument("--lr",       type=float, default=5e-4,
                        help="Initial learning rate (default 5e-4)")
    parser.add_argument("--wd",       type=float, default=1e-4)
    parser.add_argument("--batch",    type=int,   default=64)
    parser.add_argument("--epochs",   type=int,   default=200)
    parser.add_argument("--patience", type=int,   default=25)
    parser.add_argument("--clip",     type=float, default=1.0)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--out",      type=str,   default="outputs")
    # Month-stratified early stopping
    parser.add_argument("--no-strat-es", action="store_true",
                        help="Disable month-stratified internal val split (use external val for ES)")
    parser.add_argument("--es-holdout", type=float, default=0.20,
                        help="Fraction of train windows held out per month for ES (default 0.20)")
    args = parser.parse_args()

    # Build config from augmentation metadata
    aug_meta_path = _BASE / "data" / "augmented" / "augmentation_meta.json"
    proc_meta_path = _BASE / "data" / "processed" / "daily_aqi_all_stations_meta.json"
    aug_meta  = json.loads(aug_meta_path.read_text())
    proc_meta = json.loads(proc_meta_path.read_text())

    # AQI scaler parameters (for inverse-transform)
    import pickle
    scaler_path = _BASE / "data" / "processed" / "scaler_standard.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    scaler_feat_cols = proc_meta.get("feature_cols", [])
    aqi_scaler_idx = scaler_feat_cols.index("aqi") if "aqi" in scaler_feat_cols else 6
    aqi_mean = float(scaler.mean_[aqi_scaler_idx])
    aqi_std  = float(scaler.scale_[aqi_scaler_idx])

    feat_cols = aug_meta["feature_cols"]
    # Find the month feature index in the window feature columns
    month_feat_idx = feat_cols.index("month") if "month" in feat_cols else -1

    cfg = ModelConfig(
        n_features=len(feat_cols),
        lookback=aug_meta["lookback"],
        horizon=aug_meta["horizon"],
        target_idx=feat_cols.index(aug_meta["target_col"]),
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.wd,
        batch_size=args.batch,
        max_epochs=args.epochs,
        patience=args.patience,
        grad_clip=args.clip,
        seed=args.seed,
        month_stratified_es=not args.no_strat_es,
        es_holdout_frac=args.es_holdout,
        month_feat_idx=month_feat_idx,
        aqi_mean=aqi_mean,
        aqi_std=aqi_std,
        feature_cols=feat_cols,
        output_dir=args.out,
    )

    # Save config
    cfg_path = Path(args.out) / "model_config.json"
    cfg.save(cfg_path)
    logger.info("Config saved -> %s", cfg_path)
    logger.info("n_features=%d  lookback=%d  horizon=%d  target_idx=%d",
                cfg.n_features, cfg.lookback, cfg.horizon, cfg.target_idx)
    logger.info("month_stratified_es=%s  month_feat_idx=%d  es_holdout=%.2f",
                cfg.month_stratified_es, cfg.month_feat_idx, cfg.es_holdout_frac)

    train(cfg)


if __name__ == "__main__":
    main()

