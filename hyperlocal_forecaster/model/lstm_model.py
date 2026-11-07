"""
model/lstm_model.py -- LSTM model for multi-step AQI forecasting
=================================================================
Architecture
------------
  Input  : (batch, lookback, n_features)  -- standardised feature windows
  Output : (batch, horizon)               -- predicted AQI for the next
                                             `horizon` days (standardised)

Design choices
--------------
* Stacked LSTM (configurable depth) with per-layer dropout for
  regularisation (applied between LSTM layers, not on the last layer's
  output to avoid masking the hidden state).
* A fully-connected head maps the final hidden state to `horizon` outputs
  simultaneously (direct multi-step forecasting -- avoids error accumulation
  of recursive strategies on a dataset this size).
* Layer Norm applied to the output of the last LSTM layer improves
  training stability for standardised targets.
* Weight initialisation follows the PyTorch default (Kaiming uniform for
  linear layers), which is appropriate for tanh/sigmoid LSTM gates.

Usage
-----
  from model.lstm_model import AQIForecastLSTM, ModelConfig
  cfg = ModelConfig()
  net = AQIForecastLSTM(cfg)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """All hyperparameters needed to build and train the model."""
    # Data dimensions (set from augmentation metadata)
    n_features: int = 29          # number of input features per time step
    lookback:   int = 14          # input sequence length
    horizon:    int = 7           # forecast horizon (output length)
    target_idx: int = 12          # index of 'aqi' in feature_cols

    # Architecture
    hidden_size:  int        = 64    # LSTM hidden units (reduced from 128 to curb memorisation)
    num_layers:   int        = 2     # stacked LSTM depth
    dropout:      float      = 0.3   # dropout between LSTM layers (increased for regularisation)
    bidirectional: bool      = False # bidirectional LSTM (doubles hidden_size)

    # Training
    lr:            float = 5e-4   # lower LR for more stable convergence
    weight_decay:  float = 1e-4
    batch_size:    int   = 64
    max_epochs:    int   = 200
    patience:      int   = 25     # early-stopping patience (val-loss epochs)
    grad_clip:     float = 1.0    # gradient clipping max norm
    seed:          int   = 42

    # Month-stratified early-stopping split
    # Carve out a balanced 20% of training windows (all 12 months) for
    # early stopping, instead of using the external val set which covers
    # only Apr-Sep and would cause misleading val-loss signals.
    month_stratified_es: bool  = True
    es_holdout_frac:     float = 0.20  # fraction of train windows held out per month
    month_feat_idx:      int   = 15    # index of 'month' feature in feature_cols

    # Scaler info (for inverse-transforming predictions)
    aqi_mean: float = 132.823
    aqi_std:  float = 124.883

    # Feature column names (populated from metadata)
    feature_cols: list = field(default_factory=list)

    # I/O
    output_dir: str = "outputs"

    # ------------------------------------------------------------------ #
    @classmethod
    def from_json(cls, path: Path) -> "ModelConfig":
        data = json.loads(path.read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AQIForecastLSTM(nn.Module):
    """
    Stacked LSTM for multi-step AQI forecasting.

    Parameters
    ----------
    cfg : ModelConfig
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.lstm = nn.LSTM(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )

        # Layer normalisation on the LSTM output
        lstm_out_size = cfg.hidden_size * (2 if cfg.bidirectional else 1)
        self.layer_norm = nn.LayerNorm(lstm_out_size)

        # Fully-connected head: direct multi-step output
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size, cfg.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size // 2, cfg.horizon),
        )

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, lookback, n_features)

        Returns
        -------
        Tensor of shape (batch, horizon) -- standardised AQI predictions
        """
        # lstm_out: (batch, seq, hidden)
        lstm_out, _ = self.lstm(x)
        # Use only the last time step's hidden state
        last_hidden = lstm_out[:, -1, :]        # (batch, hidden)
        normed = self.layer_norm(last_hidden)   # (batch, hidden)
        out = self.fc(normed)                   # (batch, horizon)
        return out

    # ------------------------------------------------------------------ #
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

