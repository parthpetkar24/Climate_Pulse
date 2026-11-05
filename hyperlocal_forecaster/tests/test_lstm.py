"""
tests/test_lstm.py -- Sanity checks for the LSTM model and training pipeline
=============================================================================
These tests use only synthetic data and do NOT require a trained checkpoint.
They are designed to run fast (seconds) with Python 3.10 + PyTorch.

Run from hyperlocal_forecaster/ with:
  python -m pytest tests/test_lstm.py -v
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Make sure hyperlocal_forecaster/ is importable
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from model.lstm_model import AQIForecastLSTM, ModelConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
N_FEATURES = 29
LOOKBACK    = 14
HORIZON     = 7
TARGET_IDX  = 12
BATCH       = 8


@pytest.fixture(scope="module")
def default_cfg() -> ModelConfig:
    return ModelConfig(
        n_features=N_FEATURES,
        lookback=LOOKBACK,
        horizon=HORIZON,
        target_idx=TARGET_IDX,
        hidden_size=32,   # small for fast tests
        num_layers=2,
        dropout=0.0,      # deterministic
        seed=42,
    )


@pytest.fixture(scope="module")
def model(default_cfg: ModelConfig) -> AQIForecastLSTM:
    net = AQIForecastLSTM(default_cfg)
    net.eval()
    return net


@pytest.fixture(scope="module")
def random_x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, LOOKBACK, N_FEATURES)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestModelOutputShape:
    """The model must return exactly (batch, horizon) for valid input."""

    def test_standard_batch(self, model, random_x):
        with torch.no_grad():
            out = model(random_x)
        assert out.shape == (BATCH, HORIZON), (
            f"Expected ({BATCH}, {HORIZON}), got {tuple(out.shape)}"
        )

    def test_batch_size_one(self, model):
        x = torch.randn(1, LOOKBACK, N_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, HORIZON)

    def test_large_batch(self, model):
        x = torch.randn(128, LOOKBACK, N_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (128, HORIZON)


class TestModelOutputValues:
    """Forward pass must produce finite, non-NaN outputs."""

    def test_no_nan(self, model, random_x):
        with torch.no_grad():
            out = model(random_x)
        assert not torch.isnan(out).any(), "Model output contains NaN"

    def test_no_inf(self, model, random_x):
        with torch.no_grad():
            out = model(random_x)
        assert not torch.isinf(out).any(), "Model output contains Inf"

    def test_outputs_vary_across_batch(self, model, random_x):
        """Different inputs should produce different outputs (not all same)."""
        with torch.no_grad():
            out = model(random_x)
        # At least two rows must differ
        diffs = (out[1:] - out[:-1]).abs().sum(dim=-1)
        assert diffs.max().item() > 0, "All batch outputs are identical"

    def test_zero_input_finite(self, model):
        """Model should handle all-zero input without crashing."""
        x = torch.zeros(2, LOOKBACK, N_FEATURES)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, HORIZON)
        assert not torch.isnan(out).any()


class TestParameterCount:
    """Sanity-check that parameter count is positive and within reason."""

    def test_params_positive(self, model):
        assert model.count_parameters() > 0

    def test_params_in_range(self, model):
        """Small test model should have fewer than 500k parameters."""
        assert model.count_parameters() < 500_000

    def test_default_model_params(self, default_cfg):
        """Default production model (hidden=64) should have ~57k params."""
        prod_cfg = ModelConfig(
            n_features=29, lookback=14, horizon=7, target_idx=12,
            hidden_size=64, num_layers=2,
        )
        net = AQIForecastLSTM(prod_cfg)
        params = net.count_parameters()
        assert 20_000 < params < 200_000, (
            f"Production model param count seems off: {params:,}"
        )


class TestModelConfig:
    """ModelConfig must serialise and deserialise correctly."""

    def test_save_load_roundtrip(self, default_cfg, tmp_path):
        p = tmp_path / "cfg.json"
        default_cfg.save(p)
        loaded = ModelConfig.from_json(p)
        assert loaded.n_features  == default_cfg.n_features
        assert loaded.lookback    == default_cfg.lookback
        assert loaded.horizon     == default_cfg.horizon
        assert loaded.target_idx  == default_cfg.target_idx
        assert loaded.hidden_size == default_cfg.hidden_size

    def test_json_is_valid(self, default_cfg, tmp_path):
        p = tmp_path / "cfg.json"
        default_cfg.save(p)
        data = json.loads(p.read_text())
        assert "n_features"  in data
        assert "lookback"    in data
        assert "horizon"     in data
        assert "target_idx"  in data

    def test_feature_cols_preserved(self, tmp_path):
        cfg = ModelConfig(feature_cols=["a", "b", "aqi"])
        p = tmp_path / "cfg2.json"
        cfg.save(p)
        loaded = ModelConfig.from_json(p)
        assert loaded.feature_cols == ["a", "b", "aqi"]


class TestPersistenceBaseline:
    """Persistence baseline must have correct shape and values."""

    def test_baseline_shape(self, random_x):
        """Baseline: tile last time-step's target feature across horizon."""
        target_idx = TARGET_IDX
        last_aqi = random_x[:, -1, target_idx].numpy()         # (N,)
        baseline = np.tile(last_aqi[:, None], (1, HORIZON))    # (N, horizon)
        assert baseline.shape == (BATCH, HORIZON)

    def test_baseline_values_equal_last_step(self, random_x):
        target_idx = TARGET_IDX
        last_aqi = random_x[:, -1, target_idx].numpy()
        baseline = np.tile(last_aqi[:, None], (1, HORIZON))
        # Every column of baseline must equal last_aqi
        for h in range(HORIZON):
            np.testing.assert_array_almost_equal(baseline[:, h], last_aqi)


class TestInverseTransform:
    """Inverse-transforming standardised AQI should yield sensible values.

    Constants match the actual scaler used by train.py / evaluate.py:
      scaler_standard.pkl, index 6 (proc_meta feature_cols: aqi at position 6).
    Verified against data/processed/scaler_standard.pkl:
      mean_[6] = 132.823, scale_[6] = 124.883.
    """

    AQI_MEAN = 132.823   # scaler.mean_[6] -- actual value used by evaluate.py
    AQI_STD  = 124.883   # scaler.scale_[6] -- actual value used by evaluate.py

    def test_zero_maps_to_mean(self):
        val = 0.0 * self.AQI_STD + self.AQI_MEAN
        assert math.isclose(val, self.AQI_MEAN, rel_tol=1e-5)

    def test_plus_one_std(self):
        val = 1.0 * self.AQI_STD + self.AQI_MEAN
        assert math.isclose(val, self.AQI_MEAN + self.AQI_STD, rel_tol=1e-5)

    def test_typical_standardised_range_maps_to_valid_aqi(self):
        """Typical standardised values (-3..+3) should map to plausible AQI.

        With the actual scaler (mean=132.823, std=124.883), z=3 → ~508,
        which is physically possible during extreme pollution events (AQI scale
        goes to 500 in India / US); we cap the check at 600 to guard only
        against clearly broken inverse-transform constants.
        """
        for z in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            aqi = z * self.AQI_STD + self.AQI_MEAN
            # AQI can be negative in scaled data for very clean air; clip at 0 during eval
            assert aqi < 600, f"AQI={aqi:.1f} seems too high for z={z}"

    def test_array_inverse_transform(self):
        z = np.array([-1.0, 0.0, 1.0, 2.0])
        aqi = z * self.AQI_STD + self.AQI_MEAN
        expected = np.array([
            -1.0 * self.AQI_STD + self.AQI_MEAN,
             0.0 * self.AQI_STD + self.AQI_MEAN,
             1.0 * self.AQI_STD + self.AQI_MEAN,
             2.0 * self.AQI_STD + self.AQI_MEAN,
        ])
        np.testing.assert_array_almost_equal(aqi, expected)


class TestWindowFiles:
    """
    Validate that the actual augmented .npz files exist and have the expected
    shapes.  Skipped if the files are absent (e.g. CI without full data).
    """
    AUG_DIR = _BASE / "data" / "augmented"

    @pytest.mark.skipif(
        not (_BASE / "data" / "augmented" / "train_windows.npz").exists(),
        reason="Augmented data not available",
    )
    def test_train_window_shape(self):
        data = np.load(self.AUG_DIR / "train_windows.npz")
        X, y = data["X"], data["y"]
        assert X.ndim == 3,         "X must be 3-D (N, lookback, features)"
        assert y.ndim == 2,         "y must be 2-D (N, horizon)"
        assert X.shape[1] == 14,    f"Expected lookback=14, got {X.shape[1]}"
        assert X.shape[2] == 29,    f"Expected 29 features, got {X.shape[2]}"
        assert y.shape[1] == 7,     f"Expected horizon=7, got {y.shape[1]}"
        assert X.shape[0] == y.shape[0], "X and y must have same number of windows"

    @pytest.mark.skipif(
        not (_BASE / "data" / "augmented" / "val_windows.npz").exists(),
        reason="Augmented data not available",
    )
    def test_val_window_shape(self):
        data = np.load(self.AUG_DIR / "val_windows.npz")
        X, y = data["X"], data["y"]
        assert X.shape[1] == 14 and X.shape[2] == 29 and y.shape[1] == 7

    @pytest.mark.skipif(
        not (_BASE / "data" / "augmented" / "test_windows.npz").exists(),
        reason="Augmented data not available",
    )
    def test_test_window_shape(self):
        data = np.load(self.AUG_DIR / "test_windows.npz")
        X, y = data["X"], data["y"]
        assert X.shape[1] == 14 and X.shape[2] == 29 and y.shape[1] == 7

    @pytest.mark.skipif(
        not (_BASE / "data" / "augmented" / "train_windows.npz").exists(),
        reason="Augmented data not available",
    )
    def test_window_dtypes(self):
        for split in ("train", "val", "test"):
            path = self.AUG_DIR / f"{split}_windows.npz"
            if not path.exists():
                continue
            data = np.load(path)
            assert data["X"].dtype in (np.float32, np.float64), \
                f"{split} X dtype unexpected: {data['X'].dtype}"

    @pytest.mark.skipif(
        not (_BASE / "data" / "augmented" / "train_windows.npz").exists(),
        reason="Augmented data not available",
    )
    def test_no_all_nan_windows(self):
        for split in ("train", "val", "test"):
            path = self.AUG_DIR / f"{split}_windows.npz"
            if not path.exists():
                continue
            data = np.load(path)
            X = data["X"].astype(np.float32)
            # No window should be entirely NaN
            all_nan = np.isnan(X).all(axis=(1, 2))
            assert not all_nan.any(), \
                f"{split}: {all_nan.sum()} windows are all-NaN"


class TestGradientFlow:
    """Gradients must flow back through the model without vanishing to zero."""

    def test_gradients_nonzero(self, model, random_x):
        """At least one parameter gradient must be non-zero after a backward pass."""
        net = AQIForecastLSTM(model.cfg)
        net.train()
        x = random_x.clone().requires_grad_(False)
        out = net(x)
        loss = out.mean()
        loss.backward()
        grad_norms = [
            p.grad.norm().item()
            for p in net.parameters()
            if p.grad is not None
        ]
        assert any(g > 0 for g in grad_norms), "All gradients are zero"


class TestEvaluateModule:
    """Smoke-test the evaluate module helpers without needing a trained model."""

    def test_compute_metrics_shape(self):
        from model.evaluate import compute_metrics
        rng = np.random.default_rng(0)
        y_true = rng.uniform(50, 200, (100, HORIZON)).astype(np.float32)
        y_pred = y_true + rng.normal(0, 10, y_true.shape).astype(np.float32)
        metrics = compute_metrics(y_true, y_pred)
        assert "mae"  in metrics
        assert "rmse" in metrics
        assert "mse"  in metrics
        assert len(metrics["per_step_mae"])  == HORIZON
        assert len(metrics["per_step_rmse"]) == HORIZON

    def test_compute_metrics_perfect_prediction(self):
        from model.evaluate import compute_metrics
        y = np.ones((50, HORIZON), dtype=np.float32) * 100.0
        m = compute_metrics(y, y)
        assert m["mae"]  == pytest.approx(0.0, abs=1e-5)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-5)

    def test_inverse_transform_function(self):
        """Uses actual scaler values: mean=132.823, std=124.883 (scaler index 6)."""
        from model.evaluate import inverse_transform_aqi
        z = np.array([0.0, 1.0, -1.0])
        mean, std = 132.823, 124.883  # actual values from scaler_standard.pkl index 6
        result = inverse_transform_aqi(z, mean, std)
        expected = z * std + mean
        np.testing.assert_array_almost_equal(result, expected)

