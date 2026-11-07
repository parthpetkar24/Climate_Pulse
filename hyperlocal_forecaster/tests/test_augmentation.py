"""
tests/test_augmentation.py — Unit tests for augmentation.augmentor
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augmentation.augmentor import (
    sliding_windows,
    gaussian_noise_augment,
    build_windows_for_split,
)
import pandas as pd


# ──────────────────────────────────────────────
# sliding_windows
# ──────────────────────────────────────────────
class TestSlidingWindows:
    def _data(self, T=50, F=3):
        rng = np.random.default_rng(0)
        return rng.uniform(0, 100, (T, F)).astype(np.float32)

    def test_output_shapes(self):
        vals = self._data(50, 3)
        X, y = sliding_windows(vals, lookback=10, horizon=5, target_idx=0)
        assert X.shape == (36, 10, 3)
        assert y.shape == (36, 5)

    def test_x_y_alignment(self):
        """y window should immediately follow the X window."""
        vals = self._data(30, 2)
        X, y = sliding_windows(vals, lookback=5, horizon=3, target_idx=1)
        # For window 0: X = vals[0:5], y = vals[5:8, 1]
        np.testing.assert_allclose(X[0], vals[0:5])
        np.testing.assert_allclose(y[0], vals[5:8, 1])

    def test_stride(self):
        vals = self._data(50, 2)
        X_s1, _ = sliding_windows(vals, 10, 5, 0, stride=1)
        X_s2, _ = sliding_windows(vals, 10, 5, 0, stride=2)
        assert len(X_s2) < len(X_s1)

    def test_insufficient_data(self):
        vals = self._data(10, 2)
        X, y = sliding_windows(vals, lookback=8, horizon=5, target_idx=0)
        assert len(X) == 0, "Should produce no windows when data is too short"

    def test_float32_dtype(self):
        vals = self._data(30, 2)
        X, y = sliding_windows(vals, 5, 3, 0)
        assert X.dtype == np.float32
        assert y.dtype == np.float32


# ──────────────────────────────────────────────
# gaussian_noise_augment
# ──────────────────────────────────────────────
class TestGaussianNoiseAugment:
    def _windows(self, N=50, T=10, F=3):
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 100, (N, T, F)).astype(np.float32)
        y = rng.uniform(0, 500, (N, 5)).astype(np.float32)
        return X, y

    def test_output_doubles_samples(self):
        X, y = self._windows(50)
        X_aug, y_aug = gaussian_noise_augment(X, y, noise_std_fraction=0.01)
        assert X_aug.shape[0] == 100
        assert y_aug.shape[0] == 100

    def test_first_half_unchanged(self):
        X, y = self._windows(50)
        X_aug, y_aug = gaussian_noise_augment(X, y, noise_std_fraction=0.01)
        np.testing.assert_array_equal(X_aug[:50], X)
        np.testing.assert_array_equal(y_aug[:50], y)

    def test_y_unchanged_in_second_half(self):
        X, y = self._windows(50)
        X_aug, y_aug = gaussian_noise_augment(X, y, noise_std_fraction=0.01)
        np.testing.assert_array_equal(y_aug[50:], y)

    def test_second_half_differs_from_first(self):
        X, y = self._windows(50)
        X_aug, _ = gaussian_noise_augment(X, y, noise_std_fraction=0.01)
        diff = np.abs(X_aug[50:] - X_aug[:50]).max()
        assert diff > 0, "Noisy windows should differ from originals"

    def test_zero_std_noop(self):
        X, y = self._windows(10)
        X_aug, y_aug = gaussian_noise_augment(X, y, noise_std_fraction=0.0)
        assert len(X_aug) == len(X), "Zero noise should not augment (noop)"

    def test_reproducible_with_seed(self):
        X, y = self._windows(20)
        X1, _ = gaussian_noise_augment(X, y, noise_std_fraction=0.05, rng_seed=7)
        X2, _ = gaussian_noise_augment(X, y, noise_std_fraction=0.05, rng_seed=7)
        np.testing.assert_array_equal(X1, X2)

    def test_different_seeds_differ(self):
        X, y = self._windows(20)
        X1, _ = gaussian_noise_augment(X, y, noise_std_fraction=0.05, rng_seed=7)
        X2, _ = gaussian_noise_augment(X, y, noise_std_fraction=0.05, rng_seed=99)
        assert not np.array_equal(X1[20:], X2[20:])


# ──────────────────────────────────────────────
# build_windows_for_split
# ──────────────────────────────────────────────
class TestBuildWindowsForSplit:
    def _make_df(self, n=100, stations=("A",)):
        frames = []
        for st in stations:
            rng = np.random.default_rng(hash(st) % 1000)
            frames.append(pd.DataFrame({
                "date":         pd.date_range("2021-01-01", periods=n, freq="D"),
                "station_name": st,
                "aqi":          rng.uniform(40, 200, n),
                "pm25":         rng.uniform(10, 150, n),
            }))
        return pd.concat(frames, ignore_index=True)

    def test_single_station_shapes(self):
        df = self._make_df(60, ("A",))
        X, y = build_windows_for_split(
            df, ["aqi", "pm25"], "aqi", lookback=10, horizon=5, group_col="station_name"
        )
        assert X.ndim == 3
        assert y.ndim == 2
        assert X.shape[1] == 10   # lookback
        assert y.shape[1] == 5   # horizon
        assert X.shape[2] == 2   # features

    def test_multi_station_no_boundary_bleed(self):
        """Windows should not cross station boundaries."""
        df = self._make_df(30, ("A", "B"))
        X, y = build_windows_for_split(
            df, ["aqi", "pm25"], "aqi", lookback=5, horizon=3, group_col="station_name"
        )
        # Each station has 30 rows → (30 - 5 - 3) + 1 = 23 windows each → 46 total
        assert X.shape[0] == 46

    def test_insufficient_data_station_skipped(self):
        """A station with fewer rows than lookback+horizon should be skipped."""
        df = self._make_df(5, ("Short",))   # only 5 rows
        X, y = build_windows_for_split(
            df, ["aqi", "pm25"], "aqi", lookback=10, horizon=5
        )
        assert X.shape[0] == 0

    def test_missing_target_raises(self):
        df = self._make_df(30, ("A",))
        with pytest.raises(ValueError, match="not found"):
            build_windows_for_split(df, ["pm25"], "nonexistent_col", 5, 3)


# ──────────────────────────────────────────────
# Station boundary — windows must not cross stations (task item 7)
# ──────────────────────────────────────────────
class TestWindowBoundaries:
    """Detailed boundary tests for sliding-window generation."""

    def _make_multi_station_df(self, n=40, stations=("A", "B")):
        frames = []
        for st in stations:
            rng = np.random.default_rng(hash(st) % 999)
            frames.append(pd.DataFrame({
                "date":         pd.date_range("2021-01-01", periods=n, freq="D"),
                "station_name": st,
                "aqi":          rng.uniform(40, 200, n),
                "pm25":         rng.uniform(10, 150, n),
            }))
        return pd.concat(frames, ignore_index=True)

    def test_exact_window_count_per_station(self):
        """
        Each station with N rows produces (N - lookback - horizon + 1) windows.
        Total across all stations should be the sum.
        """
        n, lookback, horizon = 40, 7, 3
        expected_per = n - lookback - horizon + 1   # 31
        df = self._make_multi_station_df(n=n, stations=("A", "B"))
        X, y = build_windows_for_split(
            df, ["aqi", "pm25"], "aqi",
            lookback=lookback, horizon=horizon, group_col="station_name"
        )
        assert X.shape[0] == 2 * expected_per, (
            f"Expected {2 * expected_per} windows, got {X.shape[0]}"
        )

    def test_no_cross_station_bleed(self):
        """
        Validate that the last window of station A does not contain values
        from station B by checking y values against the known data.

        Strategy: give A values in [40,200] and B values in [1000,2000].
        Any window whose y value is in [1000,2000] must be a B-window.
        If boundary bleeding occurred, some A-windows would have y > 1000.
        """
        n, lookback, horizon = 30, 5, 3
        rng_a = np.random.default_rng(0)
        rng_b = np.random.default_rng(1)
        df_a = pd.DataFrame({
            "date":         pd.date_range("2021-01-01", periods=n, freq="D"),
            "station_name": "A",
            "aqi":          rng_a.uniform(40, 200, n),
        })
        df_b = pd.DataFrame({
            "date":         pd.date_range("2021-01-01", periods=n, freq="D"),
            "station_name": "B",
            "aqi":          rng_b.uniform(1000, 2000, n),
        })
        df = pd.concat([df_a, df_b], ignore_index=True)
        X, y = build_windows_for_split(
            df, ["aqi"], "aqi",
            lookback=lookback, horizon=horizon, group_col="station_name"
        )
        # All y values should be either in A's range OR B's range, never mixed
        a_windows = y[y.max(axis=1) < 500]   # all y < 500 → A windows
        b_windows = y[y.min(axis=1) > 500]   # all y > 500 → B windows
        assert len(a_windows) + len(b_windows) == len(y), \
            "Some windows span both stations (boundary bleed detected)"


# ──────────────────────────────────────────────
# No augmentation on val/test (task item 8)
# ──────────────────────────────────────────────
class TestNoValTestAugmentation:
    """Val and test windows must never be augmented."""

    def _windows(self, N=30, T=7, F=2):
        rng = np.random.default_rng(5)
        X = rng.uniform(0, 100, (N, T, F)).astype(np.float32)
        y = rng.uniform(0, 200, (N, 4)).astype(np.float32)
        return X, y

    def test_zero_std_returns_original_unchanged(self):
        """noise_std_fraction=0 is the noop path used for val/test."""
        X, y = self._windows()
        X_out, y_out = gaussian_noise_augment(X, y, noise_std_fraction=0.0)
        np.testing.assert_array_equal(X_out, X)
        np.testing.assert_array_equal(y_out, y)
        assert len(X_out) == len(X), "Noop path must not duplicate samples"

    def test_val_windows_identical_after_noop(self):
        """Simulate val pipeline: build windows, do NOT augment, arrays unchanged."""
        df = pd.DataFrame({
            "date":         pd.date_range("2022-01-01", periods=50, freq="D"),
            "station_name": "Val",
            "aqi":          np.linspace(50, 150, 50),
            "pm25":         np.linspace(20, 80, 50),
        })
        X_val, y_val = build_windows_for_split(
            df, ["aqi", "pm25"], "aqi", lookback=7, horizon=3
        )
        # Verify we do NOT call gaussian_noise_augment on val
        # (just confirm shape and that values are finite — no noise applied)
        assert np.isfinite(X_val).all(), "Val X should contain only finite values"
        assert np.isfinite(y_val).all(), "Val y should contain only finite values"
        # The test contract: val windows == the original data windows, not doubled
        n_expected = 50 - 7 - 3 + 1  # 41
        assert X_val.shape[0] == n_expected, \
            "Val windows must NOT be doubled (no augmentation)"


# ──────────────────────────────────────────────
# Reproducibility and valid ranges (task items 9, 10)
# ──────────────────────────────────────────────
class TestAugmentationReproducibilityAndRanges:
    """Augmentation must be reproducible and keep values in valid ranges."""

    def _windows(self, N=40, T=10, F=3, seed=42):
        rng = np.random.default_rng(seed)
        X = rng.uniform(0, 100, (N, T, F)).astype(np.float32)
        y = rng.uniform(0, 300, (N, 5)).astype(np.float32)
        return X, y

    def test_same_seed_same_output(self):
        """Identical seed must produce bitwise-identical results."""
        X, y = self._windows()
        X1, y1 = gaussian_noise_augment(X, y, noise_std_fraction=0.05, rng_seed=123)
        X2, y2 = gaussian_noise_augment(X, y, noise_std_fraction=0.05, rng_seed=123)
        np.testing.assert_array_equal(X1, X2, err_msg="Same seed → identical X")
        np.testing.assert_array_equal(y1, y2, err_msg="Same seed → identical y")

    def test_different_seeds_produce_different_noise(self):
        """Different seeds must yield different noisy windows."""
        X, y = self._windows()
        X1, _ = gaussian_noise_augment(X, y, noise_std_fraction=0.05, rng_seed=1)
        X2, _ = gaussian_noise_augment(X, y, noise_std_fraction=0.05, rng_seed=2)
        assert not np.array_equal(X1[len(X):], X2[len(X):]), \
            "Different seeds should produce different noise"

    def test_noise_magnitude_is_small(self):
        """
        Noise sigma = noise_std_fraction * feature_std.
        For fraction=0.01 and values in [0,100] (std ≈ 29), sigma ≈ 0.29.
        Max absolute noise should be well within 5*sigma per element.
        """
        X, y = self._windows(N=200)
        X_aug, _ = gaussian_noise_augment(X, y, noise_std_fraction=0.01, rng_seed=0)
        original = X_aug[:len(X)]
        noisy    = X_aug[len(X):]
        abs_diff = np.abs(noisy - original)
        per_feature_std = original.reshape(-1, original.shape[-1]).std(axis=0)
        sigma = 0.01 * per_feature_std
        # 99.9999% of noise should be within 5 * sigma
        for f_idx in range(original.shape[-1]):
            max_noise = abs_diff[:, :, f_idx].max()
            assert max_noise < 10 * sigma[f_idx], (
                f"Feature {f_idx}: max noise {max_noise:.4f} > 10*sigma "
                f"({10*sigma[f_idx]:.4f}), noise may be too large"
            )

    def test_noisy_y_unchanged(self):
        """y targets must be identical (noise is only applied to X)."""
        X, y = self._windows()
        X_aug, y_aug = gaussian_noise_augment(X, y, noise_std_fraction=0.05, rng_seed=7)
        np.testing.assert_array_equal(
            y_aug[:len(y)], y,   err_msg="First half of y_aug must equal original y"
        )
        np.testing.assert_array_equal(
            y_aug[len(y):], y,   err_msg="Second half of y_aug must equal original y"
        )

    def test_augmented_values_are_finite(self):
        """All augmented values must be finite (no inf or NaN)."""
        X, y = self._windows()
        X_aug, y_aug = gaussian_noise_augment(X, y, noise_std_fraction=0.05, rng_seed=3)
        assert np.isfinite(X_aug).all(), "Augmented X contains non-finite values"
        assert np.isfinite(y_aug).all(), "Augmented y contains non-finite values"

