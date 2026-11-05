"""
tests/test_preprocessing.py — Unit tests for preprocessing.preprocessor
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.preprocessor import (
    add_time_features,
    add_lag_features,
    add_rolling_features,
    handle_duplicate_timestamps,
    chronological_split,
    fit_and_scale,
    impute_missing,
    fit_imputer,
    apply_imputer,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────
def _daily_df(n: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date":    dates,
        "station_name": ["StA"] * n,
        "aqi":     np.random.default_rng(0).uniform(40, 200, n),
        "pm25":    np.random.default_rng(1).uniform(10, 150, n),
        "no2":     np.random.default_rng(2).uniform(5, 80, n),
    })


# ──────────────────────────────────────────────
# add_time_features
# ──────────────────────────────────────────────
class TestAddTimeFeatures:
    def test_columns_added(self):
        df = _daily_df(10)
        result = add_time_features(df, "date")
        for col in ["hour", "day_of_week", "month", "day_of_year",
                    "week_of_year", "is_weekend", "quarter"]:
            assert col in result.columns, f"Missing time feature: {col}"

    def test_month_range(self):
        df = _daily_df(365)
        result = add_time_features(df, "date")
        assert result["month"].between(1, 12).all()

    def test_day_of_week_range(self):
        df = _daily_df(14)
        result = add_time_features(df, "date")
        assert result["day_of_week"].between(0, 6).all()

    def test_is_weekend_binary(self):
        df = _daily_df(14)
        result = add_time_features(df, "date")
        assert set(result["is_weekend"].unique()).issubset({0, 1})

    def test_missing_dt_col_raises(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found"):
            add_time_features(df, "date")

    def test_row_count_preserved(self):
        df = _daily_df(20)
        result = add_time_features(df, "date")
        assert len(result) == len(df)


# ──────────────────────────────────────────────
# add_lag_features
# ──────────────────────────────────────────────
class TestAddLagFeatures:
    def test_lag1_values(self):
        df = _daily_df(10)
        result = add_lag_features(df, "aqi", lags=[1])
        # lag1 of row i == aqi of row i-1
        for i in range(1, len(result)):
            assert result["aqi_lag1"].iloc[i] == pytest.approx(result["aqi"].iloc[i - 1])

    def test_first_row_is_nan(self):
        df = _daily_df(10)
        result = add_lag_features(df, "aqi", lags=[1])
        assert pd.isna(result["aqi_lag1"].iloc[0])

    def test_multiple_lags(self):
        df = _daily_df(20)
        result = add_lag_features(df, "aqi", lags=[1, 3, 7])
        for lag in [1, 3, 7]:
            assert f"aqi_lag{lag}" in result.columns

    def test_group_lag_no_cross_leakage(self):
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=6, freq="D"),
            "station_name": ["A", "A", "A", "B", "B", "B"],
            "aqi": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        })
        result = add_lag_features(df, "aqi", lags=[1], group_col="station_name")
        # Station B's first row lag should be NaN (not bleed from station A)
        b_first = result[result["station_name"] == "B"].iloc[0]
        assert pd.isna(b_first["aqi_lag1"]), \
            "Lag should not bleed across station boundaries"


# ──────────────────────────────────────────────
# add_rolling_features
# ──────────────────────────────────────────────
class TestAddRollingFeatures:
    def test_cols_created(self):
        df = _daily_df(15)
        result = add_rolling_features(df, "aqi", windows=[3, 7])
        for col in ["aqi_roll3_mean", "aqi_roll3_std", "aqi_roll7_mean", "aqi_roll7_std"]:
            assert col in result.columns

    def test_no_future_leakage(self):
        """Rolling features should be lag-shifted (no row uses its own value)."""
        df = _daily_df(10).copy()
        df["aqi"] = np.arange(10.0)   # 0,1,2,...,9
        result = add_rolling_features(df, "aqi", windows=[3])
        # roll3_mean at row i uses rows [i-3..i-1] (shifted by 1)
        # At row 1: window is just [0], mean = 0.0
        assert result["aqi_roll3_mean"].iloc[1] == pytest.approx(0.0)


# ──────────────────────────────────────────────
# handle_duplicate_timestamps
# ──────────────────────────────────────────────
class TestHandleDuplicateTimestamps:
    def test_no_dups_unchanged(self):
        df = _daily_df(5)
        result = handle_duplicate_timestamps(df, "date", "station_name")
        assert len(result) == len(df)

    def test_duplicates_aggregated(self):
        df = _daily_df(5)
        df_dup = pd.concat([df, df.iloc[[2]]], ignore_index=True)  # dup row 2
        result = handle_duplicate_timestamps(df_dup, "date", "station_name")
        assert len(result) == 5, "Duplicate timestamp should be aggregated"

    def test_mean_aggregation(self):
        df = pd.DataFrame({
            "date":         [pd.Timestamp("2021-01-01")] * 2,
            "station_name": ["A", "A"],
            "aqi":          [100.0, 200.0],
        })
        result = handle_duplicate_timestamps(df, "date", "station_name", agg="mean")
        assert result["aqi"].iloc[0] == pytest.approx(150.0)


# ──────────────────────────────────────────────
# chronological_split
# ──────────────────────────────────────────────
class TestChronologicalSplit:
    def test_sizes_sum_to_total(self):
        df = _daily_df(100)
        train, val, test = chronological_split(df, 0.70, 0.15)
        assert len(train) + len(val) + len(test) == 100

    def test_no_overlap(self):
        df = _daily_df(100)
        df["idx"] = range(100)
        train, val, test = chronological_split(df)
        train_idx = set(train["idx"])
        val_idx   = set(val["idx"])
        test_idx  = set(test["idx"])
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_chronological_order(self):
        df = _daily_df(100)
        train, val, test = chronological_split(df)
        assert train["date"].max() <= val["date"].min(), "Train must precede val"
        assert val["date"].max() <= test["date"].min(), "Val must precede test"

    def test_approximate_ratios(self):
        df = _daily_df(200)
        train, val, test = chronological_split(df, 0.70, 0.15)
        assert abs(len(train) / 200 - 0.70) < 0.02
        assert abs(len(val)   / 200 - 0.15) < 0.02


# ──────────────────────────────────────────────
# fit_and_scale
# ──────────────────────────────────────────────
class TestFitAndScale:
    def _make_splits(self, n_train=50, n_val=15, n_test=15):
        rng = np.random.default_rng(42)
        def _frame(n):
            return pd.DataFrame({"aqi": rng.uniform(40, 200, n),
                                 "pm25": rng.uniform(10, 150, n)})
        return _frame(n_train), _frame(n_val), _frame(n_test)

    def test_scaled_train_mean_approx_zero(self):
        train, val, test = self._make_splits()
        tr, va, te, sc = fit_and_scale(train, val, test, ["aqi", "pm25"], "standard")
        assert abs(tr["aqi"].mean()) < 0.2, "Scaled train mean should be ≈ 0"

    def test_val_test_not_refit(self):
        """Val / test should be scaled with train stats, not their own."""
        train, val, test = self._make_splits()
        # Make val mean very different from train
        val["aqi"] += 5000
        tr, va, te, sc = fit_and_scale(train, val, test, ["aqi"], "standard")
        # If scaler was refit on val, val mean would ≈ 0; if correctly used
        # train scaler, val mean should be >> 0
        assert abs(va["aqi"].mean()) > 1.0, \
            "Val mean should deviate from 0 (scaled with train stats)"

    def test_scaler_saved(self, tmp_path):
        import pickle
        train, val, test = self._make_splits()
        sc_path = tmp_path / "scaler.pkl"
        fit_and_scale(train, val, test, ["aqi"], "standard", save_path=sc_path)
        assert sc_path.exists()
        with open(sc_path, "rb") as f:
            sc = pickle.load(f)
        assert hasattr(sc, "transform")

    def test_minmax_range(self):
        train, val, test = self._make_splits()
        tr, va, te, sc = fit_and_scale(train, val, test, ["aqi", "pm25"], "minmax")
        assert tr["aqi"].min() >= -0.01
        assert tr["aqi"].max() <= 1.01


# ──────────────────────────────────────────────
# impute_missing
# ──────────────────────────────────────────────
class TestImputeMissing:
    def test_no_nans_after_impute(self):
        df = _daily_df(20)
        df.loc[5, "aqi"] = np.nan
        df.loc[10, "pm25"] = np.nan
        result = impute_missing(df, ["aqi", "pm25"])
        assert result["aqi"].isna().sum() == 0
        assert result["pm25"].isna().sum() == 0

    def test_station_group_impute(self):
        df = pd.DataFrame({
            "date":         pd.date_range("2021-01-01", periods=6, freq="D"),
            "station_name": ["A", "A", "A", "B", "B", "B"],
            "aqi":          [100.0, np.nan, 120.0, 50.0, np.nan, 70.0],
        })
        result = impute_missing(df, ["aqi"], group_col="station_name")
        assert result["aqi"].isna().sum() == 0


# ──────────────────────────────────────────────
# chronological_split — multi-station (task item 4)
# ──────────────────────────────────────────────
class TestChronologicalSplitMultiStation:
    """Test that global date-cutoff split keeps all stations in every split."""

    def _two_station_df(self):
        """
        Station A: 2020-01-01 → 2022-12-31 (1096 days)
        Station B: 2021-06-01 → 2023-12-31 (944 days)
        Together: 2040 rows with overlapping but different ranges.
        """
        dates_a = pd.date_range("2020-01-01", "2022-12-31", freq="D")
        dates_b = pd.date_range("2021-06-01", "2023-12-31", freq="D")
        df_a = pd.DataFrame({"date": dates_a, "station_name": "A",
                              "aqi": np.random.default_rng(0).uniform(40, 200, len(dates_a))})
        df_b = pd.DataFrame({"date": dates_b, "station_name": "B",
                              "aqi": np.random.default_rng(1).uniform(40, 200, len(dates_b))})
        return pd.concat([df_a, df_b], ignore_index=True)

    def test_all_stations_appear_in_train(self):
        """With a global date cutoff both stations should appear in train."""
        df = self._two_station_df()
        train, val, test = chronological_split(df, 0.70, 0.15)
        assert "A" in train["station_name"].values
        assert "B" in train["station_name"].values

    def test_all_stations_appear_in_val(self):
        """Both stations should appear in val (they have rows in that period)."""
        df = self._two_station_df()
        train, val, test = chronological_split(df, 0.70, 0.15)
        assert "A" in val["station_name"].values
        assert "B" in val["station_name"].values

    def test_no_temporal_overlap_per_station(self):
        """For each individual station: train.max <= val.min and val.max <= test.min."""
        df = self._two_station_df()
        train, val, test = chronological_split(df, 0.70, 0.15)
        for station in ["A", "B"]:
            tr_dates  = train[train["station_name"] == station]["date"]
            val_dates = val[val["station_name"] == station]["date"]
            te_dates  = test[test["station_name"] == station]["date"]
            if len(tr_dates) > 0 and len(val_dates) > 0:
                assert tr_dates.max() <= val_dates.min(), \
                    f"Station {station}: train dates overlap with val"
            if len(val_dates) > 0 and len(te_dates) > 0:
                assert val_dates.max() <= te_dates.min(), \
                    f"Station {station}: val dates overlap with test"

    def test_global_cutoff_not_per_station_slice(self):
        """
        The old buggy implementation sorted by (station, date) and then did
        iloc[:int(0.7*n)], which put all of station A in train and all of B in
        val/test.  Verify this doesn't happen with a scenario where station A
        and B both have rows in the train AND val windows.

        Station A: 2020-01-01 → 2022-12-31
        Station B: 2021-06-01 → 2023-12-31
        Combined 2040 rows; train cutoff ≈ 2022-08-30, val cutoff ≈ 2023-02-28.
        Station A ends 2022-12-31 so it has rows in both train AND val splits.
        Station B starts 2021-06-01 so it also has rows in both train AND val.

        The old bug would put ALL of A in train and ALL of B in val+test,
        meaning train contains only A and val contains only B.
        The fix: both A and B appear in train AND val.
        """
        df = self._two_station_df()
        train, val, test = chronological_split(df, 0.70, 0.15)

        # Both stations must appear in TRAIN
        assert "A" in train["station_name"].values, "A missing from train"
        assert "B" in train["station_name"].values, "B missing from train"

        # Both stations must appear in VAL (A ends 2022-12-31, val starts ~2022-09,
        # so A has ~4 months in val; B starts 2021-06 and runs to 2023-12-31)
        assert "A" in val["station_name"].values, \
            "A missing from val (old row-slice bug: all A rows crammed into train?)"
        assert "B" in val["station_name"].values, \
            "B missing from val (old row-slice bug: all B rows placed in val/test?)"


# ──────────────────────────────────────────────
# fit_imputer / apply_imputer — leakage prevention (task item 5)
# ──────────────────────────────────────────────
class TestFitApplyImputer:
    """Verify that imputer statistics are derived solely from training data."""

    def _make_splits_with_inflated_val(self):
        """
        Train: aqi values around 100.
        Val:   ALL aqi values NaN for station S2 (new station not in train).

        When the entire val column for a station is NaN, ffill and bfill both
        produce nothing, so apply_imputer falls back to the global training
        median from `stats`.  This verifies no re-fitting on val data occurs.
        """
        rng = np.random.default_rng(42)
        train = pd.DataFrame({
            "date":         pd.date_range("2021-01-01", periods=30, freq="D"),
            "station_name": "S1",
            "aqi":          rng.uniform(90, 110, 30),
        })
        # S2 is entirely absent from train; all its val values are NaN
        val = pd.DataFrame({
            "date":         pd.date_range("2021-01-31", periods=5, freq="D"),
            "station_name": "S2",
            "aqi":          [np.nan] * 5,
        })
        return train, val

    def test_val_nan_filled_with_train_median(self):
        """
        A station that is entirely NaN in val and absent from train must be
        filled with the GLOBAL training median (from `stats['aqi']`), not by
        re-computing statistics on val.

        The training global median ≈ 100; if the imputer refitted on val it
        would find median=NaN and fall back to 0 — clearly wrong.
        """
        train, val = self._make_splits_with_inflated_val()
        stats = fit_imputer(train, ["aqi"], group_col="station_name")
        val_imputed = apply_imputer(val, ["aqi"], stats, group_col="station_name")

        train_median = train["aqi"].median()  # ≈ 100

        # All values should now be filled; and since S2 is absent from train,
        # they must be filled with the global training median
        assert val_imputed["aqi"].isna().sum() == 0, "NaNs remain after imputation"
        np.testing.assert_allclose(
            val_imputed["aqi"].values, train_median, rtol=0.01,
            err_msg=(
                f"Val NaN rows for unknown station should be filled with "
                f"global training median ({train_median:.1f}), got "
                f"{val_imputed['aqi'].values}"
            )
        )

    def test_fit_imputer_uses_only_train_rows(self):
        """Global median in stats dict should match training data's median."""
        rng = np.random.default_rng(0)
        train = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=20, freq="D"),
            "aqi":  rng.uniform(50, 100, 20),
        })
        stats = fit_imputer(train, ["aqi"])
        expected_median = float(train["aqi"].median())
        assert abs(stats["aqi"] - expected_median) < 1e-9, \
            "fit_imputer global median should match training data exactly"

    def test_apply_imputer_no_refitting(self):
        """apply_imputer must not use the split's own data to derive fill values."""
        train = pd.DataFrame({
            "date":         pd.date_range("2021-01-01", periods=10, freq="D"),
            "station_name": "S1",
            "aqi":          [1.0] * 10,    # train median = 1.0
        })
        test = pd.DataFrame({
            "date":         pd.date_range("2021-01-11", periods=5, freq="D"),
            "station_name": "S1",
            "aqi":          [np.nan] * 5,  # all NaN in test
        })
        stats = fit_imputer(train, ["aqi"], group_col="station_name")
        test_imputed = apply_imputer(test, ["aqi"], stats, group_col="station_name")

        # If apply_imputer refitted on test it would find no median (all NaN → 0);
        # correct behaviour fills with the training median (1.0)
        np.testing.assert_allclose(
            test_imputed["aqi"].values, 1.0,
            err_msg="apply_imputer should fill with training median (1.0), "
                    "not test's own median (nan→0)"
        )

    def test_no_nans_remain_after_apply(self):
        rng = np.random.default_rng(7)
        train = pd.DataFrame({"aqi": rng.uniform(40, 200, 20)})
        val   = pd.DataFrame({"aqi": [np.nan, 50.0, np.nan, 80.0]})
        stats = fit_imputer(train, ["aqi"])
        result = apply_imputer(val, ["aqi"], stats)
        assert result["aqi"].isna().sum() == 0


# ──────────────────────────────────────────────
# Lag and rolling features — boundary conditions (task item 6)
# ──────────────────────────────────────────────
class TestLagRollingBoundaries:
    """Ensure lag/rolling features do not introduce future information."""

    def _sorted_df(self, n=20, station="A"):
        rng = np.random.default_rng(99)
        return pd.DataFrame({
            "date":         pd.date_range("2021-01-01", periods=n, freq="D"),
            "station_name": station,
            "aqi":          rng.uniform(40, 200, n),
        })

    def test_lag_never_uses_future(self):
        """lag-k of row i must equal the value at row i-k, never row i or later."""
        df = self._sorted_df(30)
        df["aqi"] = np.arange(30.0)   # deterministic sequence 0,1,...,29
        result = add_lag_features(df, "aqi", lags=[1, 3])
        for i in range(3, 30):
            assert result["aqi_lag1"].iloc[i] == pytest.approx(i - 1)
            assert result["aqi_lag3"].iloc[i] == pytest.approx(i - 3)

    def test_rolling_never_uses_current_value(self):
        """
        Rolling mean at row i is computed from rows [i-w .. i-1] (shifted by 1).
        Row 0: no previous rows → NaN (or min_periods=1 gives the shifted value).
        Row 1 with window=3: only one previous value → mean of aqi[0].
        """
        df = self._sorted_df(10)
        df["aqi"] = np.arange(10.0)    # 0,1,...,9
        result = add_rolling_features(df, "aqi", windows=[3])
        # At row 1 the shifted window sees only [0] → mean = 0.0
        assert result["aqi_roll3_mean"].iloc[1] == pytest.approx(0.0)
        # At row 3 the shifted window sees [0,1,2] → mean = 1.0
        assert result["aqi_roll3_mean"].iloc[3] == pytest.approx(1.0)

    def test_lag_first_rows_are_nan(self):
        """The first k rows of a lag-k feature must be NaN."""
        df = self._sorted_df(15)
        result = add_lag_features(df, "aqi", lags=[3])
        assert result["aqi_lag3"].iloc[:3].isna().all(), \
            "First 3 rows of lag-3 feature must be NaN"
        assert not pd.isna(result["aqi_lag3"].iloc[3])

    def test_cross_station_lag_boundary(self):
        """
        When computing lags per station, station B's lag-1 at its first row
        must be NaN — not station A's last value.
        """
        df = pd.DataFrame({
            "date":         list(pd.date_range("2021-01-01", periods=5)) * 2,
            "station_name": ["A"] * 5 + ["B"] * 5,
            "aqi":          [10.0, 20.0, 30.0, 40.0, 50.0,
                             100.0, 110.0, 120.0, 130.0, 140.0],
        })
        result = add_lag_features(df, "aqi", lags=[1], group_col="station_name")
        b_first_lag = result[result["station_name"] == "B"]["aqi_lag1"].iloc[0]
        assert pd.isna(b_first_lag), \
            "Station B's first lag value must be NaN, not A's last value"

