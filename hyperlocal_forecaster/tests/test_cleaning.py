"""
tests/test_cleaning.py — Unit tests for cleaning.cleaner
"""

import io
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure hyperlocal_forecaster/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cleaning.cleaner import (
    _range_flag,
    _all_zero_measurement_row,
    clean_daily_aqi,
    CleaningReport,
    POLLUTANT_RANGES,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────
def _make_logger() -> logging.Logger:
    logger = logging.getLogger("test_cleaning")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def _sample_daily_df() -> pd.DataFrame:
    """Minimal valid daily AQI DataFrame."""
    return pd.DataFrame({
        "city":             ["Pune", "Pune", "Pune", "Pune"],
        "station_name":     ["StationA"] * 4,
        "station_id":       [1, 1, 1, 1],
        "co":               [np.nan, np.nan, np.nan, np.nan],
        "date":             ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
        "no2":              [30.0, 25.0, None, 28.0],
        "o3":               [45.0, 50.0, 40.0, 55.0],
        "pm10":             [80.0, 90.0, 85.0, 95.0],
        "pm25":             [35.0, 40.0, 38.0, 42.0],
        "so2":              [5.0, 6.0, 5.5, 7.0],
        "sub_pm25":         [50.0, 60.0, 55.0, 65.0],
        "sub_pm10":         [60.0, 70.0, 65.0, 75.0],
        "sub_no2":          [30.0, 25.0, None, 28.0],
        "sub_o3":           [40.0, 45.0, 42.0, 48.0],
        "sub_co":           [np.nan, np.nan, np.nan, np.nan],
        "sub_so2":          [5.0, 6.0, 5.5, 7.0],
        "aqi":              [60.0, 70.0, 65.0, 75.0],
        "dominant_pollutant": ["pm25", "pm25", "pm25", "pm25"],
    })


# ──────────────────────────────────────────────
# _range_flag
# ──────────────────────────────────────────────
class TestRangeFlag:
    def test_all_valid(self):
        s = pd.Series([10.0, 50.0, 200.0])
        result = _range_flag(s, "pm25")
        assert not result.any(), "All valid values should produce no flags"

    def test_negative_flagged(self):
        s = pd.Series([-1.0, 10.0, 50.0])
        result = _range_flag(s, "pm25")
        assert result.iloc[0], "Negative PM2.5 should be flagged"
        assert not result.iloc[1]
        assert not result.iloc[2]

    def test_above_max_flagged(self):
        lo, hi = POLLUTANT_RANGES["pm25"]
        s = pd.Series([hi + 1.0, 100.0])
        result = _range_flag(s, "pm25")
        assert result.iloc[0], "Value above max should be flagged"
        assert not result.iloc[1]

    def test_nan_not_flagged(self):
        s = pd.Series([np.nan, 10.0])
        result = _range_flag(s, "pm25")
        assert not result.iloc[0], "NaN should not produce an out-of-range flag"

    def test_unknown_col_returns_false(self):
        s = pd.Series([9999.0, -999.0])
        result = _range_flag(s, "unknown_pollutant")
        assert not result.any(), "Unknown column should return all-False"


# ──────────────────────────────────────────────
# _all_zero_measurement_row
# ──────────────────────────────────────────────
class TestAllZeroRow:
    def test_zero_row_flagged(self):
        df = pd.DataFrame({"pm25": [0.0, 1.0], "pm10": [0.0, 2.0], "no2": [0.0, 3.0]})
        result = _all_zero_measurement_row(df, ["pm25", "pm10", "no2"])
        assert result.iloc[0], "Row with all zeros should be flagged"
        assert not result.iloc[1]

    def test_nan_row_not_flagged(self):
        df = pd.DataFrame({"pm25": [np.nan, np.nan], "pm10": [np.nan, 1.0]})
        result = _all_zero_measurement_row(df, ["pm25", "pm10"])
        assert not result.iloc[0], "All-NaN row should NOT be flagged as all-zero"

    def test_mixed_zero_nan_not_flagged(self):
        df = pd.DataFrame({"pm25": [0.0, 1.0], "pm10": [np.nan, 2.0]})
        result = _all_zero_measurement_row(df, ["pm25", "pm10"])
        # Row 0: pm25=0, pm10=NaN → not ALL non-null zero (pm10 is null)
        assert not result.iloc[0], "Row with some NaN and some 0 should NOT be all-zero flagged"

    def test_empty_cols(self):
        df = pd.DataFrame({"irrelevant": [1, 2]})
        result = _all_zero_measurement_row(df, ["pm25"])
        assert not result.any()


# ──────────────────────────────────────────────
# clean_daily_aqi (uses tmp_path fixture)
# ──────────────────────────────────────────────
class TestCleanDailyAQI:
    def _write_csv(self, df: pd.DataFrame, path: Path) -> Path:
        path.write_text(df.to_csv(index=False), encoding="utf-8")
        return path

    def test_basic_clean_produces_output(self, tmp_path):
        df = _sample_daily_df()
        input_csv = self._write_csv(df, tmp_path / "input.csv")
        output_csv = tmp_path / "output.csv"
        report = CleaningReport()
        logger = _make_logger()

        result = clean_daily_aqi(input_csv, output_csv, report, logger, save=True)

        assert not result.empty, "Cleaned result should not be empty"
        assert output_csv.exists(), "Output CSV should be created"
        loaded = pd.read_csv(output_csv)
        assert len(loaded) == len(df), "Row count should be preserved (no deletion)"

    def test_duplicate_removal(self, tmp_path):
        df = _sample_daily_df()
        df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # add 1 exact dup
        input_csv = self._write_csv(df_dup, tmp_path / "dup_input.csv")
        output_csv = tmp_path / "dup_output.csv"
        report = CleaningReport()

        result = clean_daily_aqi(input_csv, output_csv, report, _make_logger(), save=False)
        assert len(result) == len(df), "One duplicate should be removed"

    def test_date_parsed_to_datetime(self, tmp_path):
        df = _sample_daily_df()
        input_csv = self._write_csv(df, tmp_path / "dates.csv")
        output_csv = tmp_path / "dates_out.csv"
        report = CleaningReport()

        result = clean_daily_aqi(input_csv, output_csv, report, _make_logger(), save=False)
        assert pd.api.types.is_datetime64_any_dtype(result["date"]), \
            "'date' column should be datetime64 after cleaning"

    def test_none_strings_replaced(self, tmp_path):
        df = _sample_daily_df()
        # Introduce a literal "None" string in a numeric column
        df["pm25"] = df["pm25"].astype(str)
        df.loc[2, "pm25"] = "None"
        input_csv = self._write_csv(df, tmp_path / "none_str.csv")
        output_csv = tmp_path / "none_str_out.csv"
        report = CleaningReport()

        result = clean_daily_aqi(input_csv, output_csv, report, _make_logger(), save=False)
        assert pd.isna(result.loc[result["pm25"].isna(), "pm25"]).all(), \
            "'None' string should be replaced with NaN"

    def test_out_of_range_flagged_not_deleted(self, tmp_path):
        df = _sample_daily_df()
        df.loc[0, "pm25"] = 9999.0   # clearly out of range
        input_csv = self._write_csv(df, tmp_path / "oor.csv")
        output_csv = tmp_path / "oor_out.csv"
        report = CleaningReport()

        result = clean_daily_aqi(input_csv, output_csv, report, _make_logger(), save=False)
        assert len(result) == len(df), "Out-of-range rows should NOT be deleted"
        assert "flag_out_of_range" in result.columns
        assert result.loc[result["pm25"] == 9999.0, "flag_out_of_range"].all()

    def test_missing_file_returns_empty(self, tmp_path):
        report = CleaningReport()
        result = clean_daily_aqi(
            tmp_path / "nonexistent.csv",
            tmp_path / "out.csv",
            report, _make_logger(), save=False,
        )
        assert result.empty, "Missing input file should return empty DataFrame"

    def test_sorted_chronologically(self, tmp_path):
        df = _sample_daily_df()
        df_shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
        input_csv = self._write_csv(df_shuffled, tmp_path / "shuffled.csv")
        output_csv = tmp_path / "sorted_out.csv"
        report = CleaningReport()

        result = clean_daily_aqi(input_csv, output_csv, report, _make_logger(), save=False)
        dates = pd.to_datetime(result["date"])
        assert (dates.diff().dropna() >= pd.Timedelta(0)).all(), \
            "Dates should be sorted ascending after cleaning"


# ──────────────────────────────────────────────
# CleaningReport
# ──────────────────────────────────────────────
class TestCleaningReport:
    def test_save_creates_file(self, tmp_path):
        r = CleaningReport()
        r.add("Test Section", ["Line 1", "Line 2"])
        out = tmp_path / "report.txt"
        r.save(out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "Test Section" in content
        assert "Line 1" in content

    def test_report_contains_row_counts(self, tmp_path):
        """Report text must mention raw and cleaned row counts."""
        r = CleaningReport()
        r.add("Row Counts", ["Raw rows: 100", "Cleaned rows: 95"])
        out = tmp_path / "counts.txt"
        r.save(out)
        content = out.read_text(encoding="utf-8")
        assert "Raw rows: 100" in content
        assert "Cleaned rows: 95" in content

    def test_multiple_sections_preserved(self, tmp_path):
        """All sections added to a report should appear in the saved file."""
        r = CleaningReport()
        r.add("Missing Values", ["pm25: 10 missing"])
        r.add("Duplicates", ["3 exact duplicates removed"])
        r.add("Out of Range", ["pm10: 2 flagged"])
        out = tmp_path / "multi.txt"
        r.save(out)
        content = out.read_text(encoding="utf-8")
        assert "Missing Values" in content
        assert "pm25: 10 missing" in content
        assert "Duplicates" in content
        assert "3 exact duplicates removed" in content
        assert "Out of Range" in content


# ──────────────────────────────────────────────
# All-zero row — additional edge cases
# ──────────────────────────────────────────────
class TestAllZeroRowEdgeCases:
    """Additional edge cases for _all_zero_measurement_row (task item 3)."""

    def test_two_zeros_one_nan_not_flagged(self):
        """Row where 2 cols are 0 and 1 is NaN.

        The cleaner's all-zero check uses (row == 0).all(), which evaluates
        NaN == 0 as False.  Therefore a row with 2 zeros + 1 NaN is NOT
        flagged as all-zero — correct, because at least one measurement is
        genuinely missing rather than a suspicious zero read.
        """
        df = pd.DataFrame({
            "pm25": [0.0],
            "pm10": [0.0],
            "no2":  [np.nan],
        })
        result = _all_zero_measurement_row(df, ["pm25", "pm10", "no2"])
        # NaN != 0 so (row == 0).all() is False → not flagged
        assert not result.iloc[0], (
            "Row with two zeros and one NaN should NOT be flagged as all-zero "
            "(the NaN breaks the 'all values are zero' condition)"
        )

    def test_one_nonzero_not_flagged(self):
        """A row with at least one non-zero measurement should NOT be flagged."""
        df = pd.DataFrame({
            "pm25": [0.0],
            "pm10": [5.0],   # non-zero
            "no2":  [0.0],
        })
        result = _all_zero_measurement_row(df, ["pm25", "pm10", "no2"])
        assert not result.iloc[0], "Row with a non-zero measurement should not be flagged"

    def test_all_nan_not_flagged(self):
        """A row where every column is NaN has no non-null zeros → not flagged."""
        df = pd.DataFrame({
            "pm25": [np.nan],
            "pm10": [np.nan],
        })
        result = _all_zero_measurement_row(df, ["pm25", "pm10"])
        assert not result.iloc[0], "All-NaN row should NOT be flagged as all-zero"

    def test_single_zero_col_flagged(self):
        """Single measurement column that is zero → flagged."""
        df = pd.DataFrame({"pm25": [0.0, 1.0]})
        result = _all_zero_measurement_row(df, ["pm25"])
        assert result.iloc[0], "Single 0-value should be flagged"
        assert not result.iloc[1]

