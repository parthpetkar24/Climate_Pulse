"""
cleaner.py — Data Cleaning for Climate_Pulse / hyperlocal_forecaster
=====================================================================
Cleans the raw air-quality datasets:
  • Dataset Daily AQI (all_stations_combined.csv + per-station CSVs)
  • CPCB CCR hourly per-pollutant folders (pm25_ccr_copy, no2_ccr_copy, …)

All cleaning decisions are logged and summarised in a human-readable report.
Raw files are NEVER modified.

Usage (from hyperlocal_forecaster/):
    python -m cleaning.cleaner                          # default paths
    python -m cleaning.cleaner --source daily           # daily AQI only
    python -m cleaning.cleaner --source hourly          # CCR hourly only
    python -m cleaning.cleaner --report-only            # report without saving
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to the hyperlocal_forecaster/ directory)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
REPORTS_DIR = BASE_DIR / "reports"

DAILY_AQI_DIR = RAW_DIR / "Dataset Daily Air Quality Index and Pollutant Conc"
CCR_FOLDERS = {
    "pm25": RAW_DIR / "pm25_ccr_copy",
    "pm10": RAW_DIR / "pm10_ccr_copy",
    "no2":  RAW_DIR / "no2_ccr_copy",
    "so2":  RAW_DIR / "so2_ccr_copy",
    "o3":   RAW_DIR / "o3_ccr_copy",
}

# ---------------------------------------------------------------------------
# Domain validity ranges (µg/m³ for particulates, ppb for gases, AQI 0-500)
# Source: CPCB NAAQS guidelines + WHO ambient air quality guidelines
# ---------------------------------------------------------------------------
POLLUTANT_RANGES = {
    "pm25":           (0.0, 900.0),   # µg/m³  — extreme Delhi winter spikes ~900
    "pm10":           (0.0, 1500.0),  # µg/m³  — dust storm events included
    "no2":            (0.0, 500.0),   # µg/m³
    "so2":            (0.0, 500.0),   # µg/m³
    "o3":             (0.0, 500.0),   # µg/m³  (as Ozone)
    "co":             (0.0, 50.0),    # mg/m³  — CPCB CCR data is in mg/m³
    "aqi":            (0.0, 500.0),
    "sub_pm25":       (0.0, 500.0),
    "sub_pm10":       (0.0, 500.0),
    "sub_no2":        (0.0, 500.0),
    "sub_o3":         (0.0, 500.0),
    "sub_co":         (0.0, 500.0),
    "sub_so2":        (0.0, 500.0),
}

# Columns that are pollutant measurements in the daily AQI dataset
DAILY_POLLUTANT_COLS = ["pm25", "pm10", "no2", "so2", "o3", "co"]
DAILY_SUBINDEX_COLS  = ["sub_pm25", "sub_pm10", "sub_no2", "sub_o3", "sub_co", "sub_so2"]

# Columns that are pollutant measurements in the CCR hourly dataset
CCR_POLLUTANT_COLS = ["PM10", "PM2.5", "NO2", "SO2", "Ozone"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging(log_path: Path) -> logging.Logger:
    """Configure root logger to write to both console and a log file."""
    logger = logging.getLogger("cleaner")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Report helper
# ---------------------------------------------------------------------------
class CleaningReport:
    """Accumulates cleaning statistics, writes a text summary at the end."""

    def __init__(self):
        self._sections: list[str] = []
        self._started = datetime.now()

    def add(self, title: str, lines: list[str]) -> None:
        block = [f"\n{'='*70}", f"  {title}", f"{'='*70}"]
        block.extend(lines)
        self._sections.append("\n".join(block))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"Climate_Pulse / hyperlocal_forecaster — Data Quality Report\n"
            f"Generated : {self._started.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'='*70}"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + "\n".join(self._sections) + "\n")
        # Note: file written in UTF-8 so any unicode chars (arrows etc.) are fine there
        print(f"\n[Report] Saved to {path}")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _range_flag(series: pd.Series, col: str) -> pd.Series:
    """Return boolean mask of rows that are out of domain range for `col`."""
    lo, hi = POLLUTANT_RANGES.get(col, (None, None))
    if lo is None:
        return pd.Series(False, index=series.index)
    return (series < lo) | (series > hi)


def _all_zero_measurement_row(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """True where every non-null measurement column is exactly 0."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(False, index=df.index)
    sub = df[present].copy()
    has_any_value = sub.notna().any(axis=1)
    all_zero = (sub.fillna(np.nan) == 0).all(axis=1)
    return has_any_value & all_zero


# ===========================================================================
# DAILY AQI DATASET
# ===========================================================================

def clean_daily_aqi(
    input_path: Path,
    output_path: Path,
    report: CleaningReport,
    logger: logging.Logger,
    save: bool = True,
) -> pd.DataFrame:
    """
    Clean one Daily AQI CSV file.

    Steps
    -----
    1. Load and record raw shape.
    2. Standardise column names (lower-snake-case).
    3. Parse `date` column to datetime.
    4. Replace sentinel strings / empty strings with NaN.
    5. Coerce numeric columns.
    6. Remove exact duplicate rows.
    7. Flag and report out-of-range pollutant values (do NOT delete — log).
    8. Flag and report all-zero measurement rows (mark, do NOT delete).
    9. Sort chronologically, then by station.
    10. Save cleaned file.
    """
    logger.info(f"[Daily AQI] Cleaning: {input_path.name}")

    if not input_path.exists():
        logger.error(f"  File not found: {input_path}")
        return pd.DataFrame()

    df = pd.read_csv(input_path, dtype=str)
    raw_shape = df.shape
    logger.debug(f"  Raw shape: {raw_shape}")

    # 1. Standardise column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Rename known columns to canonical names
    rename_map = {
        "pm2_5": "pm25",
        "pm_2_5": "pm25",
        "ozone": "o3",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns},
              inplace=True)

    # 2. Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        bad_dates = df["date"].isna().sum()
        if bad_dates:
            logger.warning(f"  {bad_dates} unparseable date(s) → NaT (rows kept)")
    else:
        logger.warning("  No 'date' column found.")

    # 3. Replace sentinel strings → NaN
    sentinel_values = {"", "na", "nan", "none", "null", "-", "--", "n/a"}
    str_cols = df.select_dtypes(include="str").columns
    for col in str_cols:
        df[col] = df[col].str.strip()
        df[col] = df[col].where(
            ~df[col].str.lower().isin(sentinel_values), other=np.nan
        )

    # 4. Coerce numeric columns
    num_cols = [c for c in df.columns
                if c not in {"date", "city", "station_name", "dominant_pollutant"}
                and c != "station_id"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # station_id → int if possible
    if "station_id" in df.columns:
        df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce")

    # 5. Remove exact duplicates
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    n_dups = n_before - len(df)
    logger.info(f"  Duplicates removed: {n_dups}")

    # 6. Out-of-range detection
    oor_counts: dict[str, int] = {}
    for col in DAILY_POLLUTANT_COLS + DAILY_SUBINDEX_COLS + ["aqi"]:
        if col not in df.columns:
            continue
        mask = _range_flag(df[col], col)
        count = mask.sum()
        if count:
            oor_counts[col] = int(count)
            logger.warning(f"  Out-of-range [{col}]: {count} rows flagged (NOT deleted)")

    # Add flag column
    df["flag_out_of_range"] = False
    for col in oor_counts:
        df["flag_out_of_range"] |= _range_flag(df[col], col)

    # 7. All-zero rows
    zero_mask = _all_zero_measurement_row(df, DAILY_POLLUTANT_COLS)
    n_zero = zero_mask.sum()
    df["flag_all_zero"] = zero_mask
    logger.info(f"  All-zero measurement rows flagged: {n_zero}")

    # 8. Missing value summary
    missing = df[DAILY_POLLUTANT_COLS + ["aqi"]].isna().sum()
    missing_pct = (missing / len(df) * 100).round(1)

    # 9. Sort
    sort_cols = []
    if "station_name" in df.columns:
        sort_cols.append("station_name")
    if "date" in df.columns:
        sort_cols.append("date")
    if sort_cols:
        df.sort_values(sort_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)

    # 10. Save
    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"  Saved cleaned file -> {output_path}")

    # Report
    lines = [
        f"  Source file    : {input_path.name}",
        f"  Raw rows×cols  : {raw_shape[0]} × {raw_shape[1]}",
        f"  Clean rows×cols: {len(df)} × {len(df.columns)}",
        f"  Duplicates removed       : {n_dups}",
        f"  All-zero rows flagged    : {n_zero}",
        f"  Out-of-range flags       : { {k:v for k,v in oor_counts.items()} }",
        "",
        "  Missing values (measurement cols):",
    ]
    for col in missing.index:
        lines.append(f"    {col:20s}: {missing[col]:5d} ({missing_pct[col]:.1f}%)")
    report.add(f"Daily AQI — {input_path.name}", lines)

    return df


# ===========================================================================
# CCR HOURLY DATASET
# ===========================================================================

def _parse_ccr_datetime(series: pd.Series) -> pd.Series:
    """
    Parse CCR 'From Date' format: 'M/D/YY H:MM' → datetime.
    Falls back to pandas inference on failure.
    """
    parsed = pd.to_datetime(series, format="%m/%d/%y %H:%M", errors="coerce")
    n_fail = parsed.isna().sum() - series.isna().sum()
    if n_fail > 0:
        # Try alternative with mixed/dayfirst formats (pandas 3+ compatible)
        alt = pd.to_datetime(series, format="mixed", dayfirst=False, errors="coerce")
        parsed = parsed.fillna(alt)
    return parsed


def clean_ccr_folder(
    pollutant_key: str,
    folder: Path,
    output_dir: Path,
    report: CleaningReport,
    logger: logging.Logger,
    save: bool = True,
) -> pd.DataFrame:
    """
    Merge all station CSVs in a CCR pollutant folder into one clean DataFrame.

    Steps
    -----
    1. Load all CSVs; attach state/city parsed from filename.
    2. Replace 'None' strings → NaN.
    3. Parse 'From Date' to datetime.
    4. Coerce numeric columns.
    5. Remove exact duplicates within each station.
    6. Flag out-of-range values (do NOT delete).
    7. Sort by station → datetime.
    8. Save merged file.
    """
    logger.info(f"[CCR {pollutant_key.upper()}] Processing folder: {folder.name}")

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        logger.warning(f"  No CSV files found in {folder}")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    skipped = 0
    for fpath in csv_files:
        if fpath.stem.startswith("."):
            continue
        try:
            df_f = pd.read_csv(fpath, dtype=str)
        except Exception as exc:
            logger.warning(f"  Could not read {fpath.name}: {exc}")
            skipped += 1
            continue

        # Parse state/city/station from filename: State_City_StationName.csv
        parts = fpath.stem.split("_", maxsplit=2)
        df_f["source_file"] = fpath.stem
        df_f["state_parsed"] = parts[0] if len(parts) >= 1 else ""
        df_f["city_parsed"]  = parts[1] if len(parts) >= 2 else ""

        frames.append(df_f)

    if not frames:
        logger.error("  No valid frames loaded.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    raw_rows = len(df)
    logger.info(f"  Loaded {len(csv_files)-skipped} files, {raw_rows} raw rows")

    # Standardise column names
    df.columns = df.columns.str.strip()

    # Replace 'None' strings → NaN (CCR uses the Python repr 'None')
    sentinel_values = {"none", "nan", "na", "null", "", "-", "--"}
    str_cols = df.select_dtypes(include="str").columns
    for col in str_cols:
        df[col] = df[col].str.strip()
        df[col] = df[col].where(
            ~df[col].str.lower().isin(sentinel_values), other=np.nan
        )

    # Parse datetime
    if "From Date" in df.columns:
        df["datetime"] = _parse_ccr_datetime(df["From Date"])
        bad_ts = df["datetime"].isna().sum()
        if bad_ts:
            logger.warning(f"  {bad_ts} unparseable timestamps → NaT (rows kept)")
    else:
        logger.warning("  'From Date' column missing.")
        df["datetime"] = pd.NaT

    # Coerce numeric
    numeric_targets = CCR_POLLUTANT_COLS + ["lat", "lon"]
    for col in numeric_targets:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove exact duplicates (same station + From Date + all measurements)
    dedup_cols = ["Station", "From Date"] + [c for c in CCR_POLLUTANT_COLS if c in df.columns]
    dedup_cols = [c for c in dedup_cols if c in df.columns]
    n_before = len(df)
    df.drop_duplicates(subset=dedup_cols, inplace=True)
    n_dups = n_before - len(df)
    logger.info(f"  Duplicates removed: {n_dups}")

    # CCR pollutant range mapping (column name → canonical key)
    ccr_col_map = {
        "PM2.5": "pm25",
        "PM10":  "pm10",
        "NO2":   "no2",
        "SO2":   "so2",
        "Ozone": "o3",
    }
    oor_counts: dict[str, int] = {}
    df["flag_out_of_range"] = False
    for col, key in ccr_col_map.items():
        if col not in df.columns:
            continue
        mask = _range_flag(df[col], key)
        count = mask.sum()
        if count:
            oor_counts[col] = int(count)
            logger.warning(f"  Out-of-range [{col}]: {count} rows flagged")
        df["flag_out_of_range"] |= mask

    # Missing value counts for the primary pollutant
    missing: dict[str, int] = {}
    for col in CCR_POLLUTANT_COLS:
        if col in df.columns:
            missing[col] = int(df[col].isna().sum())

    # Sort
    sort_cols = []
    if "Station" in df.columns:
        sort_cols.append("Station")
    if "datetime" in df.columns:
        sort_cols.append("datetime")
    if sort_cols:
        df.sort_values(sort_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)

    if save:
        out = output_dir / f"ccr_{pollutant_key}_cleaned.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info(f"  Saved -> {out}")

    lines = [
        f"  Pollutant folder : {folder.name}",
        f"  Station files    : {len(csv_files) - skipped} loaded, {skipped} skipped",
        f"  Raw rows         : {raw_rows}",
        f"  After dedup      : {len(df)}",
        f"  Duplicates removed: {n_dups}",
        f"  Out-of-range flags: { {k: v for k,v in oor_counts.items()} }",
        "",
        "  Missing values (measurement cols):",
    ]
    for col, cnt in missing.items():
        pct = cnt / len(df) * 100 if len(df) else 0
        lines.append(f"    {col:12s}: {cnt:7d} ({pct:.1f}%)")
    report.add(f"CCR Hourly — {pollutant_key.upper()}", lines)

    return df


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean raw air-quality datasets for hyperlocal_forecaster."
    )
    parser.add_argument(
        "--source", choices=["daily", "hourly", "all"], default="all",
        help="Which dataset to clean (default: all)"
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Generate report without saving cleaned files"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR,
        help="Override path to raw data directory"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=CLEANED_DIR,
        help="Override path to cleaned output directory"
    )
    args = parser.parse_args()

    raw_dir:     Path = args.raw_dir
    cleaned_dir: Path = args.out_dir
    save = not args.report_only

    log_path = REPORTS_DIR / "cleaning.log"
    logger = _setup_logging(log_path)
    report = CleaningReport()

    logger.info("=" * 60)
    logger.info("Climate_Pulse — Data Cleaning Pipeline")
    logger.info(f"Source    : {args.source}")
    logger.info(f"Raw dir   : {raw_dir}")
    logger.info(f"Output dir: {cleaned_dir}")
    logger.info(f"Save files: {save}")
    logger.info("=" * 60)

    if args.source in ("daily", "all"):
        daily_dir = raw_dir / "Dataset Daily Air Quality Index and Pollutant Conc"
        if not daily_dir.exists():
            logger.warning(f"Daily AQI directory not found: {daily_dir}")
        else:
            # Clean all_stations_combined (primary)
            clean_daily_aqi(
                input_path=daily_dir / "all_stations_combined.csv",
                output_path=cleaned_dir / "daily_aqi_all_stations_cleaned.csv",
                report=report,
                logger=logger,
                save=save,
            )
            # Clean per-station files
            for csv in sorted(daily_dir.glob("station_*.csv")):
                clean_daily_aqi(
                    input_path=csv,
                    output_path=cleaned_dir / f"daily_aqi_{csv.stem}_cleaned.csv",
                    report=report,
                    logger=logger,
                    save=save,
                )
            # Clean city_averaged
            clean_daily_aqi(
                input_path=daily_dir / "city_averaged.csv",
                output_path=cleaned_dir / "daily_aqi_city_averaged_cleaned.csv",
                report=report,
                logger=logger,
                save=save,
            )

    if args.source in ("hourly", "all"):
        ccr_folders = {
            k: raw_dir / v.name
            for k, v in CCR_FOLDERS.items()
        }
        for key, folder in ccr_folders.items():
            if not folder.exists():
                logger.warning(f"CCR folder not found: {folder}")
                continue
            clean_ccr_folder(
                pollutant_key=key,
                folder=folder,
                output_dir=cleaned_dir,
                report=report,
                logger=logger,
                save=save,
            )

    report_path = REPORTS_DIR / "data_quality_report.txt"
    report.save(report_path)
    logger.info("Cleaning complete.")


if __name__ == "__main__":
    main()
