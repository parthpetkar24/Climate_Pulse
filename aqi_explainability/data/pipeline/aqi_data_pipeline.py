"""
================================================================================
  AQI DATA PIPELINE  v2  --  Memory-Efficient Streaming Edition
  India Air Quality Dataset (CPCB 2015-2019)
  ClimatePulse | aqi_explainability
================================================================================

Key change from v1: every stage streams data file-by-file and appends directly
to the output CSV, so RAM usage stays low regardless of dataset size.

Output layout:
  data/
  +-- raw/            (untouched originals)
  +-- cleaned/        (per-pollutant cleaned CSVs, written incrementally)
  +-- preprocessed/   (unified merged datasets)
  +-- augmented/      (augmented training-ready datasets)
================================================================================
"""

import os
import sys
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent   # → data/
RAW_DIR   = BASE_DIR / "raw" / "PCCOE_Data"
CLEAN_DIR = BASE_DIR / "cleaned"
PREP_DIR  = BASE_DIR / "preprocessed"
AUG_DIR   = BASE_DIR / "augmented"

for d in [CLEAN_DIR, PREP_DIR, AUG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "pipeline_run.log", mode="w"),
    ],
)
log = logging.getLogger("aqi_pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
POLLUTANT_FOLDERS = {
    "PM25":  "pm25_ccr_copy",
    "PM10":  "pm10_ccr_copy",
    "NO2":   "no2_ccr_copy",
    "SO2":   "so2_ccr_copy",
    "Ozone": "o3_ccr_copy",
}

COL_RENAME = {
    "From Date": "timestamp",
    "To Date":   "to_date",
    "PM2.5":     "PM25",
    "State":     "state",
    "City":      "city",
    "Station":   "station",
    "lat":       "lat",
    "lon":       "lon",
}

PHYSICAL_LIMITS = {
    "PM25":  (0.0, 1000.0),
    "PM10":  (0.0, 2000.0),
    "NO2":   (0.0,  500.0),
    "SO2":   (0.0,  500.0),
    "Ozone": (0.0,  300.0),
}

AQI_BREAKPOINTS = {
    "PM25":  [(0,30,0,50),(31,60,51,100),(61,90,101,200),
              (91,120,201,300),(121,250,301,400),(251,500,401,500)],
    "PM10":  [(0,50,0,50),(51,100,51,100),(101,250,101,200),
              (251,350,201,300),(351,430,301,400),(431,600,401,500)],
    "NO2":   [(0,40,0,50),(41,80,51,100),(81,180,101,200),
              (181,280,201,300),(281,400,301,400),(401,800,401,500)],
    "SO2":   [(0,40,0,50),(41,80,51,100),(81,380,101,200),
              (381,800,201,300),(801,1600,301,400),(1601,2620,401,500)],
    "Ozone": [(0,50,0,50),(51,100,51,100),(101,168,101,200),
              (169,208,201,300),(209,748,301,400),(749,1000,401,500)],
}

AQI_CATEGORIES = [
    (0,   50,  "Good"),
    (51,  100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]

SEASON_MAP = {1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",
              6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON",12:"DJF"}

# Final columns written in cleaned CSVs
CLEAN_COLS = ["timestamp","state","city","station","lat","lon",
              "PM25","PM10","NO2","SO2","Ozone","source_file"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cap_outliers_iqr(series, factor=3.0):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series.clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr)


def _calc_aqi_single(conc, poll):
    if pd.isna(conc) or conc < 0:
        return np.nan
    for (c_lo, c_hi, i_lo, i_hi) in AQI_BREAKPOINTS.get(poll, []):
        if c_lo <= conc <= c_hi:
            return i_lo + (i_hi - i_lo) / (c_hi - c_lo) * (conc - c_lo)
    return np.nan


def compute_aqi(row):
    vals = []
    for poll in PHYSICAL_LIMITS:
        if poll in row.index and not pd.isna(row[poll]):
            v = _calc_aqi_single(row[poll], poll)
            if not pd.isna(v):
                vals.append(v)
    return max(vals) if vals else np.nan


def aqi_category(aqi):
    if pd.isna(aqi):
        return "Unknown"
    for lo, hi, label in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label
    return "Severe" if aqi > 500 else "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — ANALYSIS  (already completed; skip if report exists)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_raw_data():
    out_path = PREP_DIR / "raw_data_analysis_report.csv"
    if out_path.exists():
        log.info("STAGE 1 -- ANALYSIS already done. Skipping.")
        return

    log.info("=" * 68)
    log.info("STAGE 1 -- RAW DATA ANALYSIS")
    log.info("=" * 68)

    records = []
    grand_total = 0

    for poll_label, folder_name in POLLUTANT_FOLDERS.items():
        folder_path = RAW_DIR / folder_name
        csv_files   = sorted(folder_path.glob("*.csv"))
        log.info("  [%s] Found %d CSV files", poll_label, len(csv_files))

        for fp in csv_files:
            try:
                df = pd.read_csv(fp, low_memory=False)
            except Exception as e:
                log.warning("    Cannot read %s: %s", fp.name, e)
                continue

            df = df.rename(columns={k: v for k, v in COL_RENAME.items() if k in df.columns})
            nrows = len(df)
            grand_total += nrows

            rec = {"pollutant_folder": poll_label, "filename": fp.name,
                   "state_from_file": fp.stem.split("_")[0], "rows": nrows}

            for col in ["PM25","PM10","NO2","SO2","Ozone"]:
                if col in df.columns:
                    na = df[col].isna().sum()
                    vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    rec.update({
                        "missing_%s" % col:     na,
                        "pct_missing_%s" % col: round(na / nrows * 100, 1),
                        "mean_%s" % col:  round(vals.mean(), 2) if len(vals) else np.nan,
                        "max_%s"  % col:  round(vals.max(),  2) if len(vals) else np.nan,
                        "min_%s"  % col:  round(vals.min(),  2) if len(vals) else np.nan,
                        "neg_%s"  % col:  int((vals < 0).sum()),
                    })
                else:
                    for pfx in ["missing","pct_missing","mean","max","min","neg"]:
                        rec["%s_%s" % (pfx, col)] = np.nan

            rec["duplicate_rows"] = int(df.duplicated().sum())
            records.append(rec)

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(out_path, index=False)
    log.info("  Grand total rows: %s across %d files", f"{grand_total:,}", len(records))
    log.info("  Report -> %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — CLEANING  (streaming: write each cleaned file immediately)
# ─────────────────────────────────────────────────────────────────────────────

def clean_one_file(fp):
    """Return cleaned DataFrame for one CSV, or None."""
    try:
        df = pd.read_csv(fp, low_memory=False)
    except Exception as e:
        log.warning("    Cannot read %s: %s", fp.name, e)
        return None

    df = df.rename(columns={k: v for k, v in COL_RENAME.items() if k in df.columns})

    # Ensure city column
    if "city" not in df.columns or df["city"].isna().all():
        parts = fp.stem.split("_")
        df["city"] = parts[1] if len(parts) > 1 else "Unknown"

    # Parse timestamp
    if "timestamp" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=False, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if len(df) == 0:
        return None

    # Drop duplicates
    df = df.drop_duplicates()
    df = df.sort_values("timestamp")

    # Physical limits + outlier capping per pollutant
    for col, (lo, hi) in PHYSICAL_LIMITS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < lo, col] = np.nan
            df.loc[df[col] > hi, col] = np.nan
            mask = df[col].notna()
            if mask.sum() > 4:
                df.loc[mask, col] = _cap_outliers_iqr(df.loc[mask, col])

    df["source_file"] = fp.name

    # Keep only standard columns that exist
    keep = [c for c in CLEAN_COLS if c in df.columns]
    return df[keep]


def clean_all_pollutants():
    """Stream-clean every file, appending to one CSV per pollutant."""
    log.info("=" * 68)
    log.info("STAGE 2 -- DATA CLEANING  (streaming mode)")
    log.info("=" * 68)

    cleaned_paths = {}

    for poll_label, folder_name in POLLUTANT_FOLDERS.items():
        out_path  = CLEAN_DIR / ("%s_cleaned.csv" % poll_label.lower())

        # Skip if already done
        if out_path.exists():
            log.info("  [%s] Already cleaned -> %s", poll_label, out_path)
            cleaned_paths[poll_label] = out_path
            continue

        folder_path = RAW_DIR / folder_name
        csv_files   = sorted(folder_path.glob("*.csv"))
        log.info("  [%s] Cleaning %d files -> %s", poll_label, len(csv_files), out_path)

        header_written = False
        total_rows     = 0
        skipped        = 0

        for i, fp in enumerate(csv_files, 1):
            cdf = clean_one_file(fp)
            if cdf is None or len(cdf) == 0:
                skipped += 1
                continue

            # Append directly to output CSV
            cdf.to_csv(out_path,
                       mode="a",
                       header=not header_written,
                       index=False)
            header_written = True
            total_rows    += len(cdf)

            if i % 20 == 0 or i == len(csv_files):
                log.info("    [%s] %d/%d files | %s rows written",
                         poll_label, i, len(csv_files), f"{total_rows:,}")

        log.info("  [%s] DONE -> %s rows | %d skipped | %s",
                 poll_label, f"{total_rows:,}", skipped, out_path)
        cleaned_paths[poll_label] = out_path

    return cleaned_paths


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — PREPROCESSING  (chunk-based merge → hourly → daily)
# ─────────────────────────────────────────────────────────────────────────────

def add_temporal_features(df):
    ts = df["timestamp"]
    df["year"]        = ts.dt.year
    df["month"]       = ts.dt.month
    df["day"]         = ts.dt.day
    df["hour"]        = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_year"] = ts.dt.dayofyear
    df["quarter"]     = ts.dt.quarter
    df["season"]      = ts.dt.month.map(SEASON_MAP)
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def preprocess(cleaned_paths):
    log.info("=" * 68)
    log.info("STAGE 3 -- PREPROCESSING")
    log.info("=" * 68)

    hourly_out = PREP_DIR / "aqi_hourly_preprocessed.csv"
    daily_out  = PREP_DIR / "aqi_daily_preprocessed.csv"

    # ── Build station-level hourly aggregation ────────────────────────────────
    # Strategy: read each cleaned CSV, group by (timestamp, station), average
    # all pollutants, then append to the hourly output in chunks.

    log.info("  Merging all cleaned pollutant files (station x timestamp) ...")

    # Collect all unique station-timestamp records from ALL cleaned files
    # We'll do this in two passes:
    #   Pass 1: stream each cleaned CSV → group by (date, station) → daily agg
    #           (avoids keeping 38M rows in RAM)
    #   Then produce hourly from PM25 file (richest) + add temporal features

    # --- Pass 1: daily aggregation (memory efficient) ---
    log.info("  Pass 1: daily aggregation from cleaned files ...")
    daily_header = False
    total_daily  = 0

    for poll_label, fpath in cleaned_paths.items():
        if not fpath.exists():
            log.warning("    Missing cleaned file: %s", fpath)
            continue

        log.info("    Processing %s ...", fpath.name)
        poll_cols = [c for c in PHYSICAL_LIMITS if c in
                     pd.read_csv(fpath, nrows=1).columns]

        CHUNK = 200_000
        for chunk in pd.read_csv(fpath, chunksize=CHUNK, low_memory=False):
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
            chunk = chunk.dropna(subset=["timestamp"])
            chunk["date"] = chunk["timestamp"].dt.date

            agg_d = {c: "mean" for c in poll_cols if c in chunk.columns}
            for mc in ["state","city","lat","lon"]:
                if mc in chunk.columns:
                    agg_d[mc] = "first"

            daily_chunk = (
                chunk.groupby(["date","station"], as_index=False)
                .agg(agg_d)
            )
            daily_chunk["date"] = pd.to_datetime(daily_chunk["date"])
            daily_chunk["pollutant_source"] = poll_label

            daily_chunk.to_csv(daily_out,
                               mode="a",
                               header=not daily_header,
                               index=False)
            daily_header = True
            total_daily += len(daily_chunk)

    log.info("  Pass 1 complete: %s daily rows written", f"{total_daily:,}")

    # --- Consolidate daily: merge duplicate station-dates across pollutant sources ---
    log.info("  Consolidating daily dataset ...")
    daily_df = pd.read_csv(daily_out, low_memory=False)
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    poll_cols = [c for c in PHYSICAL_LIMITS if c in daily_df.columns]
    agg2 = {c: "mean" for c in poll_cols}
    for mc in ["state","city","lat","lon"]:
        if mc in daily_df.columns:
            agg2[mc] = "first"

    daily_final = (
        daily_df.groupby(["date","station"], as_index=False)
        .agg(agg2)
        .sort_values(["station","date"])
        .reset_index(drop=True)
    )

    # Temporal features on daily
    daily_final["year"]        = daily_final["date"].dt.year
    daily_final["month"]       = daily_final["date"].dt.month
    daily_final["day_of_week"] = daily_final["date"].dt.dayofweek
    daily_final["season"]      = daily_final["month"].map(SEASON_MAP)
    daily_final["is_weekend"]  = (daily_final["day_of_week"] >= 5).astype(int)
    daily_final["month_sin"]   = np.sin(2 * np.pi * daily_final["month"] / 12)
    daily_final["month_cos"]   = np.cos(2 * np.pi * daily_final["month"] / 12)

    # Compute AQI
    log.info("  Computing daily AQI ...")
    daily_final["AQI"] = daily_final[poll_cols].apply(compute_aqi, axis=1)
    daily_final["AQI_category"] = daily_final["AQI"].apply(aqi_category)

    daily_final.to_csv(daily_out, index=False)
    log.info("  Daily dataset: %s rows -> %s", f"{len(daily_final):,}", daily_out)

    # --- Pass 2: hourly from PM25 file (best temporal coverage) ---
    log.info("  Pass 2: hourly dataset from PM25 cleaned file ...")
    pm25_path = cleaned_paths.get("PM25")
    hourly_header = False
    total_hourly  = 0

    if pm25_path and pm25_path.exists():
        CHUNK = 100_000
        for chunk in pd.read_csv(pm25_path, chunksize=CHUNK, low_memory=False):
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
            chunk = chunk.dropna(subset=["timestamp"])
            chunk = add_temporal_features(chunk)

            poll_present = [c for c in PHYSICAL_LIMITS if c in chunk.columns]
            chunk["AQI"]          = chunk[poll_present].apply(compute_aqi, axis=1)
            chunk["AQI_category"] = chunk["AQI"].apply(aqi_category)

            chunk.to_csv(hourly_out,
                         mode="a",
                         header=not hourly_header,
                         index=False)
            hourly_header = True
            total_hourly += len(chunk)

        log.info("  Hourly dataset: %s rows -> %s", f"{total_hourly:,}", hourly_out)
    else:
        log.warning("  PM25 cleaned file not found; hourly skipped.")

    # Print AQI distribution
    log.info("\n  AQI Category distribution (daily):")
    vc = daily_final["AQI_category"].value_counts()
    for cat, cnt in vc.items():
        log.info("    %-15s: %7s  (%.1f%%)", cat, f"{cnt:,}", cnt/len(daily_final)*100)

    return daily_final


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — AUGMENTATION  (operates on the daily dataset)
# ─────────────────────────────────────────────────────────────────────────────

def augment_jitter(df, cols, noise_pct=0.02, n_copies=2, seed=42):
    rng   = np.random.default_rng(seed)
    parts = [df.copy()]
    for i in range(n_copies):
        aug = df.copy()
        for col in cols:
            if col in aug.columns and aug[col].notna().any():
                sigma = noise_pct * aug[col].std()
                noise = rng.normal(0, sigma, size=len(aug))
                aug[col] = (aug[col] + noise).clip(lower=0)
        aug["aug_method"] = "jitter_%d" % (i + 1)
        parts.append(aug)
    return pd.concat(parts, ignore_index=True)


def augment_smote_like(df, target_col, minority_cats, feat_cols,
                       multiplier=3, seed=42):
    rng   = np.random.default_rng(seed)
    parts = [df.copy()]
    for cat in minority_cats:
        minority = df[df[target_col] == cat].reset_index(drop=True)
        if len(minority) < 2:
            continue
        n_synth = len(minority) * multiplier
        log.info("    SMOTE '%s': %d real -> +%d synthetic", cat, len(minority), n_synth)
        idx_a = rng.integers(0, len(minority), size=n_synth)
        idx_b = rng.integers(0, len(minority), size=n_synth)
        lam   = rng.uniform(0, 1, size=n_synth)
        synth = minority.iloc[idx_a].copy().reset_index(drop=True)
        for fc in feat_cols:
            if fc in minority.columns and pd.api.types.is_numeric_dtype(minority[fc]):
                a_v = minority[fc].iloc[idx_a].values
                b_v = minority[fc].iloc[idx_b].values
                synth[fc] = a_v + lam * (b_v - a_v)
        synth["aug_method"] = "smote_%s" % cat.replace(" ","_")
        parts.append(synth)
    return pd.concat(parts, ignore_index=True)


def augment_time_shift(df, shift_days=None, seed=42):
    if shift_days is None:
        shift_days = [-7, 7]
    parts = [df.copy()]
    for sh in shift_days:
        aug = df.copy()
        aug["date"]       = aug["date"] + pd.Timedelta(days=sh)
        aug["month"]      = aug["date"].dt.month
        aug["day_of_week"]= aug["date"].dt.dayofweek
        aug["season"]     = aug["month"].map(SEASON_MAP)
        aug["aug_method"] = "date_shift_%+dd" % sh
        parts.append(aug)
    return pd.concat(parts, ignore_index=True)


def augment_data(daily_df):
    log.info("=" * 68)
    log.info("STAGE 4 -- DATA AUGMENTATION")
    log.info("=" * 68)

    poll_cols = [c for c in PHYSICAL_LIMITS if c in daily_df.columns]
    num_cols  = [c for c in daily_df.columns
                 if pd.api.types.is_numeric_dtype(daily_df[c])
                 and c not in ["lat","lon"]]

    # A. Gaussian jitter
    log.info("  [A] Gaussian jitter (2%%, 2 copies) ...")
    jitter = augment_jitter(daily_df, cols=poll_cols, noise_pct=0.02, n_copies=2)
    out_a  = AUG_DIR / "daily_jitter_augmented.csv"
    jitter.to_csv(out_a, index=False)
    log.info("    %s rows -> %s", f"{len(jitter):,}", out_a)

    # B. Date-shift ±7 days
    log.info("  [B] Date-shift augmentation (+/-7 days) ...")
    shifted = augment_time_shift(daily_df, shift_days=[-7, 7])
    out_b   = AUG_DIR / "daily_dateshift_augmented.csv"
    shifted.to_csv(out_b, index=False)
    log.info("    %s rows -> %s", f"{len(shifted):,}", out_b)

    # C. SMOTE for minority AQI categories
    if "AQI_category" in daily_df.columns:
        vc         = daily_df["AQI_category"].value_counts()
        majority_n = vc.iloc[0]
        minority   = [c for c, n in vc.items()
                      if n < majority_n * 0.3 and c not in ("Unknown",)]
        log.info("  [C] SMOTE for minority categories: %s", minority)
        smote  = augment_smote_like(daily_df, "AQI_category", minority,
                                    feat_cols=num_cols, multiplier=3)
        out_c  = AUG_DIR / "daily_smote_augmented.csv"
        smote.to_csv(out_c, index=False)
        log.info("    %s rows -> %s", f"{len(smote):,}", out_c)
        log.info("    Post-SMOTE distribution:")
        vc2 = smote["AQI_category"].value_counts()
        for cat, cnt in vc2.items():
            log.info("      %-15s: %7s  (%.1f%%)", cat, f"{cnt:,}",
                     cnt / len(smote) * 100)

    # D. Combined augmented dataset (jitter + shift together)
    log.info("  [D] Combined augmented dataset ...")
    combined = pd.concat([
        jitter,
        shifted[shifted["aug_method"].notna()],  # only the shifted copies
    ], ignore_index=True)
    out_d = AUG_DIR / "daily_combined_augmented.csv"
    combined.to_csv(out_d, index=False)
    log.info("    %s rows -> %s", f"{len(combined):,}", out_d)

    log.info("\n  Augmented files:")
    for f in sorted(AUG_DIR.iterdir()):
        log.info("    %-45s  %.1f MB", f.name, f.stat().st_size / 1e6)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 68)
    log.info("  AQI DATA PIPELINE v2 (streaming) -- ClimatePulse / PCCOE")
    log.info("=" * 68)
    log.info("BASE DIR : %s", BASE_DIR)

    analyse_raw_data()                     # Stage 1 (skips if report exists)
    cleaned_paths = clean_all_pollutants() # Stage 2 (streaming)
    daily_df      = preprocess(cleaned_paths) # Stage 3
    augment_data(daily_df)                 # Stage 4

    log.info("")
    log.info("=" * 68)
    log.info("  PIPELINE COMPLETE")
    log.info("=" * 68)
    log.info("Log -> %s", BASE_DIR / "pipeline_run.log")


if __name__ == "__main__":
    main()
