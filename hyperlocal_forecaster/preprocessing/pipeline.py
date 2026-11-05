"""
pipeline.py — End-to-end orchestration for hyperlocal_forecaster preprocessing.
=================================================================================
Runs cleaning → preprocessing in one command.

Usage (from hyperlocal_forecaster/):
    python -m preprocessing.pipeline
    python -m preprocessing.pipeline --scaler minmax --lookback 7 --horizon 3
"""

import argparse
import logging
import sys
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
CLEANED_DIR  = BASE_DIR / "data" / "cleaned"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("pipeline")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(ch)
    return logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full cleaning + preprocessing pipeline."
    )
    parser.add_argument("--source", choices=["daily", "hourly", "all"],
                        default="daily",
                        help="Dataset source to clean (default: daily)")
    parser.add_argument("--scaler", choices=["standard", "minmax"],
                        default="standard")
    parser.add_argument("--lookback", type=int, default=14)
    parser.add_argument("--horizon",  type=int, default=7)
    parser.add_argument("--target",   default="aqi")
    parser.add_argument("--report-only", action="store_true",
                        help="Generate quality report without saving files")
    args = parser.parse_args()

    logger = _setup_logging()
    logger.info("=" * 60)
    logger.info("Climate_Pulse — Full Pipeline")
    logger.info("=" * 60)

    # --- Step 1: Clean ---
    from cleaning.cleaner import (
        clean_daily_aqi, clean_ccr_folder, CleaningReport, _setup_logging as cl_log,
        CLEANED_DIR as CL_DIR, REPORTS_DIR, CCR_FOLDERS,
        RAW_DIR,
    )
    report = CleaningReport()
    cl_logger = cl_log(REPORTS_DIR / "cleaning.log")

    if args.source in ("daily", "all"):
        daily_dir = RAW_DIR / "Dataset Daily Air Quality Index and Pollutant Conc"
        clean_daily_aqi(
            input_path=daily_dir / "all_stations_combined.csv",
            output_path=CLEANED_DIR / "daily_aqi_all_stations_cleaned.csv",
            report=report,
            logger=cl_logger,
            save=not args.report_only,
        )
        for csv in sorted(daily_dir.glob("station_*.csv")):
            clean_daily_aqi(
                input_path=csv,
                output_path=CLEANED_DIR / f"daily_aqi_{csv.stem}_cleaned.csv",
                report=report,
                logger=cl_logger,
                save=not args.report_only,
            )
        clean_daily_aqi(
            input_path=daily_dir / "city_averaged.csv",
            output_path=CLEANED_DIR / "daily_aqi_city_averaged_cleaned.csv",
            report=report,
            logger=cl_logger,
            save=not args.report_only,
        )

    if args.source in ("hourly", "all"):
        for key, folder in CCR_FOLDERS.items():
            if not folder.exists():
                continue
            clean_ccr_folder(
                pollutant_key=key,
                folder=folder,
                output_dir=CLEANED_DIR,
                report=report,
                logger=cl_logger,
                save=not args.report_only,
            )

    report.save(REPORTS_DIR / "data_quality_report.txt")

    if args.report_only:
        logger.info("Report-only mode: preprocessing skipped.")
        return

    # --- Step 2: Preprocess ---
    from preprocessing.preprocessor import preprocess_daily_aqi, _setup_logging as pr_log
    pr_logger = pr_log()

    cleaned_input = CLEANED_DIR / "daily_aqi_all_stations_cleaned.csv"
    if cleaned_input.exists():
        preprocess_daily_aqi(
            input_path=cleaned_input,
            output_dir=PROCESSED_DIR,
            scaler_type=args.scaler,
            lookback=args.lookback,
            horizon=args.horizon,
            target_col=args.target,
            logger=pr_logger,
        )
    else:
        logger.warning(f"Cleaned file not found: {cleaned_input}")

    logger.info("Full pipeline complete.")


if __name__ == "__main__":
    main()

