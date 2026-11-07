"""
================================================================================
  test_interactive.py  —  Interactive AQI Prediction & Explainability Tester
  ClimatePulse | PCCOE Hackathon  —  MVP 1
================================================================================

Allows testing the trained AQI model, SHAP explainability, and WHO retrieval
with custom user input or standard test presets.

Usage:
------
    # Interactive mode (prompts for values or presets):
    python test_interactive.py

    # Quick test with a preset:
    python test_interactive.py --preset good
    python test_interactive.py --preset moderate
    python test_interactive.py --preset severe

    # Command-line input:
    python test_interactive.py --pm25 85.5 --pm10 160.0 --no2 35.0 --co 1.4
================================================================================
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path so 'src' can be imported cleanly
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from src.predict import load_model, prepare_input_from_dict, predict_and_explain

PRESETS = {
    "good": {
        "name": "Clean Air Day (Good)",
        "pm25": 18.0,
        "pm10": 42.0,
        "no2": 10.5,
        "so2": 4.2,
        "co": 0.5,
        "o3": 25.0,
    },
    "moderate": {
        "name": "Typical Urban Day (Moderate)",
        "pm25": 68.0,
        "pm10": 135.0,
        "no2": 32.0,
        "so2": 12.0,
        "co": 1.1,
        "o3": 48.0,
    },
    "poor": {
        "name": "High Pollution Day (Poor)",
        "pm25": 140.0,
        "pm10": 230.0,
        "no2": 65.0,
        "so2": 28.0,
        "co": 2.1,
        "o3": 75.0,
    },
    "severe": {
        "name": "Winter Smog Spike (Severe)",
        "pm25": 280.0,
        "pm10": 420.0,
        "no2": 110.0,
        "so2": 55.0,
        "co": 3.8,
        "o3": 95.0,
    },
}


def print_banner():
    print("=" * 72)
    print("      CLIMATE PULSE — AQI PREDICTION & EXPLAINABILITY TESTER")
    print("=" * 72)


def get_user_float(prompt: str, default_val: float) -> float:
    """Prompt user for a float with a default fallback."""
    user_str = input(f"  {prompt} [{default_val}]: ").strip()
    if not user_str:
        return default_val
    try:
        return float(user_str)
    except ValueError:
        print(f"    Invalid number, using default: {default_val}")
        return default_val


def interactive_menu(medians: dict) -> tuple[dict, str]:
    """Present menu to user and return (pollutants_dict, date_str)."""
    print_banner()
    print("\nSelect an input method:")
    print("  1. Preset: Clean Air Day (Good)")
    print("  2. Preset: Typical Urban Day (Moderate)")
    print("  3. Preset: High Pollution Day (Poor)")
    print("  4. Preset: Winter Smog Spike (Severe)")
    print("  5. Manual Entry (enter your own pollutant concentrations)")
    print("  q. Quit")

    choice = input("\nEnter choice [1-5]: ").strip().lower()

    if choice == "q":
        sys.exit(0)
    elif choice == "1":
        return PRESETS["good"], None
    elif choice == "2":
        return PRESETS["moderate"], None
    elif choice == "3":
        return PRESETS["poor"], None
    elif choice == "4":
        return PRESETS["severe"], None
    elif choice == "5":
        print("\nEnter pollutant concentrations (press Enter to use training median):")
        pm25 = get_user_float("PM2.5 (µg/m³)", round(medians.get("pm25", 45.0), 1))
        pm10 = get_user_float("PM10  (µg/m³)", round(medians.get("pm10", 95.0), 1))
        no2  = get_user_float("NO2   (µg/m³)", round(medians.get("no2", 30.0), 1))
        so2  = get_user_float("SO2   (µg/m³)", round(medians.get("so2", 12.0), 1))
        co   = get_user_float("CO    (mg/m³)", round(medians.get("co", 1.2), 2))
        o3   = get_user_float("Ozone (µg/m³)", round(medians.get("o3", 35.0), 1))
        date_in = input("  Date (YYYY-MM-DD, optional) [Today]: ").strip()
        date_str = date_in if date_in else None

        pollutants = {
            "name": "Custom User Input",
            "pm25": pm25,
            "pm10": pm10,
            "no2": no2,
            "so2": so2,
            "co": co,
            "o3": o3,
        }
        return pollutants, date_str
    else:
        print("Invalid choice, defaulting to Moderate preset.")
        return PRESETS["moderate"], None


def display_results(result: dict, scenario_name: str, sample_df: pd.DataFrame):
    """Format and print prediction and explainability results."""
    pred = result["prediction"]
    shap_info = result["shap_explanation"]
    category = pred["aqi_category"]

    cat_colors = {
        "Good": "\033[92m",          # Green
        "Satisfactory": "\033[92m",  # Green
        "Moderate": "\033[93m",      # Yellow
        "Poor": "\033[91m",          # Red
        "Very Poor": "\033[95m",     # Magenta
        "Severe": "\033[91m\033[1m", # Bold Red
    }
    color = cat_colors.get(category, "")
    reset = "\033[0m"

    print("\n" + "=" * 72)
    print(f"  SCENARIO: {scenario_name}")
    print("=" * 72)

    print("\n[1] INPUT POLLUTANT CONCENTRATIONS:")
    pollutants = ["pm25", "pm10", "no2", "so2", "co", "o3"]
    for p in pollutants:
        if p in sample_df.columns:
            val = sample_df[p].iloc[0]
            unit = "mg/m³" if p == "co" else "µg/m³"
            print(f"    • {p.upper():<6}: {val:>6.2f} {unit}")

    print("\n[2] AQI PREDICTION:")
    print(f"    • Predicted AQI : {color}{pred['aqi_value']:.1f}{reset}")
    print(f"    • CPCB Category : {color}{category}{reset}")

    print("\n[3] SHAP EXPLAINABILITY BREAKDOWN:")
    print(f"    • Model Base (Average) AQI: {shap_info['base_value']:.1f}")
    print("    • Top Influential Factors:")
    for i, factor in enumerate(shap_info["top_features"][:4], 1):
        arrow = "↑ pushes higher" if factor["direction"] == "increases" else "↓ pulls lower "
        print(f"      {i}. {factor['feature']:<11} = {factor['feature_value']:>6.2f} "
              f" [{arrow}] (SHAP: {factor['shap_value']:>+6.2f})")

    print("\n[4] EVIDENCE RETRIEVED FROM WHO GUIDELINES:")
    if result["who_evidence"]:
        for i, ev in enumerate(result["who_evidence"][:3], 1):
            print(f"    [{i}] Citation : {ev['citation']} (Relevance: {ev['score']:.3f})")
            clean_text = " ".join(ev["text"].split())[:180]
            print(f"        Excerpt  : \"{clean_text}...\"\n")
    else:
        print("    No matching WHO excerpts found.")

    print("[5] HEALTH GUIDANCE & EXPLANATION:")
    if result["health_explanation"]:
        # Indent each line
        for line in result["health_explanation"].splitlines():
            print(f"    {line}")
    else:
        print("    (Health explanation prompt prepared. Configure GROQ_API_KEY to run live LLM.)")

    print("\n" + "-" * 72)
    print(f"  Waterfall plot saved → outputs/shap/prediction_waterfall.png")
    print(f"  Full JSON saved       → outputs/latest_prediction.json")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test AQI Prediction with User Input")
    parser.add_argument("--preset", choices=["good", "moderate", "poor", "severe"],
                        help="Run with a predefined scenario")
    parser.add_argument("--pm25", type=float, help="PM2.5 concentration in µg/m³")
    parser.add_argument("--pm10", type=float, help="PM10 concentration in µg/m³")
    parser.add_argument("--no2",  type=float, help="NO2 concentration in µg/m³")
    parser.add_argument("--so2",  type=float, help="SO2 concentration in µg/m³")
    parser.add_argument("--co",   type=float, help="CO concentration in mg/m³")
    parser.add_argument("--o3",   type=float, help="Ozone concentration in µg/m³")
    parser.add_argument("--date", type=str,   help="Date in YYYY-MM-DD format")
    parser.add_argument("--no-groq", action="store_true", help="Skip Groq API call")

    args = parser.parse_args()

    # Load model and medians
    try:
        model, feature_names, medians = load_model()
    except FileNotFoundError as e:
        print(f"\n[ERROR] Model artefacts not found!\n{e}\n")
        print("Please run train_model.py first:")
        print("    python -m src.train_model\n")
        sys.exit(1)

    date_str = args.date

    # 1. Preset from CLI
    if args.preset:
        pollutants = PRESETS[args.preset]
        scenario_name = pollutants["name"]

    # 2. CLI flags for pollutants
    elif any([args.pm25 is not None, args.pm10 is not None, args.no2 is not None]):
        pollutants = {
            "name": "CLI Custom Input",
            "pm25": args.pm25 if args.pm25 is not None else medians.get("pm25", 45.0),
            "pm10": args.pm10 if args.pm10 is not None else medians.get("pm10", 95.0),
            "no2":  args.no2  if args.no2  is not None else medians.get("no2", 30.0),
            "so2":  args.so2  if args.so2  is not None else medians.get("so2", 12.0),
            "co":   args.co   if args.co   is not None else medians.get("co", 1.2),
            "o3":   args.o3   if args.o3   is not None else medians.get("o3", 35.0),
        }
        scenario_name = "CLI Custom Input"

    # 3. Interactive Menu
    else:
        pollutants, date_str = interactive_menu(medians)
        scenario_name = pollutants.get("name", "Custom Input")

    # Format sample DataFrame
    sample_df = prepare_input_from_dict(
        pollutants=pollutants,
        date_str=date_str,
        feature_names=feature_names,
        medians=medians,
    )

    # Run unified prediction + explainability
    print("\nRunning model inference and SHAP explainability...")
    result = predict_and_explain(
        sample=sample_df,
        model=model,
        feature_names=feature_names,
        medians=medians,
        use_groq=not args.no_groq,
    )

    # Display formatted result
    display_results(result, scenario_name, sample_df)


if __name__ == "__main__":
    main()

