"""
================================================================================
  run_all.py  —  Run the full AQI Explainability Pipeline end-to-end
  ClimatePulse | PCCOE Hackathon  —  MVP 1
================================================================================

Convenience script that runs all steps sequentially:
  Step 1 → Data preprocessing
  Step 2 → Model training + evaluation + saving
  Step 3 → SHAP explainability (global + local)
  Step 4 → WHO knowledge base indexing
  Step 5 → Prediction with explanation (Groq if API key available)

Usage
-----
    cd C:\\PCCOE26\\Climate_Pulse\\aqi_explainability
    python run_all.py
================================================================================
"""

import logging
import sys
import warnings
from pathlib import Path

# Ensure the aqi_explainability directory is on the path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aqi_explainability")


def main():
    log.info("=" * 68)
    log.info("  AQI EXPLAINABILITY PIPELINE  —  ClimatePulse MVP 1")
    log.info("=" * 68)

    # ── Step 1: Data Preprocessing ───────────────────────────────────────
    log.info("\n▶ STEP 1: Data Preprocessing")
    log.info("-" * 50)
    from src.data_preprocessing import load_and_prepare
    data = load_and_prepare()
    log.info("✓ Data preprocessed — %d train / %d val / %d test samples",
             len(data["X_train"]), len(data["X_val"]), len(data["X_test"]))

    # ── Step 2: Model Training ───────────────────────────────────────────
    log.info("\n▶ STEP 2: Model Training & Evaluation")
    log.info("-" * 50)
    from src.train_model import train_and_select
    results = train_and_select(data=data)
    log.info("✓ Model trained — selected: %s", results["selected_model"])

    # ── Step 3: SHAP Explainability ──────────────────────────────────────
    log.info("\n▶ STEP 3: SHAP Explainability")
    log.info("-" * 50)
    from src.explainability import AQIExplainer

    explainer = AQIExplainer()
    X_test = data["X_test"]

    # Global SHAP
    explainer.compute_shap_values(X_test)
    imp_df = explainer.global_feature_importance()
    explainer.plot_summary()
    explainer.plot_feature_importance_bar()
    log.info("✓ Global SHAP analysis complete")

    # Local SHAP for first test sample
    sample = X_test.iloc[[0]]
    local_result = explainer.explain_single(sample, plot_prefix="example")
    log.info("✓ Local SHAP explanation generated")
    print("\n" + local_result["narrative"])

    # ── Step 4: Knowledge Base ───────────────────────────────────────────
    log.info("\n▶ STEP 4: WHO Knowledge Base")
    log.info("-" * 50)
    try:
        from src.knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        kb.build()
        test_results = kb.retrieve("PM2.5 health effects respiratory cardiovascular", top_k=3)
        log.info("✓ Knowledge base ready — %d chunks indexed, test query returned %d results",
                 len(kb.chunks), len(test_results))
    except ImportError as e:
        log.warning("⚠ Knowledge base skipped: %s", e)
        kb = None

    # ── Step 5: Full Prediction ──────────────────────────────────────────
    log.info("\n▶ STEP 5: Full Prediction + Explanation")
    log.info("-" * 50)
    from src.predict import predict_and_explain, _print_result

    bg = X_test.sample(n=min(100, len(X_test)), random_state=42)
    result = predict_and_explain(
        sample=sample,
        background_data=bg,
        use_groq=True,
    )
    _print_result(result)

    # ── Done ─────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 68)
    log.info("  ✓ ALL STEPS COMPLETE")
    log.info("=" * 68)
    log.info("  Model      : models/best_aqi_model.joblib")
    log.info("  Metrics    : models/evaluation_metrics.json")
    log.info("  SHAP plots : outputs/shap/")
    log.info("  Prediction : outputs/latest_prediction.json")
    log.info("=" * 68)


if __name__ == "__main__":
    main()

