"""
================================================================================
  explainability.py  —  SHAP-based AQI Model Explainability
  ClimatePulse | PCCOE Hackathon  —  MVP 1
================================================================================

Generates global and local SHAP explanations for the trained AQI model.

Features
--------
* Global feature importance (bar chart).
* SHAP summary (beeswarm) plot.
* Local explanations for individual predictions (waterfall / force).
* Human-readable narrative from SHAP contributions.

Important caveat
----------------
SHAP values indicate feature *influence* on the model's prediction, NOT causal
proof that a pollutant caused an AQI change.  The accompanying text always
distinguishes "associated with" / "influential" from "caused."
================================================================================
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")                        # headless — no GUI needed
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    shap = None

log = logging.getLogger("aqi_explainability.explainability")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs" / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── AQI categories (India CPCB) ─────────────────────────────────────────────
AQI_CATEGORIES = [
    (0,   50,  "Good"),
    (51,  100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]

def aqi_category(aqi_value: float) -> str:
    """Map a numeric AQI value to its CPCB category string."""
    if pd.isna(aqi_value):
        return "Unknown"
    for lo, hi, label in AQI_CATEGORIES:
        if lo <= aqi_value <= hi:
            return label
    return "Severe" if aqi_value > 500 else "Unknown"


# ── Core class ───────────────────────────────────────────────────────────────

class AQIExplainer:
    """
    SHAP explainer wrapper for the AQI model.

    Parameters
    ----------
    model_path : Path or str, optional
        Path to the saved joblib model.  Defaults to models/best_aqi_model.joblib.
    """

    def __init__(self, model_path: Optional[Path] = None):
        if shap is None:
            raise ImportError(
                "SHAP is required for explainability.  "
                "Install with:  pip install shap"
            )

        self.model_path = Path(model_path) if model_path else MODEL_DIR / "best_aqi_model.joblib"
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = joblib.load(self.model_path)
        log.info("Loaded model from %s", self.model_path)

        # Load feature names
        fn_path = MODEL_DIR / "feature_names.json"
        with open(fn_path) as f:
            self.feature_names = json.load(f)

        self._explainer = None
        self._shap_values = None

    # ── SHAP explainer (lazy init) ───────────────────────────────────────

    def _get_explainer(self, X_background: pd.DataFrame):
        """Create a TreeExplainer using a background sample."""
        if self._explainer is None:
            log.info("Initialising SHAP TreeExplainer ...")
            self._explainer = shap.TreeExplainer(
                self.model,
                data=X_background,
                feature_perturbation="interventional",
            )
        return self._explainer

    # ── Global explanations ──────────────────────────────────────────────

    def compute_shap_values(self, X: pd.DataFrame, background: Optional[pd.DataFrame] = None):
        """
        Compute SHAP values for every row in X.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same columns as training).
        background : pd.DataFrame, optional
            Background sample for the explainer.  If not supplied, a random
            sample of min(100, len(X)) rows from X is used.
        """
        if background is None:
            n_bg = min(100, len(X))
            background = X.sample(n=n_bg, random_state=42)

        explainer = self._get_explainer(background)
        log.info("Computing SHAP values for %d samples ...", len(X))
        self._shap_values = explainer.shap_values(X)
        self._X_for_shap  = X
        return self._shap_values

    def global_feature_importance(self, save: bool = True) -> pd.DataFrame:
        """
        Mean |SHAP| importance for each feature.

        Returns a DataFrame sorted by importance.
        """
        if self._shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")

        importance = np.abs(self._shap_values).mean(axis=0)
        imp_df = pd.DataFrame({
            "feature":    self.feature_names,
            "mean_abs_shap": importance,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        if save:
            imp_df.to_csv(OUTPUT_DIR / "global_feature_importance.csv", index=False)
            log.info("Global importance saved → %s", OUTPUT_DIR / "global_feature_importance.csv")

        return imp_df

    def plot_summary(self, save: bool = True):
        """SHAP beeswarm summary plot."""
        if self._shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            self._shap_values,
            self._X_for_shap,
            feature_names=self.feature_names,
            show=False,
        )
        plt.tight_layout()
        if save:
            path = OUTPUT_DIR / "shap_summary_beeswarm.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            log.info("Summary plot saved → %s", path)
        plt.close()

    def plot_feature_importance_bar(self, top_n: int = 15, save: bool = True):
        """Bar plot of mean |SHAP| importance."""
        imp_df = self.global_feature_importance(save=False)
        top = imp_df.head(top_n)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#2196F3")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Top Feature Importance (SHAP)")
        plt.tight_layout()
        if save:
            path = OUTPUT_DIR / "shap_feature_importance_bar.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            log.info("Bar plot saved → %s", path)
        plt.close()

    # ── Local explanations ───────────────────────────────────────────────

    def explain_single(
        self,
        X_row: pd.DataFrame,
        background: Optional[pd.DataFrame] = None,
        save_plot: bool = True,
        plot_prefix: str = "local",
    ) -> dict:
        """
        Explain one prediction.

        Parameters
        ----------
        X_row : pd.DataFrame
            Single-row feature DataFrame.
        background : pd.DataFrame, optional
            Background for the explainer.

        Returns
        -------
        dict with keys:
            prediction, aqi_category, shap_values, top_features,
            narrative (human-readable string).
        """
        if background is not None:
            explainer = self._get_explainer(background)
        elif self._explainer is not None:
            explainer = self._explainer
        else:
            raise RuntimeError(
                "Either pass a background DataFrame or call compute_shap_values() first."
            )

        prediction = float(self.model.predict(X_row)[0])
        sv = explainer.shap_values(X_row)[0]
        base_value = float(explainer.expected_value)

        # Top contributing features
        contrib = sorted(
            zip(self.feature_names, sv, X_row.iloc[0].values),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        top_features = [
            {
                "feature": name,
                "shap_value": round(float(shap_val), 4),
                "feature_value": round(float(feat_val), 4),
                "direction": "increases" if shap_val > 0 else "decreases",
            }
            for name, shap_val, feat_val in contrib[:5]
        ]

        category = aqi_category(prediction)

        # Human-readable narrative
        narrative = self._build_narrative(prediction, category, top_features)

        # Waterfall plot
        if save_plot:
            try:
                explanation = shap.Explanation(
                    values=sv,
                    base_values=base_value,
                    data=X_row.iloc[0].values,
                    feature_names=self.feature_names,
                )
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(explanation, show=False)
                plt.tight_layout()
                path = OUTPUT_DIR / f"{plot_prefix}_waterfall.png"
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                log.info("Waterfall plot saved → %s", path)
            except Exception as e:
                log.warning("Could not generate waterfall plot: %s", e)

        return {
            "prediction":   round(prediction, 2),
            "aqi_category": category,
            "base_value":   round(base_value, 2),
            "shap_values":  {name: round(float(v), 4) for name, v in zip(self.feature_names, sv)},
            "top_features": top_features,
            "narrative":    narrative,
        }

    # ── Narrative builder ────────────────────────────────────────────────

    @staticmethod
    def _build_narrative(prediction: float, category: str, top_features: list) -> str:
        """
        Build a short human-readable narrative from SHAP results.

        Uses language that distinguishes *influence* from *causation*.
        """
        lines = [
            f"Predicted AQI: {prediction:.1f} ({category})",
            "",
            "Key factors influencing this prediction:",
        ]
        for i, f in enumerate(top_features[:3], 1):
            direction = "pushing AQI higher" if f["direction"] == "increases" else "pulling AQI lower"
            lines.append(
                f"  {i}. {f['feature']} = {f['feature_value']:.2f}  →  "
                f"{direction} (SHAP contribution: {f['shap_value']:+.2f})"
            )

        lines.append("")
        lines.append(
            "Note: SHAP values indicate how much each feature influenced the "
            "model's prediction. They do not prove that a pollutant directly "
            "caused an AQI change — external factors and interactions may also "
            "play a role."
        )
        return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.data_preprocessing import load_and_prepare

    # Load data
    data = load_and_prepare()
    X_test, y_test = data["X_test"], data["y_test"]

    # Create explainer
    explainer = AQIExplainer()
    explainer.compute_shap_values(X_test)

    # Global plots
    imp_df = explainer.global_feature_importance()
    print("\nTop 10 features by mean |SHAP|:")
    print(imp_df.head(10).to_string(index=False))

    explainer.plot_summary()
    explainer.plot_feature_importance_bar()

    # Local explanation for first test sample
    sample = X_test.iloc[[0]]
    result = explainer.explain_single(sample, plot_prefix="example")
    print("\n" + result["narrative"])
    print(f"\n✓ All SHAP outputs saved to {OUTPUT_DIR}")

