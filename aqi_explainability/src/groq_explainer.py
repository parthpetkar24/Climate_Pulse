"""
================================================================================
  groq_explainer.py  —  LLM-Powered Cited Health Explanations
  ClimatePulse | PCCOE Hackathon  —  MVP 1
================================================================================

Uses the Groq API (fast inference on open-source LLMs) to generate short,
cited health explanations from:
  1. The predicted AQI and its category.
  2. SHAP feature contributions (top influential factors).
  3. Retrieved WHO evidence chunks with page citations.

The LLM is instructed to:
  - Use ONLY the supplied evidence for health claims.
  - Include proper citations (document name + page number).
  - Never invent predictions, contributions, or health facts.
  - Keep the explanation short and nontechnical.

API Key
-------
Set the environment variable  GROQ_API_KEY  before running.
The key is NEVER hardcoded, logged, or printed.
================================================================================
"""

import os
import json
import logging
from typing import Optional

log = logging.getLogger("aqi_explainability.groq_explainer")


# ── Prompt Template ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a concise air quality health advisor for ClimatePulse.
Your role is to explain a predicted AQI value in plain language, using ONLY the
evidence provided below.

RULES:
1. State the predicted AQI and its category.
2. List the 2–3 most influential environmental factors and their effect direction.
3. Explain health implications using ONLY the WHO evidence provided.
4. Give practical health advice appropriate to the AQI category.
5. Cite every health claim with the source document name and page number in
   parentheses — e.g., (9789240034433-eng.pdf, p. 42).
6. Do NOT invent facts, citations, causal claims, or numbers not in the evidence.
7. If the evidence is insufficient for a claim, explicitly say so.
8. Keep the total explanation under 250 words.
9. Distinguish feature *influence* from *causation*: say "associated with" or
   "the model indicates", not "caused by".
10. WHO guideline values are separate from India's CPCB AQI standards.
    Do not treat them as interchangeable.
"""

USER_TEMPLATE = """\
## Prediction Details
- **Predicted AQI**: {aqi_value:.1f}
- **AQI Category**: {aqi_category}

## Top Influential Features (from SHAP analysis)
{shap_section}

## Retrieved WHO Evidence
{evidence_section}

---
Generate a short, cited health explanation based on the above.
"""


def _format_shap_section(top_features: list[dict]) -> str:
    """Format SHAP features into a readable section for the prompt."""
    lines = []
    for f in top_features:
        direction = "↑ pushes AQI higher" if f["direction"] == "increases" else "↓ pulls AQI lower"
        lines.append(
            f"- **{f['feature']}** = {f['feature_value']:.2f}  "
            f"(SHAP: {f['shap_value']:+.2f}, {direction})"
        )
    return "\n".join(lines) if lines else "No SHAP data available."


def _format_evidence_section(evidence: list[dict]) -> str:
    """Format retrieved evidence chunks into a readable section."""
    if not evidence:
        return "No WHO evidence retrieved for this query."

    lines = []
    for i, e in enumerate(evidence, 1):
        lines.append(
            f"### Evidence {i}  [{e['citation']}]  (relevance: {e['score']:.3f})\n"
            f"{e['text'][:800]}"
        )
    return "\n\n".join(lines)


# ── Groq API integration ────────────────────────────────────────────────────

class GroqExplainer:
    """
    Generate cited AQI health explanations using the Groq API.

    Requires the GROQ_API_KEY environment variable.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        self._api_key = os.environ.get("GROQ_API_KEY")
        self._client  = None

        if not self._api_key:
            log.warning(
                "GROQ_API_KEY not set. The explainer will prepare prompts "
                "but cannot call the API. Set the key with:\n"
                "  export GROQ_API_KEY=your_key_here  (Linux/Mac)\n"
                "  set GROQ_API_KEY=your_key_here     (Windows)"
            )

    @property
    def is_available(self) -> bool:
        return self._api_key is not None

    def _get_client(self):
        """Lazy-init the Groq client."""
        if self._client is None:
            try:
                from groq import Groq
            except ImportError:
                raise ImportError(
                    "The groq package is required.  Install with:  pip install groq"
                )
            self._client = Groq(api_key=self._api_key)
        return self._client

    def build_prompt(
        self,
        aqi_value: float,
        aqi_category: str,
        top_features: list[dict],
        evidence: list[dict],
    ) -> tuple[str, str]:
        """
        Build the system and user prompts (useful for inspection even
        without an API key).

        Returns (system_prompt, user_prompt).
        """
        user_prompt = USER_TEMPLATE.format(
            aqi_value=aqi_value,
            aqi_category=aqi_category,
            shap_section=_format_shap_section(top_features),
            evidence_section=_format_evidence_section(evidence),
        )
        return SYSTEM_PROMPT, user_prompt

    def generate_explanation(
        self,
        aqi_value: float,
        aqi_category: str,
        top_features: list[dict],
        evidence: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 600,
    ) -> dict:
        """
        Generate a cited health explanation.

        Returns
        -------
        dict with keys:
            explanation : str   — the generated text
            model       : str   — Groq model used
            prompt      : str   — the user prompt (for traceability)
            api_called  : bool  — whether the API was actually called
        """
        system_prompt, user_prompt = self.build_prompt(
            aqi_value, aqi_category, top_features, evidence,
        )

        if not self.is_available:
            return {
                "explanation": (
                    "[Groq API key not configured. Below is the prompt that "
                    "would be sent to the LLM.]\n\n"
                    f"--- SYSTEM PROMPT ---\n{system_prompt}\n\n"
                    f"--- USER PROMPT ---\n{user_prompt}"
                ),
                "model":      self.model_name,
                "prompt":     user_prompt,
                "api_called": False,
            }

        client = self._get_client()
        log.info("Calling Groq API (model=%s) ...", self.model_name)

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        explanation = response.choices[0].message.content
        log.info("Groq response received (%d chars)", len(explanation))

        return {
            "explanation": explanation,
            "model":       self.model_name,
            "prompt":      user_prompt,
            "api_called":  True,
        }


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Demo with mock data
    explainer = GroqExplainer()

    mock_features = [
        {"feature": "pm25", "shap_value": 45.2, "feature_value": 120.5, "direction": "increases"},
        {"feature": "pm10", "shap_value": 12.1, "feature_value": 180.0, "direction": "increases"},
        {"feature": "no2",  "shap_value": -3.5, "feature_value": 25.0,  "direction": "decreases"},
    ]

    mock_evidence = [
        {
            "text": "PM2.5 particles can penetrate deep into the lungs and enter the bloodstream. "
                    "Long-term exposure is associated with cardiovascular and respiratory disease.",
            "citation": "9789240034433-eng.pdf, p. 12",
            "source": "9789240034433-eng.pdf",
            "pages": [12],
            "score": 0.85,
        },
    ]

    result = explainer.generate_explanation(
        aqi_value=185.0,
        aqi_category="Moderate",
        top_features=mock_features,
        evidence=mock_evidence,
    )

    print("\n--- Generated Explanation ---")
    print(result["explanation"])
    print(f"\nAPI called: {result['api_called']}")

