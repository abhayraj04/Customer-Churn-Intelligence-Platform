"""
Customer Churn Intelligence Platform
Batch Scoring Command-Line Utility (v2.0.0 Production Release)

Usage:
    python src/batch_score.py --input data/WA_Fn-UseC_-Telco-Customer-Churn.csv --output reports/batch_scored_customers.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("batch-scorer")

src_dir = Path(__file__).parent
if str(src_dir.resolve()) not in sys.path:
    sys.path.insert(0, str(src_dir.resolve()))

from pipeline_components import TelcoFeatureEngineer, get_column_transformer


def resolve_model_and_threshold(custom_threshold: float = None):
    """Loads champion model and its verified threshold; falls back to baseline if needed."""
    champ_model = Path("models/phase2_churn_pipeline.joblib")
    champ_metrics = Path("models/phase2_metrics.json")
    base_model = Path("models/final_churn_pipeline.joblib")
    base_metrics = Path("models/model_metrics.json")

    if champ_model.exists() and champ_metrics.exists():
        model_path = champ_model
        with open(champ_metrics, "r") as f:
            m = json.load(f)
        default_threshold = float(m["threshold_tuning"]["best_threshold"])
        model_name = m.get("selected_model", "XGBoost (Tuned)")
    elif base_model.exists():
        model_path = base_model
        with open(base_metrics, "r") as f:
            m = json.load(f)
        default_threshold = float(m["threshold_tuning"]["best_threshold"])
        model_name = m.get("selected_model", "Logistic Regression (Default)")
    else:
        raise FileNotFoundError("No serialized model artifact found in models/")

    threshold = custom_threshold if custom_threshold is not None else default_threshold
    pipeline = joblib.load(model_path)
    return pipeline, threshold, model_name


def batch_score(input_csv: Path, output_csv: Path, threshold: float = None):
    print("=" * 80)
    print("CUSTOMER CHURN INTELLIGENCE PLATFORM - BATCH SCORING UTILITY")
    print("=" * 80)
    print(f"Reading input records from: {input_csv.resolve()}")

    pipeline, active_threshold, model_name = resolve_model_and_threshold(threshold)
    print(f"Loaded active model: {model_name} (Operational Threshold: {active_threshold:.2f})")

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df):,} customer records.")

    print("Computing churn probabilities using pipeline...")
    probs = pipeline.predict_proba(df)[:, 1]

    # Assign risk tiers and predictions
    df_output = df.copy()
    df_output["Churn_Probability"] = probs.round(4)
    df_output["Churn_Prediction"] = (probs >= active_threshold).astype(int)

    def get_tier(p):
        if p >= 0.60:
            return "HIGH RISK"
        elif p >= active_threshold:
            return "MODERATE RISK"
        return "LOW RISK"

    def get_action(p):
        if p >= 0.60:
            return "Urgent 1-on-1 account review, bundle upgrade, or price discount"
        elif p >= active_threshold:
            return "Automated retention email with autopay discount / free TechSupport"
        return "Standard loyalty communications and feature updates"

    df_output["Risk_Tier"] = [get_tier(p) for p in probs]
    df_output["Recommended_Action"] = [get_action(p) for p in probs]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_csv, index=False)
    print(f"[OK] Scored records saved to: {output_csv.resolve()}")

    high_risk_count = (df_output["Risk_Tier"] == "HIGH RISK").sum()
    mod_risk_count = (df_output["Risk_Tier"] == "MODERATE RISK").sum()
    low_risk_count = (df_output["Risk_Tier"] == "LOW RISK").sum()

    print("\n--- BATCH SUMMARY ---")
    print(f"High Risk Accounts:     {high_risk_count:,} ({high_risk_count/len(df)*100:.1f}%)")
    print(f"Moderate Risk Accounts: {mod_risk_count:,} ({mod_risk_count/len(df)*100:.1f}%)")
    print(f"Low Risk Accounts:      {low_risk_count:,} ({low_risk_count/len(df)*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a customer CSV file with the Churn Intelligence Pipeline.")
    parser.add_argument("--input", type=Path, default=Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"), help="Path to input CSV.")
    parser.add_argument("--output", type=Path, default=Path("reports/batch_scored_customers.csv"), help="Path to output CSV.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional decision threshold override (defaults to model's optimal threshold).")
    args = parser.parse_args()

    batch_score(args.input, args.output, args.threshold)
