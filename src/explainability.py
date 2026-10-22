"""
Customer Churn Intelligence Platform
Model Explainability & Feature Contribution Engine

Provides both Global Interpretability (Odds Ratios & Feature Weights)
and Local Interpretability (Customer-Level Log-Odds Contributions & Talking Points).
"""

import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

src_dir = Path(__file__).parent
if str(src_dir.resolve()) not in sys.path:
    sys.path.insert(0, str(src_dir.resolve()))

from pipeline_components import TelcoFeatureEngineer, get_column_transformer

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.titlesize": 13,
    "figure.dpi": 150
})


def extract_global_feature_effects(pipeline_path: Path = Path("models/final_churn_pipeline.joblib")):
    """Extracts coefficients and odds ratios from the fitted Logistic Regression pipeline."""
    pipeline = joblib.load(pipeline_path)
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    feature_names = list(preprocessor.get_feature_names_out())
    coefs = clf.coef_[0]
    intercept = clf.intercept_[0]

    df_coef = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "odds_ratio": np.exp(coefs),
        "abs_importance": np.abs(coefs)
    }).sort_values(by="coefficient", ascending=False)

    return df_coef, intercept, pipeline


def explain_single_customer(pipeline, customer_df: pd.DataFrame, top_n: int = 5):
    """
    Computes exact local log-odds contributions (w_i * x_i) for an individual customer.
    Returns:
    - Base intercept
    - Predicted probability
    - Top risk accelerators (pushing toward churn)
    - Top protective factors (pushing toward retention)
    """
    fe = pipeline.named_steps["feature_engineer"]
    prep = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    # Transform through pipeline stages
    engineered = fe.transform(customer_df)
    transformed_X = prep.transform(engineered)

    feature_names = prep.get_feature_names_out()
    coefs = clf.coef_[0]
    intercept = float(clf.intercept_[0])

    row_vals = transformed_X[0]
    contributions = row_vals * coefs

    df_contrib = pd.DataFrame({
        "feature": feature_names,
        "raw_value": row_vals,
        "coefficient": coefs,
        "contribution": contributions
    })

    prob = float(pipeline.predict_proba(customer_df)[0, 1])

    # Separate risk drivers from protective anchors
    risk_drivers = df_contrib[df_contrib["contribution"] > 0].sort_values(by="contribution", ascending=False).head(top_n)
    protective_factors = df_contrib[df_contrib["contribution"] < 0].sort_values(by="contribution", ascending=True).head(top_n)

    return {
        "churn_probability": prob,
        "intercept": intercept,
        "risk_drivers": risk_drivers.to_dict(orient="records"),
        "protective_factors": protective_factors.to_dict(orient="records"),
        "all_contributions": df_contrib
    }


def generate_explainability_plots(df_coef: pd.DataFrame, pipeline, figures_dir: Path):
    """Generates global odds ratio chart and a sample local waterfall plot."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Global Odds Ratios Plot (Top 10 Accelerators + Top 10 Protective)
    top_pos = df_coef.head(10)
    top_neg = df_coef.tail(10)
    df_plot = pd.concat([top_pos, top_neg]).sort_values(by="coefficient", ascending=True)

    colors = ["#2b5c8f" if c < 0 else "#d95f02" for c in df_plot["coefficient"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(df_plot)), df_plot["coefficient"], color=colors, edgecolor="black", height=0.65)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot["feature"], fontweight="bold")
    ax.axvline(0, color="black", linestyle="--", lw=1.2)
    ax.set_xlabel("Logistic Regression Coefficient (Log-Odds Impact)")
    ax.set_title("Global Feature Effects: Top Churn Drivers vs. Retention Anchors", pad=15)

    # Add text labels
    for bar, (_, row) in zip(bars, df_plot.iterrows()):
        w = bar.get_width()
        or_val = row["odds_ratio"]
        if w > 0:
            ax.text(w + 0.03, bar.get_y() + bar.get_height()/2, f"+{w:.2f} (OR: {or_val:.2f}x)", va="center", fontsize=8, fontweight="bold")
        else:
            ax.text(w - 0.03, bar.get_y() + bar.get_height()/2, f"{w:.2f} (OR: {or_val:.2f}x)", va="center", ha="right", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig16_path = figures_dir / "16_odds_ratios.png"
    fig.savefig(fig16_path, dpi=200)
    plt.close(fig)

    # 2. Sample Local Waterfall Explanation Plot
    sample_raw = pd.DataFrame([{
        "customerID": "DEMO-7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }])

    local_exp = explain_single_customer(pipeline, sample_raw)
    all_c = local_exp["all_contributions"].sort_values(by="contribution", ascending=True)
    notable = pd.concat([all_c.head(5), all_c.tail(5)]).drop_duplicates()

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = ["#2b5c8f" if c < 0 else "#d95f02" for c in notable["contribution"]]
    ax.barh(range(len(notable)), notable["contribution"], color=bar_colors, edgecolor="black")
    ax.set_yticks(range(len(notable)))
    ax.set_yticklabels(notable["feature"], fontweight="bold")
    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.set_xlabel("Log-Odds Contribution to Churn Prediction")
    ax.set_title(f"Individual Customer Explanation Breakdown (Predicted Churn Risk: {local_exp['churn_probability']*100:.1f}%)", pad=15)
    plt.tight_layout()
    fig17_path = figures_dir / "17_local_waterfall_example.png"
    fig.savefig(fig17_path, dpi=200)
    plt.close(fig)

    return fig16_path, fig17_path


def main():
    print("=" * 80)
    print("CUSTOMER CHURN INTELLIGENCE PLATFORM - MODEL EXPLAINABILITY ENGINE")
    print("=" * 80)

    df_coef, intercept, pipeline = extract_global_feature_effects()
    print(f"[OK] Extracted coefficients for {len(df_coef)} features (Intercept: {intercept:.4f}).")

    # Save coefficients table
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    coef_csv = models_dir / "model_coefficients.csv"
    df_coef.to_csv(coef_csv, index=False)
    print(f"[OK] Feature coefficients table saved to: {coef_csv.resolve()}")

    figures_dir = Path("reports/figures")
    fig16, fig17 = generate_explainability_plots(df_coef, pipeline, figures_dir)
    print(f"[OK] Created: {fig16}")
    print(f"[OK] Created: {fig17}")

    print("\n--- TOP 5 CHURN ACCELERATORS (HIGHEST ODDS RATIOS) ---")
    for _, row in df_coef.head(5).iterrows():
        print(f"  + {row['feature']:<30} | Coef: {row['coefficient']:+.4f} | Odds Ratio: {row['odds_ratio']:.2f}x")

    print("\n--- TOP 5 RETENTION ANCHORS (LOWEST ODDS RATIOS) ---")
    for _, row in df_coef.tail(5).iterrows():
        print(f"  - {row['feature']:<30} | Coef: {row['coefficient']:+.4f} | Odds Ratio: {row['odds_ratio']:.2f}x")

    print("\n" + "=" * 80)
    print("EXPLAINABILITY ARTIFACTS GENERATED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
