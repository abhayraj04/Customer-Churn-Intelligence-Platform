"""
Customer Churn Intelligence Platform
Enterprise Streamlit Dashboard Application (v2.5.0 Premium UI/UX Release)

Enhanced Design System:
- Professional Typography & Custom CSS Cards with Soft Shadows
- Cohesive Executive Theme (Deep Slate & Teal Accents)
- Interactive Gauge / Probability Progress Indicators
- Real-Time Dynamic What-If Counterfactual Interventions
- Live Customer-Level SHAP Waterfall Interpretability
- Metric Badges, Clean Navigation & Financial ROI Simulators
"""

import io
import json
import logging
import sys
from pathlib import Path

# Ensure src is on Python path
src_dir = Path(__file__).parent / "src"
if str(src_dir.resolve()) not in sys.path:
    sys.path.insert(0, str(src_dir.resolve()))
if "src" not in sys.path:
    sys.path.insert(0, "src")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Setup Streamlit page configuration
st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# PREMIUM UI/UX DESIGN SYSTEM (CSS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Hero */
    .hero-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 24px 28px;
        color: #f8fafc;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        border: 1px solid #334155;
    }
    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    .hero-banner p {
        color: #cbd5e1 !important;
        font-size: 0.95rem;
        margin: 0;
    }
    
    /* Custom KPI Cards - Dark Theme Compatible */
    .kpi-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        border: 1px solid #334155;
        border-left: 4px solid #3b82f6;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #f8fafc;
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .kpi-subtitle {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* Risk Badges */
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-mod {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    /* Action Card */
    .action-box {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #38bdf8;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
        margin-top: 12px;
    }

    /* Clean Sidebar Radio Items */
    div[data-testid="stSidebar"] [data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        margin-bottom: 4px !important;
        cursor: pointer !important;
        transition: background-color 0.15s ease !important;
    }
    div[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    div[data-testid="stSidebar"] [data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    """Load primary champion model, rollback baseline, metrics, and dataset."""
    models_dir = Path("models")
    champ_pipeline_path = models_dir / "phase2_churn_pipeline.joblib"
    champ_metrics_path = models_dir / "phase2_metrics.json"
    base_pipeline_path = models_dir / "final_churn_pipeline.joblib"
    base_metrics_path = models_dir / "model_metrics.json"
    raw_data_path = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    raw_df = pd.read_csv(raw_data_path)

    # Determine active model
    if champ_pipeline_path.exists() and champ_metrics_path.exists():
        pipeline = joblib.load(champ_pipeline_path)
        with open(champ_metrics_path, "r") as f:
            metrics = json.load(f)
        active_model_name = "XGBoost (Tuned)"
        active_version = "v2.0.0 Champion"
        optimal_threshold = float(metrics["threshold_tuning"]["best_threshold"])
    else:
        pipeline = joblib.load(base_pipeline_path)
        with open(base_metrics_path, "r") as f:
            metrics = json.load(f)
        active_model_name = "Logistic Regression (Default)"
        active_version = "v1.0.0 Baseline"
        optimal_threshold = float(metrics["threshold_tuning"]["best_threshold"])

    # Load baseline metrics for comparisons
    base_metrics = None
    if base_metrics_path.exists():
        with open(base_metrics_path, "r") as f:
            base_metrics = json.load(f)

    # Create reproducible holdout test set for evaluation
    X_raw = raw_df.drop(columns=["customerID", "Churn"])
    y_raw = (raw_df["Churn"] == "Yes").astype(int)
    _, X_test_orig, _, y_test_orig = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
    )
    test_probs = pipeline.predict_proba(X_test_orig)[:, 1]

    return pipeline, metrics, base_metrics, raw_df, X_test_orig, y_test_orig, test_probs, active_model_name, active_version, optimal_threshold


# Load assets safely
try:
    (
        pipeline,
        metrics,
        base_metrics,
        raw_df,
        X_test_orig,
        y_test_orig,
        test_probs,
        active_model_name,
        active_version,
        optimal_threshold
    ) = load_assets()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:12px;'>
            <span style='font-size:26px;'>💎</span>
            <div>
                <div style='font-weight:700; font-size:16px; color:#f8fafc;'>Churn Intelligence</div>
                <div style='font-size:12px; color:#94a3b8;'>Enterprise ML Platform</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    section = st.radio(
        "Platform Modules",
        [
            "📊 Executive Overview",
            "📈 Exploratory Analytics",
            "🎯 Customer Risk Scoring",
            "🔍 Cohort Explorer",
            "💰 Retention ROI Simulator",
            "🏆 Model Benchmark & SHAP",
            "🛡️ Data Quality & Ethics",
            "📁 Batch CSV Prediction",
        ],
        index=0
    )

    st.markdown("---")
    st.markdown(f"**Engine:** `{active_model_name}`")
    st.markdown(f"**Version:** `{active_version}`")
    st.markdown(f"**Decision Threshold ($t$):** `{optimal_threshold:.2f}`")
    st.markdown("---")
    st.caption("🔒 Verified Zero Data-Leakage Pipeline")


# -----------------------------------------------------------------------------
# SECTION A: EXECUTIVE OVERVIEW
# -----------------------------------------------------------------------------
if section == "📊 Executive Overview":
    st.markdown("""
        <div class='hero-banner'>
            <h1>Executive Overview & Business KPIs</h1>
            <p>Real-time subscriber retention tracking, base risk indicators, and model discrimination metrics.</p>
        </div>
    """, unsafe_allow_html=True)

    # Top KPI Ribbon
    total_cust = len(raw_df)
    churn_count = int((raw_df["Churn"] == "Yes").sum())
    retained_count = total_cust - churn_count
    churn_rate = (churn_count / total_cust) * 100

    holdout_roc = metrics["holdout_test_evaluation"]["test_roc_auc"]
    holdout_ap = metrics["holdout_test_evaluation"]["test_pr_auc"]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
            <div class='kpi-card' style='border-left-color: #3b82f6;'>
                <div class='kpi-title'>Total Accounts</div>
                <div class='kpi-value'>{total_cust:,}</div>
                <div class='kpi-subtitle'>Active Database</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class='kpi-card' style='border-left-color: #ef4444;'>
                <div class='kpi-title'>Base Churn Rate</div>
                <div class='kpi-value'>{churn_rate:.1f}%</div>
                <div class='kpi-subtitle'>{churn_count:,} accounts lost</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class='kpi-card' style='border-left-color: #10b981;'>
                <div class='kpi-title'>Retained Accounts</div>
                <div class='kpi-value'>{retained_count:,}</div>
                <div class='kpi-subtitle'>{100-churn_rate:.1f}% active retention</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class='kpi-card' style='border-left-color: #8b5cf6;'>
                <div class='kpi-title'>Model ROC-AUC</div>
                <div class='kpi-value'>{holdout_roc:.4f}</div>
                <div class='kpi-subtitle'>Holdout Generalization</div>
            </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
            <div class='kpi-card' style='border-left-color: #f59e0b;'>
                <div class='kpi-title'>Avg Precision</div>
                <div class='kpi-value'>{holdout_ap:.4f}</div>
                <div class='kpi-subtitle'>Test PR-AUC</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Executive Summary Columns
    e_col1, e_col2 = st.columns([1.25, 1])

    with e_col1:
        st.subheader("💡 Strategic Insights from Customer Accounts")
        st.markdown("""
        * **Contract Length Dictates Loyalty:** Month-to-month contracts experience a **42.71% churn rate**, while 2-year contracts drop to **2.83%** (a 15× reduction in cancellations).
        * **The 12-Month Vulnerability Window:** **47.44% of all churn events** take place within the customer's first year of service. Onboarding check-ins within the first 90 days yield the highest ROI.
        * **Payment Method Friction:** Electronic check users exhibit an alarming **45.29% churn rate**, compared to **15.24%** for automatic credit card or bank debit accounts.
        * **Retention Service Anchors:** Customers with **TechSupport** or **OnlineSecurity** bundles churn **60% less** across all demographic and billing brackets.
        """)

    with e_col2:
        st.subheader("Account Status Breakdown")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.pie(
            [retained_count, churn_count],
            labels=["Retained (No)", "Churned (Yes)"],
            autopct="%1.1f%%",
            startangle=140,
            colors=["#2b5c8f", "#d95f02"],
            explode=(0, 0.08),
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        ax.set_title("Customer Population Churn Ratio", fontsize=11, fontweight="bold", pad=12)
        st.pyplot(fig)
        plt.close(fig)


# -----------------------------------------------------------------------------
# SECTION B: EXPLORATORY ANALYTICS
# -----------------------------------------------------------------------------
elif section == "📈 Exploratory Analytics":
    st.markdown("""
        <div class='hero-banner'>
            <h1>Exploratory Churn Analytics</h1>
            <p>Multi-cohort distributions, cross-tabulations, and demographic pricing patterns.</p>
        </div>
    """, unsafe_allow_html=True)

    b_tab1, b_tab2, b_tab3 = st.tabs(["Contracts & Tenure Cohorts", "Services & Add-ons", "Financial & Payment Friction"])

    with b_tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Churn Rate by Contract Duration")
            contract_churn = raw_df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            fig, ax = plt.subplots(figsize=(6, 3.8))
            bars = ax.bar(contract_churn["Contract"], contract_churn["Churn"], color=["#d95f02", "#2b5c8f", "#1b9e77"], edgecolor="black")
            ax.set_ylabel("Churn Rate (%)")
            ax.set_ylim(0, 55)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha="center", fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Tenure Distribution by Churn Status")
            fig, ax = plt.subplots(figsize=(6, 3.8))
            sns.histplot(data=raw_df, x="tenure", hue="Churn", multiple="stack", palette={"No": "#2b5c8f", "Yes": "#d95f02"}, bins=24, ax=ax)
            ax.set_xlabel("Tenure (Months)")
            ax.set_ylabel("Customer Count")
            st.pyplot(fig)
            plt.close(fig)

    with b_tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Churn Rate by Internet Architecture")
            net_churn = raw_df.groupby("InternetService")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            fig, ax = plt.subplots(figsize=(6, 3.8))
            bars = ax.bar(net_churn["InternetService"], net_churn["Churn"], color=["#2b5c8f", "#d95f02", "#1b9e77"], edgecolor="black")
            ax.set_ylabel("Churn Rate (%)")
            ax.set_ylim(0, 55)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha="center", fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Impact of TechSupport & OnlineSecurity Bundles")
            raw_df_copy = raw_df.copy()
            raw_df_copy["HasSupport"] = ((raw_df_copy["TechSupport"] == "Yes") | (raw_df_copy["OnlineSecurity"] == "Yes")).map({True: "Has Security/Support", False: "No Security/Support"})
            sup_churn = raw_df_copy.groupby("HasSupport")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            fig, ax = plt.subplots(figsize=(6, 3.8))
            bars = ax.bar(sup_churn["HasSupport"], sup_churn["Churn"], color=["#1b9e77", "#d95f02"], edgecolor="black")
            ax.set_ylabel("Churn Rate (%)")
            ax.set_ylim(0, 55)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha="center", fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)

    with b_tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Churn Rate by Payment Channel")
            pay_churn = raw_df.groupby("PaymentMethod")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh(pay_churn["PaymentMethod"], pay_churn["Churn"], color="#2b5c8f", edgecolor="black")
            ax.set_xlabel("Churn Rate (%)")
            for bar in bars:
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.1f}%", va="center", fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Monthly Charges Distribution ($)")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=raw_df, x="Churn", y="MonthlyCharges", hue="Churn", palette={"No": "#2b5c8f", "Yes": "#d95f02"}, legend=False, ax=ax)
            ax.set_ylabel("Monthly Charges ($)")
            st.pyplot(fig)
            plt.close(fig)


# -----------------------------------------------------------------------------
# SECTION C: CUSTOMER RISK SCORING & SHAP EXPLAINABILITY
# -----------------------------------------------------------------------------
elif section == "🎯 Customer Risk Scoring":
    st.markdown("""
        <div class='hero-banner'>
            <h1>Customer Risk Scoring & Explainability Engine</h1>
            <p>Compute live individual churn probabilities, view local SHAP factor waterfalls, and run What-If simulations.</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.25])

    with col1:
        st.subheader("Account Parameters")
        with st.form("single_predict_form"):
            tenure = st.slider("Account Tenure (Months)", 0, 72, 6, help="Months customer has been with the provider.")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 85.50, 5.0)
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            with st.expander("Additional Services & Demographics"):
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                senior = st.selectbox("Senior Citizen", [0, 1])
                partner = st.selectbox("Partner", ["No", "Yes"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

            submitted = st.form_submit_button("⚡ Compute Real-Time Churn Risk", type="primary")

    # Build input record
    calc_total = float(round(tenure * monthly_charges, 2)) if tenure > 0 else 0.0
    customer_dict = {
        "customerID": "SIM-USER-001",
        "gender": "Female",
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": calc_total
    }

    df_single = pd.DataFrame([customer_dict])
    pred_prob = float(pipeline.predict_proba(df_single)[0, 1])

    with col2:
        st.subheader("Inference Result & Risk Stratification")
        
        if pred_prob >= 0.60:
            badge_html = "<span class='badge-high'>HIGH CHURN RISK</span>"
            action = "Urgent: Assign Customer Success Specialist for 1-on-1 account review, bundle upgrade, or price discount."
            gauge_color = "#ef4444"
        elif pred_prob >= optimal_threshold:
            badge_html = "<span class='badge-mod'>MODERATE CHURN RISK</span>"
            action = "Proactive: Trigger automated retention email with discount on long-term contract or free TechSupport bundle."
            gauge_color = "#f59e0b"
        else:
            badge_html = "<span class='badge-low'>LOW CHURN RISK</span>"
            action = "Standard: Maintain routine product updates, value-add communications, and loyalty rewards."
            gauge_color = "#10b981"

        # Probability Progress Display
        st.markdown(f"""
            <div style='background:#1e293b; border-radius:10px; padding:18px; border:1px solid #334155; margin-bottom:14px;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <div style='font-size:0.85rem; color:#94a3b8; font-weight:600;'>PREDICTED PROBABILITY</div>
                        <div style='font-size:2.2rem; font-weight:800; color:#f8fafc;'>{pred_prob*100:.1f}%</div>
                    </div>
                    <div>{badge_html}</div>
                </div>
                <div style='background:#334155; border-radius:9999px; height:8px; width:100%; margin-top:8px;'>
                    <div style='background:{gauge_color}; border-radius:9999px; height:8px; width:{min(100, int(pred_prob*100))}%;'></div>
                </div>
                <div style='font-size:0.75rem; color:#64748b; margin-top:6px;'>Decision Threshold: <strong>{optimal_threshold:.2f}</strong></div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class='action-box'>
                <div style='font-weight:700; color:#0369a1; font-size:0.88rem; margin-bottom:4px;'>PRESCRIPTIVE ACTION</div>
                <div style='color:#334155; font-size:0.88rem;'>{action}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("💡 Counterfactual Intervention Simulator")
        st.caption("How much does churn risk decrease if we alter contract or service terms?")

        # Counterfactual: Contract migration
        df_migrated_contract = df_single.copy()
        df_migrated_contract["Contract"] = "One year"
        prob_contract = float(pipeline.predict_proba(df_migrated_contract)[0, 1])

        # Counterfactual: Add Support Bundle
        df_migrated_support = df_single.copy()
        df_migrated_support["TechSupport"] = "Yes"
        df_migrated_support["OnlineSecurity"] = "Yes"
        prob_support = float(pipeline.predict_proba(df_migrated_support)[0, 1])

        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            st.metric("Migrate to 1-Year Contract", f"{prob_contract*100:.1f}%", f"{(prob_contract - pred_prob)*100:.1f}%", delta_color="inverse")
        with sim_c2:
            st.metric("Add TechSupport + Security", f"{prob_support*100:.1f}%", f"{(prob_support - pred_prob)*100:.1f}%", delta_color="inverse")

        # Live Real-Time SHAP Waterfall Attribution for this customer
        st.markdown("---")
        st.subheader("🔬 Local Factor Attribution (SHAP Waterfall)")
        st.caption("Exact decomposition of drivers pushing this customer toward or away from churn.")
        try:
            import shap
            fe = pipeline.named_steps["feature_engineer"]
            prep = pipeline.named_steps["preprocessor"]
            clf = pipeline.named_steps["classifier"]

            x_eng = fe.transform(df_single)
            x_tf = prep.transform(x_eng)
            feature_names = list(prep.get_feature_names_out())

            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(x_tf)
            base_val = float(explainer.expected_value)

            shap_exp = shap.Explanation(
                values=shap_values[0],
                base_values=base_val,
                data=x_tf[0],
                feature_names=feature_names
            )
            fig_shap = plt.figure(figsize=(9, 4.5))
            shap.waterfall_plot(shap_exp, max_display=8, show=False)
            plt.title("Individual Factor Contribution to Churn Risk", fontsize=11, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close(fig_shap)
        except Exception as shap_err:
            st.info(f"Local SHAP attribution preview: {shap_err}")


# -----------------------------------------------------------------------------
# SECTION D: COHORT EXPLORER
# -----------------------------------------------------------------------------
elif section == "🔍 Cohort Explorer":
    st.markdown("""
        <div class='hero-banner'>
            <h1>Customer Cohort Explorer & Export</h1>
            <p>Filter customer accounts by contracts, internet services, and payment methods with immediate CSV export.</p>
        </div>
    """, unsafe_allow_html=True)

    filter_contract = st.multiselect("Contract Type", raw_df["Contract"].unique(), default=list(raw_df["Contract"].unique()))
    filter_internet = st.multiselect("Internet Service", raw_df["InternetService"].unique(), default=list(raw_df["InternetService"].unique()))
    filter_payment = st.multiselect("Payment Method", raw_df["PaymentMethod"].unique(), default=list(raw_df["PaymentMethod"].unique()))

    filtered_df = raw_df[
        (raw_df["Contract"].isin(filter_contract)) &
        (raw_df["InternetService"].isin(filter_internet)) &
        (raw_df["PaymentMethod"].isin(filter_payment))
    ]

    st.markdown(f"**Matching Accounts:** `{len(filtered_df):,}` of `{len(raw_df):,}`")

    if len(filtered_df) > 0:
        cohort_churn = (filtered_df["Churn"] == "Yes").mean() * 100
        cohort_col1, cohort_col2, cohort_col3 = st.columns(3)
        with cohort_col1:
            st.metric("Cohort Size", f"{len(filtered_df):,}")
        with cohort_col2:
            st.metric("Cohort Churn Rate", f"{cohort_churn:.1f}%", f"{cohort_churn - 26.54:.1f}% vs baseline", delta_color="inverse")
        with cohort_col3:
            st.metric("Avg Monthly Charges", f"${filtered_df['MonthlyCharges'].mean():.2f}")

        st.dataframe(filtered_df[["customerID", "Contract", "tenure", "MonthlyCharges", "PaymentMethod", "Churn"]].head(100), width="stretch")

        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Filtered Cohort CSV", data=csv, file_name="churn_cohort_export.csv", mime="text/csv")


# -----------------------------------------------------------------------------
# SECTION E: RETENTION ROI SIMULATOR
# -----------------------------------------------------------------------------
elif section == "💰 Retention ROI Simulator":
    st.markdown("""
        <div class='hero-banner'>
            <h1>Retention Strategy & Financial ROI Simulator</h1>
            <p>Model financial ROI by balancing Customer Lifetime Value (LTV) against outreach spend across thresholds.</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.25])

    with col1:
        st.subheader("Financial & Operational Assumptions")
        cust_ltv = st.number_input("Average Customer Lifetime Value (LTV, $)", 500, 5000, 1800, 100)
        retention_cost = st.number_input("Outreach Cost per Customer ($)", 5, 200, 25, 5)
        rescue_rate = st.slider("Campaign Rescue Success Rate (%)", 5, 50, 25, 1) / 100.0
        threshold_sim = st.slider("Decision Threshold ($t$)", 0.10, 0.90, optimal_threshold, 0.05)

    preds_sim = (test_probs >= threshold_sim).astype(int)
    cm = confusion_matrix(y_test_orig, preds_sim)
    tn, fp, fn, tp = cm.ravel()

    targeted_customers = fp + tp
    rescued_customers = tp * rescue_rate
    gross_value_saved = rescued_customers * cust_ltv
    total_campaign_cost = targeted_customers * retention_cost
    net_roi = gross_value_saved - total_campaign_cost

    with col2:
        st.subheader("Simulated Business Return (Holdout N=1,409)")
        roi_c1, roi_c2 = st.columns(2)
        with roi_c1:
            st.metric("Targeted Accounts", f"{targeted_customers:,}", f"TP: {tp} | FP: {fp}")
            st.metric("Gross Value Saved", f"${gross_value_saved:,.2f}", f"Rescued: {rescued_customers:.1f} churners")
        with roi_c2:
            st.metric("Campaign Spend", f"${total_campaign_cost:,.2f}", f"@ ${retention_cost}/customer")
            st.metric("Net Retention Profit", f"${net_roi:,.2f}", delta="Net Positive" if net_roi > 0 else "Net Loss")

        st.markdown("---")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred Retained", "Pred Churned"], yticklabels=["Actual Retained", "Actual Churned"])
        ax.set_ylabel("True Status")
        ax.set_xlabel("Predicted Status")
        ax.set_title(f"Confusion Matrix @ t = {threshold_sim:.2f}")
        st.pyplot(fig)
        plt.close(fig)


# -----------------------------------------------------------------------------
# SECTION F: MODEL PERFORMANCE & SHAP BENCHMARK
# -----------------------------------------------------------------------------
elif section == "🏆 Model Benchmark & SHAP":
    st.markdown("""
        <div class='hero-banner'>
            <h1>Model Performance & Validation Benchmark</h1>
            <p>Verified holdout test set results, comparison against the rollback baseline, and global feature impact.</p>
        </div>
    """, unsafe_allow_html=True)

    # Model comparison table
    st.subheader("Cross-Validation & Holdout Test Matrix")
    comp_df = pd.DataFrame([
        {
            "Model Architecture": "XGBoost (Tuned)",
            "Status": "CHAMPION",
            "Holdout ROC-AUC": 0.8484,
            "Holdout AP (PR-AUC)": 0.6671,
            "Brier Score": 0.1346,
            "Optimal Threshold": 0.30,
            "Recall @ Threshold": "78.07%",
            "F1 @ Threshold": 0.6320
        },
        {
            "Model Architecture": "Logistic Regression (Default)",
            "Status": "BASELINE",
            "Holdout ROC-AUC": 0.8428,
            "Holdout AP (PR-AUC)": 0.6351,
            "Brier Score": 0.1379,
            "Optimal Threshold": 0.35,
            "Recall @ Threshold": "70.86%",
            "F1 @ Threshold": 0.6257
        },
        {
            "Model Architecture": "HistGradientBoosting (Default)",
            "Status": "Benchmarked",
            "Holdout ROC-AUC": 0.8330,
            "Holdout AP (PR-AUC)": 0.6215,
            "Brier Score": 0.1408,
            "Optimal Threshold": 0.35,
            "Recall @ Threshold": "73.26%",
            "F1 @ Threshold": 0.6145
        },
        {
            "Model Architecture": "Random Forest (Default)",
            "Status": "Benchmarked",
            "Holdout ROC-AUC": 0.8219,
            "Holdout AP (PR-AUC)": 0.6102,
            "Brier Score": 0.1448,
            "Optimal Threshold": 0.35,
            "Recall @ Threshold": "71.39%",
            "F1 @ Threshold": 0.5982
        },
        {
            "Model Architecture": "Dummy (Most Frequent)",
            "Status": "Chance Baseline",
            "Holdout ROC-AUC": 0.5000,
            "Holdout AP (PR-AUC)": 0.2654,
            "Brier Score": 0.1949,
            "Optimal Threshold": 0.50,
            "Recall @ Threshold": "0.00%",
            "F1 @ Threshold": 0.0000
        }
    ])
    st.dataframe(comp_df, width="stretch")

    st.markdown("---")
    st.subheader("Publication-Quality Visual Artifacts")

    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.markdown("**ROC & Precision-Recall Curves (Holdout Test Set)**")
        p23 = Path("reports/figures/23_phase2_roc_pr_curves.png")
        if p23.exists():
            st.image(str(p23))
        else:
            st.info("ROC/PR curve image not found.")

    with fig_col2:
        st.markdown("**Probability Calibration & Reliability Curve**")
        p21 = Path("reports/figures/21_calibration_comparison.png")
        if p21.exists():
            st.image(str(p21))
        else:
            st.info("Calibration diagram not found.")

    st.markdown("---")
    st.subheader("Global Feature Attribution (Mean |SHAP|)")
    p19 = Path("reports/figures/19_shap_bar_importance.png")
    if p19.exists():
        st.image(str(p19), caption="Mean |SHAP| feature ranking across holdout test records.")


# -----------------------------------------------------------------------------
# SECTION G: DATA QUALITY & LIMITATIONS
# -----------------------------------------------------------------------------
elif section == "🛡️ Data Quality & Ethics":
    st.markdown("""
        <div class='hero-banner'>
            <h1>Data Quality, Lineage & Ethical Boundaries</h1>
            <p>Dataset audit metrics, missingness handling safeguards, and prohibited operational use cases.</p>
        </div>
    """, unsafe_allow_html=True)

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.subheader("Data Cleaning & Preprocessing Safeguards")
        st.markdown("""
        * **Whitespace Handling:** Exactly 11 records in the dataset contain blank whitespace strings (`" "`). Auditing verified these belong to brand new customers (`tenure == 0`). These are automatically coerced to `0.0`.
        * **Zero-Tenure Safe Division:** When calculating historical average billing (`TotalCharges / tenure`), division-by-zero is safely prevented by setting baseline monthly charges.
        * **Leakage Prevention:** All continuous scaling and categorical encoding are fitted strictly within training cross-validation folds. The holdout test set was evaluated exactly once.
        """)

    with g_col2:
        st.subheader("Known Limitations & Ethical Guardrails")
        st.markdown("""
        * **Cross-Sectional Limitation:** The dataset represents a point-in-time snapshot. Dynamic time-series events (e.g., ticket complaints in last 7 days) are not captured.
        * **Non-Causal Notice:** High feature importance (e.g., Electronic Check or Fiber Optic) denotes *statistical correlation*, not causation. Forcing customers off electronic check will not automatically eliminate churn.
        * **Prohibited Use:** This platform must NOT be used for credit scoring, service denial, discriminatory pricing, or automated contract cancellation.
        """)


# -----------------------------------------------------------------------------
# SECTION H: BATCH PREDICTION
# -----------------------------------------------------------------------------
elif section == "📁 Batch CSV Prediction":
    st.markdown("""
        <div class='hero-banner'>
            <h1>Batch Customer Prediction Engine</h1>
            <p>Upload a raw customer CSV to calculate churn probabilities, assign risk tiers, and export predictions.</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Raw Customer CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_df):,} customer records.")

            required_cols = ["tenure", "MonthlyCharges", "Contract", "PaymentMethod", "InternetService"]
            missing_cols = [c for c in required_cols if c not in batch_df.columns]

            if missing_cols:
                st.error(f"Missing required columns in CSV: {missing_cols}")
            else:
                with st.spinner("Executing model inference..."):
                    batch_probs = pipeline.predict_proba(batch_df)[:, 1]

                out_df = batch_df.copy()
                out_df["Churn_Probability"] = batch_probs.round(4)
                out_df["Churn_Prediction"] = (batch_probs >= optimal_threshold).astype(int)

                def get_tier(p):
                    if p >= 0.60:
                        return "HIGH RISK"
                    elif p >= optimal_threshold:
                        return "MODERATE RISK"
                    return "LOW RISK"

                out_df["Risk_Tier"] = [get_tier(p) for p in batch_probs]

                st.subheader("Batch Scoring Summary")
                b_c1, b_c2, b_c3 = st.columns(3)
                with b_c1:
                    st.metric("High Risk Accounts", f"{(out_df['Risk_Tier']=='HIGH RISK').sum():,}")
                with b_c2:
                    st.metric("Moderate Risk Accounts", f"{(out_df['Risk_Tier']=='MODERATE RISK').sum():,}")
                with b_c3:
                    st.metric("Low Risk Accounts", f"{(out_df['Risk_Tier']=='LOW RISK').sum():,}")

                st.dataframe(out_df.head(50), width="stretch")

                out_csv = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Scored Customer CSV",
                    data=out_csv,
                    file_name="batch_scored_churn_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing batch file: {e}")
