"""
Customer Churn Intelligence Platform
Enterprise Streamlit Dashboard Application (v2.0.0 Production Release)

Structured into 8 comprehensive analytics & operational sections:
A. Executive Overview & Business Metrics
B. Exploratory Analytics & Cohort Comparisons
C. Customer Risk Scoring & Real-Time What-If Simulator
D. Customer Cohort Explorer & Data Export
E. Retention Strategy & Financial ROI Simulator
F. Model Performance & Calibration Comparison
G. Data Quality, Preprocessing Audit & Model Governance
H. Enterprise Batch Prediction & Scored CSV Export
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
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling System
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #2b5c8f;
    }
    .badge-champion {
        background-color: #2b5c8f;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .badge-baseline {
        background-color: #6c757d;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .risk-high {
        color: #d95f02;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .risk-mod {
        color: #fca311;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .risk-low {
        color: #1b9e77;
        font-weight: bold;
        font-size: 1.1rem;
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


# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to Module:",
    [
        "A. Executive Overview",
        "B. Exploratory Analytics",
        "C. Customer Risk Scoring",
        "D. Cohort Explorer",
        "E. Retention ROI Simulator",
        "F. Model Performance & Benchmark",
        "G. Data Quality & Limitations",
        "H. Batch Customer Prediction",
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Active Model:** `{active_model_name}`")
st.sidebar.markdown(f"**Version:** `{active_version}`")
st.sidebar.markdown(f"**Default Threshold ($t$):** `{optimal_threshold:.2f}`")
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Model outputs represent multivariable statistical probabilities, not guaranteed individual behavior.")


# ==============================================================================
# SECTION A: EXECUTIVE OVERVIEW
# ==============================================================================
if section == "A. Executive Overview":
    st.title("💼 Customer Churn Intelligence Platform")
    st.markdown("### Executive Overview & Macro Business Indicators")
    st.markdown(
        "A machine learning platform for early detection, lifetime value preservation, and proactive retention."
    )

    # Top KPI Ribbon
    total_cust = len(raw_df)
    churn_count = (raw_df["Churn"] == "Yes").sum()
    retained_count = total_cust - churn_count
    churn_rate = (churn_count / total_cust) * 100

    holdout_roc = metrics["holdout_test_evaluation"]["test_roc_auc"]
    holdout_ap = metrics["holdout_test_evaluation"]["test_pr_auc"]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Accounts", f"{total_cust:,}", "Active Snapshot")
    with col2:
        st.metric("Base Churn Rate", f"{churn_rate:.1f}%", f"{churn_count:,} lost accounts")
    with col3:
        st.metric("Retained Accounts", f"{retained_count:,}", f"{100-churn_rate:.1f}% active")
    with col4:
        st.metric("Model ROC-AUC", f"{holdout_roc:.4f}", "Test Generalization")
    with col5:
        st.metric("Average Precision (AP)", f"{holdout_ap:.4f}", "Test PR-AUC")

    st.markdown("---")

    # Executive Summary Columns
    e_col1, e_col2 = st.columns([1.2, 1])

    with e_col1:
        st.subheader("Key Findings from Customer Data")
        st.markdown("""
        1. **Contract Type Dominance:** Customers on **Month-to-month contracts** exhibit a **42.71% churn rate**, while **Two-year contract** customers churn at only **2.83%** (a 15x difference).
        2. **First-Year Vulnerability Window:** **47.44% of all churn events** occur within the customer's first 12 months. Churn probability peaks in month 1 and decays exponentially over time.
        3. **Payment Friction:** Customers paying via **Electronic Check** experience a **45.29% churn rate**, compared to **15.24%** for credit card autopay.
        4. **Retention Anchors:** Subscribing to **TechSupport** or **OnlineSecurity** bundles reduces churn propensity by over **60%**, independent of contract type.
        """)

    with e_col2:
        st.subheader("Account Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(
            [retained_count, churn_count],
            labels=["Retained (No)", "Churned (Yes)"],
            autopct="%1.1f%%",
            startangle=140,
            colors=["#2b5c8f", "#d95f02"],
            explode=(0, 0.08),
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        ax.set_title("Customer Base Churn Split", fontsize=11, fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)


# ==============================================================================
# SECTION B: EXPLORATORY ANALYTICS
# ==============================================================================
elif section == "B. Exploratory Analytics":
    st.title("📊 Exploratory Churn Analytics")
    st.markdown("Explore structural correlations and demographic churn rates across customer cohorts.")

    b_tab1, b_tab2, b_tab3 = st.tabs(["Contracts & Tenure", "Services & Add-ons", "Financial & Billing"])

    with b_tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Churn Rate by Contract Type")
            contract_churn = raw_df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            fig, ax = plt.subplots(figsize=(6, 3.8))
            bars = ax.bar(contract_churn["Contract"], contract_churn["Churn"], color=["#d95f02", "#2b5c8f", "#1b9e77"], edgecolor="black")
            ax.set_ylabel("Churn Rate (%)")
            ax.set_ylim(0, 60)
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
            st.subheader("Churn Rate by Internet Service")
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
            st.subheader("Impact of TechSupport & OnlineSecurity")
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
            st.subheader("Churn Rate by Payment Method")
            pay_churn = raw_df.groupby("PaymentMethod")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh(pay_churn["PaymentMethod"], pay_churn["Churn"], color="#2b5c8f", edgecolor="black")
            ax.set_xlabel("Churn Rate (%)")
            for bar in bars:
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.1f}%", va="center", fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Monthly Charges Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=raw_df, x="Churn", y="MonthlyCharges", hue="Churn", palette={"No": "#2b5c8f", "Yes": "#d95f02"}, legend=False, ax=ax)
            ax.set_ylabel("Monthly Charges ($)")
            st.pyplot(fig)
            plt.close(fig)


# ==============================================================================
# SECTION C: CUSTOMER RISK SCORING & SIMULATOR
# ==============================================================================
elif section == "C. Customer Risk Scoring":
    st.title("🎯 Single Customer Risk Scoring & What-If Simulator")
    st.markdown("Compute live churn probabilities and simulate counterfactual retention interventions.")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Customer Profile Parameters")
        with st.form("single_predict_form"):
            tenure = st.slider("Account Tenure (Months)", 0, 72, 6)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 85.0, 5.0)
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

            submitted = st.form_submit_button("Compute Churn Risk")

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
        st.subheader("Risk Assessment & Prescriptions")
        if pred_prob >= 0.60:
            risk_tier = "HIGH RISK"
            badge_class = "risk-high"
            action = "Urgent: Assign Customer Success Specialist for 1-on-1 account review, bundle upgrade, or price discount."
        elif pred_prob >= optimal_threshold:
            risk_tier = "MODERATE RISK"
            badge_class = "risk-mod"
            action = "Proactive: Trigger automated retention email with discount on long-term contract or free TechSupport bundle."
        else:
            risk_tier = "LOW RISK"
            badge_class = "risk-low"
            action = "Standard: Maintain routine product updates, value-add communications, and loyalty rewards."

        st.metric("Predicted Churn Probability", f"{pred_prob*100:.1f}%")
        st.markdown(f"**Risk Stratification Tier:** <span class='{badge_class}'>{risk_tier}</span>", unsafe_allow_html=True)
        st.markdown(f"**Decision Threshold:** `{optimal_threshold:.2f}`")
        st.info(f"**Recommended Action:** {action}")

        st.markdown("---")
        st.subheader("💡 Counterfactual Intervention Simulator")
        st.caption("What happens to churn probability if we migrate this customer's account terms?")

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
        st.subheader("🔬 Local SHAP Factor Attribution")
        st.caption("Game-theoretic decomposition of exact drivers pushing this customer toward or away from churn.")
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
            plt.title(f"Individual Factor Attribution (Log-Odds Impact)", fontsize=11, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close(fig_shap)
        except Exception as shap_err:
            st.info(f"Local SHAP attribution preview: {shap_err}")


# ==============================================================================
# SECTION D: COHORT EXPLORER
# ==============================================================================
elif section == "D. Cohort Explorer":
    st.title("🔍 Customer Cohort Explorer")
    st.markdown("Filter, inspect, and export customer cohorts by contract, tenure, and services.")

    filter_contract = st.multiselect("Contract Type", raw_df["Contract"].unique(), default=list(raw_df["Contract"].unique()))
    filter_internet = st.multiselect("Internet Service", raw_df["InternetService"].unique(), default=list(raw_df["InternetService"].unique()))
    filter_payment = st.multiselect("Payment Method", raw_df["PaymentMethod"].unique(), default=list(raw_df["PaymentMethod"].unique()))

    filtered_df = raw_df[
        (raw_df["Contract"].isin(filter_contract)) &
        (raw_df["InternetService"].isin(filter_internet)) &
        (raw_df["PaymentMethod"].isin(filter_payment))
    ]

    st.markdown(f"**Matching Cohort Accounts:** `{len(filtered_df):,}` of `{len(raw_df):,}`")

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
        st.download_button("Download Filtered Cohort CSV", data=csv, file_name="churn_cohort_export.csv", mime="text/csv")


# ==============================================================================
# SECTION E: RETENTION ROI SIMULATOR
# ==============================================================================
elif section == "E. Retention ROI Simulator":
    st.title("💰 Retention Campaign ROI Simulator")
    st.markdown("Model financial return-on-investment across decision thresholds and outreach economics.")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Economic & Operational Assumptions")
        cust_ltv = st.number_input("Average Customer Lifetime Value (LTV, $)", 500, 5000, 1800, 100)
        retention_cost = st.number_input("Cost of Outreach per Account ($)", 5, 200, 25, 5)
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
        st.subheader("Financial Simulation Results (Test Cohort N=1,409)")
        roi_c1, roi_c2 = st.columns(2)
        with roi_c1:
            st.metric("Targeted Accounts", f"{targeted_customers:,}", f"TP: {tp} | FP: {fp}")
            st.metric("Gross Value Preserved", f"${gross_value_saved:,.2f}", f"Rescues: {rescued_customers:.1f} churners")
        with roi_c2:
            st.metric("Total Campaign Spend", f"${total_campaign_cost:,.2f}", f"@ ${retention_cost}/customer")
            st.metric("Net Retention Profit (ROI)", f"${net_roi:,.2f}", delta="Positive Return" if net_roi > 0 else "Net Loss")

        st.markdown("---")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred Retained", "Pred Churned"], yticklabels=["Actual Retained", "Actual Churned"])
        ax.set_ylabel("True Status")
        ax.set_xlabel("Predicted Status")
        ax.set_title(f"Confusion Matrix @ t = {threshold_sim:.2f}")
        st.pyplot(fig)
        plt.close(fig)


# ==============================================================================
# SECTION F: MODEL PERFORMANCE & BENCHMARK
# ==============================================================================
elif section == "F. Model Performance & Benchmark":
    st.title("📈 Model Performance & Validation Benchmark")
    st.markdown("Genuine evaluation metrics evaluated on the untouched holdout test partition ($N=1,409$).")

    # Model comparison table
    st.subheader("Model Benchmark Comparison")
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
            "Status": "Unskilled Chance",
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
    st.subheader("Visual Evaluation Artifacts")

    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.markdown("**Receiver Operating Characteristic & Precision-Recall Curves**")
        p23 = Path("reports/figures/23_phase2_roc_pr_curves.png")
        if p23.exists():
            st.image(str(p23))
        else:
            st.info("ROC/PR curve image not found.")

    with fig_col2:
        st.markdown("**Probability Reliability & Calibration**")
        p21 = Path("reports/figures/21_calibration_comparison.png")
        if p21.exists():
            st.image(str(p21))
        else:
            st.info("Calibration image not found.")

    st.markdown("---")
    st.subheader("Global Feature Attribution (SHAP TreeExplainer)")
    p19 = Path("reports/figures/19_shap_bar_importance.png")
    if p19.exists():
        st.image(str(p19), caption="Mean |SHAP| attribution across holdout test records.")


# ==============================================================================
# SECTION G: DATA QUALITY & LIMITATIONS
# ==============================================================================
elif section == "G. Data Quality & Limitations":
    st.title("🛡️ Data Quality, Governance & Model Limitations")
    st.markdown("System transparency, data anomaly resolutions, and ethical operational boundaries.")

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.subheader("Data Cleaning & Preprocessing Safeguards")
        st.markdown("""
        * **TotalCharges Missingness:** Exactly 11 records in the dataset contain blank whitespace strings (`" "`). Auditing verified these belong to brand new customers (`tenure == 0`). These are automatically coerced to `0.0`.
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


# ==============================================================================
# SECTION H: BATCH PREDICTION
# ==============================================================================
elif section == "H. Batch Customer Prediction":
    st.title("📁 Batch Customer Prediction & Export")
    st.markdown("Upload a raw CSV of customer accounts to compute churn probabilities and recommended actions.")

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
                    "Download Fully Scored Customer CSV",
                    data=out_csv,
                    file_name="batch_scored_churn_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing batch file: {e}")
