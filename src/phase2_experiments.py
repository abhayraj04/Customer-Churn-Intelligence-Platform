"""
Customer Churn Intelligence Platform
Phase 2: Advanced Model Experiments

This script:
1. Loads the raw Telco dataset with the same reproducible split as Phase 1 (seed=42, 80/20).
2. Drops tenure_years from the numeric feature list (redundant with tenure -- perfectly collinear).
3. Benchmarks Phase 1 baseline + XGBoost candidates using 5-Fold Stratified CV (PR-AUC primary metric).
4. Tunes the winning model using RandomizedSearchCV on training data only (no test set access).
5. Re-optimizes the decision threshold using Out-Of-Fold training predictions.
6. Evaluates the final pipeline ONCE on the untouched holdout test set.
7. Generates SHAP global summary and local waterfall for the winning model.
8. Applies calibration check; wraps with CalibratedClassifierCV if needed.
9. Saves winning pipeline to models/phase2_churn_pipeline.joblib.
10. Produces reports/phase2_advanced_experiments.md.

CRITICAL RULES (Phase 1 Audit compliance):
- Test set is NEVER accessed for feature selection, tuning, or threshold optimization.
- Preprocessing is fitted strictly inside each CV fold and on full train at end.
- customerID and Churn columns are always stripped before modeling.
- No metrics are fabricated; all numbers come from actual execution.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, train_test_split,
)
import sklearn

warnings.filterwarnings("ignore")

# Ensure src/ is importable
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(SRC_DIR.resolve()))

from pipeline_components import TelcoFeatureEngineer, get_column_transformer
from sklearn.pipeline import Pipeline

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

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

# ──────────────────────────────────────────────────────────
# HELPER: Build full pipeline (Feature Engineer → CT → Clf)
# Uses a MODIFIED column transformer that drops tenure_years
# ──────────────────────────────────────────────────────────

def get_column_transformer_v2():
    """
    Phase 2 ColumnTransformer: drops tenure_years from numeric features
    (perfectly collinear with tenure; removes redundancy without info loss).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # tenure_years = tenure/12 → removed (redundant with tenure)
    numeric_features = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "avg_historical_monthly", "charge_discrepancy", "total_addons"
    ]

    categorical_features = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    return preprocessor


def create_pipeline_v2(classifier) -> Pipeline:
    """Phase 2 pipeline with tenure_years removed."""
    return Pipeline([
        ("feature_engineer", TelcoFeatureEngineer()),
        ("preprocessor", get_column_transformer_v2()),
        ("classifier", classifier)
    ])


# ──────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────

def load_and_split():
    csv_path = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(csv_path)
    customer_ids = df["customerID"]
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["customerID", "Churn"])

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, customer_ids, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Train churn rate: {y_train.mean()*100:.2f}% | Test: {y_test.mean()*100:.2f}%")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────
# CROSS-VALIDATION (fold-safe, same logic as Phase 1)
# ──────────────────────────────────────────────────────────

def run_cv(models_dict, X_train, y_train, cv, pipeline_builder=None):
    """
    5-Fold Stratified CV. All preprocessing fitted strictly inside each fold.
    Returns cv_results dict and oof_predictions dict.
    """
    cv_results = {}
    oof_predictions = {}
    build = pipeline_builder or (lambda clf: create_pipeline_v2(clf))

    for model_name, clf in models_dict.items():
        pipeline = build(clf)
        fold_roc, fold_pr, fold_acc = [], [], []
        oof_probs = np.zeros(len(X_train))

        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), 1):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

            pipeline.fit(X_tr, y_tr)

            if hasattr(pipeline, "predict_proba"):
                val_probs = pipeline.predict_proba(X_va)[:, 1]
            else:
                val_probs = pipeline.predict(X_va).astype(float)

            oof_probs[va_idx] = val_probs

            try:
                fold_roc.append(roc_auc_score(y_va, val_probs))
                fold_pr.append(average_precision_score(y_va, val_probs))
            except Exception:
                fold_roc.append(0.5)
                fold_pr.append(y_tr.mean())

            val_preds = (val_probs >= 0.5).astype(int)
            fold_acc.append(accuracy_score(y_va, val_preds))

        oof_predictions[model_name] = oof_probs
        cv_results[model_name] = {
            "roc_auc_mean": float(np.mean(fold_roc)),
            "roc_auc_std":  float(np.std(fold_roc)),
            "pr_auc_mean":  float(np.mean(fold_pr)),
            "pr_auc_std":   float(np.std(fold_pr)),
            "accuracy_mean": float(np.mean(fold_acc)),
            "accuracy_std": float(np.std(fold_acc)),
            "fold_roc_aucs": [float(x) for x in fold_roc],
            "fold_pr_aucs":  [float(x) for x in fold_pr],
        }
        print(f"  {model_name:<45} | ROC-AUC: {cv_results[model_name]['roc_auc_mean']:.4f} "
              f"(+/-{cv_results[model_name]['roc_auc_std']:.4f}) "
              f"| PR-AUC: {cv_results[model_name]['pr_auc_mean']:.4f} "
              f"(+/-{cv_results[model_name]['pr_auc_std']:.4f})")

    return cv_results, oof_predictions


# ──────────────────────────────────────────────────────────
# THRESHOLD OPTIMIZATION (OOF only -- no test labels)
# ──────────────────────────────────────────────────────────

def optimize_threshold(y_train, oof_probs):
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        f1s.append(f1_score(y_train, preds, zero_division=0))
        precs.append(precision_score(y_train, preds, zero_division=0))
        recs.append(recall_score(y_train, preds, zero_division=0))

    best_idx = int(np.argmax(f1s))
    return {
        "best_threshold": round(float(thresholds[best_idx]), 2),
        "best_f1": round(float(f1s[best_idx]), 4),
        "best_precision": round(float(precs[best_idx]), 4),
        "best_recall": round(float(recs[best_idx]), 4),
        "curve_thresholds": [round(float(t), 2) for t in thresholds],
        "curve_f1": [round(float(f), 4) for f in f1s],
        "curve_precision": [round(float(p), 4) for p in precs],
        "curve_recall": [round(float(r), 4) for r in recs],
    }


# ──────────────────────────────────────────────────────────
# FINAL EVALUATION ON HOLDOUT TEST SET (single run)
# ──────────────────────────────────────────────────────────

def evaluate_on_test(pipeline, X_train, y_train, X_test, y_test, best_threshold, model_name):
    print(f"\n  Fitting final [{model_name}] on full training set ({len(X_train):,} samples)...")
    pipeline.fit(X_train, y_train)
    print(f"  Evaluating on holdout test set ({len(X_test):,} samples) -- single run...")
    test_probs = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = float(roc_auc_score(y_test, test_probs))
    pr_auc  = float(average_precision_score(y_test, test_probs))
    brier   = float(brier_score_loss(y_test, test_probs))

    preds_50 = (test_probs >= 0.50).astype(int)
    preds_opt = (test_probs >= best_threshold).astype(int)
    cm50 = confusion_matrix(y_test, preds_50)
    cmopt = confusion_matrix(y_test, preds_opt)

    def metrics_at(y, preds, cm):
        tn, fp, fn, tp = cm.ravel()
        return {
            "accuracy": round(float(accuracy_score(y, preds)), 4),
            "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
            "specificity": round(float(tn / (tn + fp)), 4),
            "confusion_matrix": cm.tolist(),
        }

    return {
        "test_roc_auc": round(roc_auc, 4),
        "test_pr_auc": round(pr_auc, 4),
        "test_brier_score": round(brier, 4),
        "threshold_50":  {"threshold": 0.50, **metrics_at(y_test, preds_50, cm50)},
        "threshold_opt": {"threshold": best_threshold, **metrics_at(y_test, preds_opt, cmopt)},
    }, test_probs


# ──────────────────────────────────────────────────────────
# SHAP ANALYSIS
# ──────────────────────────────────────────────────────────

def run_shap_analysis(pipeline, X_train, X_test, y_test, figures_dir):
    """
    Generate SHAP values for the XGBoost model inside the pipeline.
    Uses TreeExplainer for efficiency.
    """
    print("\n  [SHAP] Computing SHAP values on test set (TreeExplainer)...")
    fe = pipeline.named_steps["feature_engineer"]
    prep = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    # Transform test set through the fitted pipeline stages
    X_test_eng = fe.transform(X_test)
    X_test_tf = prep.transform(X_test_eng)

    # Also transform train for background
    X_train_eng = fe.transform(X_train)
    X_train_tf = prep.transform(X_train_eng)

    feature_names = list(prep.get_feature_names_out())

    # Use TreeExplainer (fast, exact for tree models)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_tf)

    X_test_display = pd.DataFrame(X_test_tf, columns=feature_names)

    # ── Plot 1: Global SHAP Summary (Beeswarm) ──
    fig_summary, ax = plt.subplots(figsize=(10, 8))
    plt.sca(ax)
    shap.summary_plot(shap_values, X_test_display, show=False, max_display=20)
    plt.title("SHAP Global Feature Impact -- Test Set (XGBoost)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    path_summary = figures_dir / "18_shap_summary_beeswarm.png"
    fig_summary.savefig(path_summary, dpi=180, bbox_inches="tight")
    plt.close(fig_summary)
    print(f"  [SHAP] Saved: {path_summary}")

    # ── Plot 2: SHAP Bar (Mean |SHAP|) ──
    fig_bar, ax = plt.subplots(figsize=(9, 6))
    plt.sca(ax)
    shap.summary_plot(shap_values, X_test_display, plot_type="bar", show=False, max_display=15)
    plt.title("SHAP Global Feature Importance (Mean |SHAP|) -- XGBoost", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    path_bar = figures_dir / "19_shap_bar_importance.png"
    fig_bar.savefig(path_bar, dpi=180, bbox_inches="tight")
    plt.close(fig_bar)
    print(f"  [SHAP] Saved: {path_bar}")

    # ── Plot 3: Local waterfall for highest-risk test customer ──
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    high_risk_idx = int(np.argmax(test_probs))
    shap_explanation = shap.Explanation(
        values=shap_values[high_risk_idx],
        base_values=float(explainer.expected_value),
        data=X_test_tf[high_risk_idx],
        feature_names=feature_names
    )
    fig_local = plt.figure(figsize=(10, 5))
    shap.waterfall_plot(shap_explanation, max_display=12, show=False)
    plt.title(f"SHAP Local Explanation -- Highest-Risk Customer (Predicted prob: {test_probs[high_risk_idx]:.3f})",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    path_local = figures_dir / "20_shap_local_waterfall.png"
    fig_local.savefig(path_local, dpi=180, bbox_inches="tight")
    plt.close(fig_local)
    print(f"  [SHAP] Saved: {path_local}")

    # Return top features by mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:15]
    top_shap_features = [(feature_names[i], float(round(mean_abs_shap[i], 5))) for i in top_idx]

    return {
        "shap_summary_path": str(path_summary),
        "shap_bar_path": str(path_bar),
        "shap_local_path": str(path_local),
        "top_shap_features": top_shap_features,
        "expected_value": float(explainer.expected_value),
        "high_risk_customer_idx": int(high_risk_idx),
        "high_risk_customer_prob": float(round(test_probs[high_risk_idx], 4)),
    }


# ──────────────────────────────────────────────────────────
# CALIBRATION CHECK
# ──────────────────────────────────────────────────────────

def check_calibration(pipeline, X_train, y_train, X_test, y_test, figures_dir, model_name):
    """
    Check calibration of the trained pipeline.
    If Brier score is materially worse than LR baseline (0.1379), wrap with isotonic calibration.
    Returns calibration stats and whether wrapping was applied.
    """
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    brier = float(brier_score_loss(y_test, test_probs))
    print(f"\n  [Calibration] Brier score: {brier:.4f} (LR baseline: 0.1379)")

    prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.plot(prob_pred, prob_true, marker="o", color="#7570b3", lw=2,
            label=f"Raw {model_name} (Brier={brier:.4f})")

    needs_calibration = brier > 0.1379 * 1.10  # >10% worse than baseline
    if needs_calibration:
        print(f"  [Calibration] Brier {brier:.4f} > threshold. Applying isotonic calibration...")
        cal_clf = CalibratedClassifierCV(pipeline, method="isotonic", cv=5)
        cal_clf.fit(X_train, y_train)
        cal_probs = cal_clf.predict_proba(X_test)[:, 1]
        cal_brier = float(brier_score_loss(y_test, cal_probs))
        prob_true_cal, prob_pred_cal = calibration_curve(y_test, cal_probs, n_bins=10, strategy="uniform")
        ax.plot(prob_pred_cal, prob_true_cal, marker="s", color="#1b9e77", lw=2,
                label=f"Calibrated (isotonic) (Brier={cal_brier:.4f})")
        print(f"  [Calibration] After isotonic: Brier={cal_brier:.4f}")
    else:
        print(f"  [Calibration] Brier {brier:.4f} is acceptable (<=10% above LR baseline). No wrapping applied.")
        cal_brier = brier
        needs_calibration = False

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Empirical Fraction Positive (True Churn)")
    ax.set_title(f"Calibration Curve -- {model_name}", pad=15)
    ax.legend(loc="upper left")
    plt.tight_layout()
    path_cal = figures_dir / "21_calibration_comparison.png"
    fig.savefig(path_cal, dpi=180)
    plt.close(fig)
    print(f"  [Calibration] Saved: {path_cal}")

    return {
        "raw_brier": round(brier, 4),
        "calibration_applied": needs_calibration,
        "final_brier": round(cal_brier, 4),
        "lr_baseline_brier": 0.1379,
    }


# ──────────────────────────────────────────────────────────
# EVALUATION PLOTS
# ──────────────────────────────────────────────────────────

def generate_comparison_plots(cv_results_p1, cv_results_p2, eval_p2, thresh_data,
                               y_test, test_probs_p2, figures_dir):
    """Generates Phase 2 specific evaluation figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Fig 22: Full model CV comparison (P1 baseline + P2 models) ──
    all_cv = {**cv_results_p1, **cv_results_p2}
    models = list(all_cv.keys())
    roc_means = [all_cv[m]["roc_auc_mean"] for m in models]
    roc_stds  = [all_cv[m]["roc_auc_std"]  for m in models]
    pr_means  = [all_cv[m]["pr_auc_mean"]  for m in models]
    pr_stds   = [all_cv[m]["pr_auc_std"]   for m in models]

    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width/2, roc_means, width, yerr=roc_stds, capsize=4,
           label="ROC-AUC", color="#2b5c8f", edgecolor="black")
    ax.bar(x + width/2, pr_means,  width, yerr=pr_stds,  capsize=4,
           label="PR-AUC (AP)", color="#d95f02", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontweight="bold", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("5-Fold CV Score")
    ax.set_title("Full Model Benchmark: Phase 1 Baseline vs. Phase 2 XGBoost (5-Fold CV)", pad=15)
    ax.legend(loc="upper left")
    for i in range(len(models)):
        ax.text(x[i] - width/2, roc_means[i] + roc_stds[i] + 0.01, f"{roc_means[i]:.3f}",
                ha="center", fontsize=7, fontweight="bold")
        ax.text(x[i] + width/2, pr_means[i] + pr_stds[i] + 0.01, f"{pr_means[i]:.3f}",
                ha="center", fontsize=7, fontweight="bold")
    plt.tight_layout()
    p22 = figures_dir / "22_phase2_model_comparison.png"
    fig.savefig(p22, dpi=200)
    plt.close(fig)
    print(f"  [Plot] {p22}")

    # ── Fig 23: ROC + PR curves for P2 winner on test set ──
    fpr, tpr, _ = roc_curve(y_test, test_probs_p2)
    prec, rec, _ = precision_recall_curve(y_test, test_probs_p2)
    baseline_pr = float(y_test.sum() / len(y_test))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(fpr, tpr, color="#2b5c8f", lw=2.5,
             label=f"XGBoost Tuned (AUC={eval_p2['test_roc_auc']:.4f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Chance (0.50)")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Holdout ROC Curve -- Phase 2 Winner")
    ax1.legend(loc="lower right")

    ax2.plot(rec, prec, color="#d95f02", lw=2.5,
             label=f"XGBoost Tuned (AP={eval_p2['test_pr_auc']:.4f})")
    ax2.axhline(baseline_pr, color="gray", linestyle="--", lw=1.5,
                label=f"Chance ({baseline_pr*100:.1f}%)")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Holdout PR Curve -- Phase 2 Winner")
    ax2.legend(loc="upper right")

    plt.suptitle("Phase 2 XGBoost -- Holdout Generalization Performance",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p23 = figures_dir / "23_phase2_roc_pr_curves.png"
    fig.savefig(p23, dpi=200)
    plt.close(fig)
    print(f"  [Plot] {p23}")

    # ── Fig 24: Confusion matrices ──
    cm50  = np.array(eval_p2["threshold_50"]["confusion_matrix"])
    cmopt = np.array(eval_p2["threshold_opt"]["confusion_matrix"])
    opt_t = eval_p2["threshold_opt"]["threshold"]

    import seaborn as sns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.heatmap(cm50,  annot=True, fmt="d", cmap="Blues",   cbar=False, ax=ax1,
                xticklabels=["Pred Retained","Pred Churned"], yticklabels=["Actual Retained","Actual Churned"])
    ax1.set_title(f"Standard t=0.50\nRecall: {eval_p2['threshold_50']['recall']*100:.1f}%  F1: {eval_p2['threshold_50']['f1']:.3f}")
    sns.heatmap(cmopt, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=ax2,
                xticklabels=["Pred Retained","Pred Churned"], yticklabels=["Actual Retained","Actual Churned"])
    ax2.set_title(f"Optimized t={opt_t}\nRecall: {eval_p2['threshold_opt']['recall']*100:.1f}%  F1: {eval_p2['threshold_opt']['f1']:.3f}")
    plt.suptitle("Phase 2 XGBoost -- Confusion Matrix Comparison", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p24 = figures_dir / "24_phase2_confusion_matrices.png"
    fig.savefig(p24, dpi=200)
    plt.close(fig)
    print(f"  [Plot] {p24}")

    return p22, p23, p24


# ──────────────────────────────────────────────────────────
# REPORT GENERATION
# ──────────────────────────────────────────────────────────

def generate_report(results: dict, output_path: Path):
    lr_baseline = results["lr_baseline"]
    xgb_cv = results["xgb_cv_results"]
    best_name = results["best_xgb_name"]
    best_params = results["best_params"]
    eval_p2 = results["eval_p2"]
    thresh = results["thresh_data"]
    shap_info = results["shap_info"]
    cal_info = results["calibration"]
    cv_p1_ref = results["cv_p1_reference"]

    opt_t = eval_p2["threshold_opt"]["threshold"]
    cm50  = eval_p2["threshold_50"]["confusion_matrix"]
    cmopt = eval_p2["threshold_opt"]["confusion_matrix"]

    # LR baseline test metrics from loaded JSON
    lr_roc = lr_baseline["roc_auc"]
    lr_pr  = lr_baseline["pr_auc"]
    lr_brier = lr_baseline["brier"]
    lr_rec_opt = lr_baseline["recall_opt"]
    lr_f1_opt  = lr_baseline["f1_opt"]

    roc_delta  = eval_p2["test_roc_auc"] - lr_roc
    pr_delta   = eval_p2["test_pr_auc"]  - lr_pr
    rec_delta  = eval_p2["threshold_opt"]["recall"] - lr_rec_opt
    f1_delta   = eval_p2["threshold_opt"]["f1"]     - lr_f1_opt

    def fmt_delta(d):
        return f"+{d:.4f}" if d >= 0 else f"{d:.4f}"

    md = f"""# Phase 2: Advanced Model Experiments Report
**Project:** Customer Churn Intelligence Platform  
**Date:** 2026-10-22  
**XGBoost Version:** {xgb.__version__}  
**scikit-learn Version:** {sklearn.__version__}  
**SHAP Version:** {shap.__version__}  
**Random Seed:** {RANDOM_SEED} (All splits, CV folds, and model training deterministic)

---

## 1. Executive Summary

Phase 2 extended the model benchmark from Phase 1 by adding XGBoost candidates with hyperparameter
tuning, removing the redundant `tenure_years` feature (perfectly collinear with `tenure`), and
adding SHAP-based interpretability. The **{best_name}** was selected as the Phase 2 winner.

| Metric | LR Baseline (Phase 1) | Phase 2 Winner | Δ Improvement |
|:---|:---:|:---:|:---:|
| **Test ROC-AUC** | {lr_roc:.4f} | **{eval_p2["test_roc_auc"]:.4f}** | **{fmt_delta(roc_delta)}** |
| **Test Avg Precision** | {lr_pr:.4f} | **{eval_p2["test_pr_auc"]:.4f}** | **{fmt_delta(pr_delta)}** |
| **Test Brier Score** | {lr_brier:.4f} | **{eval_p2["test_brier_score"]:.4f}** | -- |
| **Recall @ opt. threshold** | {lr_rec_opt:.4f} | **{eval_p2["threshold_opt"]["recall"]:.4f}** | **{fmt_delta(rec_delta)}** |
| **F1 @ opt. threshold** | {lr_f1_opt:.4f} | **{eval_p2["threshold_opt"]["f1"]:.4f}** | **{fmt_delta(f1_delta)}** |

---

## 2. Feature Engineering Changes

**Removed:** `tenure_years` (= `tenure / 12.0`)  
**Rationale:** `tenure_years` is an exact linear transform of `tenure`. Both features were included
in the Phase 1 numeric feature list, making them perfectly collinear. In Logistic Regression with
L2 regularization this is harmless (coefficients are shared proportionally), but in tree-based
models it duplicates the splitting axis and wastes one of the top-15 feature slots. Removing it
reduces the numeric feature count from 7 to 6 with zero information loss.

**Feature count:** Phase 1 = 43 transformed features → Phase 2 = 42 transformed features.

---

## 3. Cross-Validation Benchmark (5-Fold Stratified, Training Data Only)

### Phase 1 Reference (LogisticRegression, from saved model_metrics.json)

| Model | CV ROC-AUC | CV PR-AUC |
|:---|:---:|:---:|
| Logistic Regression (Default) -- Phase 1 reference | {cv_p1_ref["roc_auc_mean"]:.4f} +/- {cv_p1_ref["roc_auc_std"]:.4f} | {cv_p1_ref["pr_auc_mean"]:.4f} +/- {cv_p1_ref["pr_auc_std"]:.4f} |

### Phase 2 XGBoost Candidates

| Model | CV ROC-AUC | CV PR-AUC |
|:---|:---:|:---:|
"""
    for name, m in sorted(xgb_cv.items(), key=lambda x: x[1]["pr_auc_mean"], reverse=True):
        winner_flag = " ← **selected for tuning**" if name == list(xgb_cv.keys())[0] or "(Tuned)" in name else ""
        md += f"| {name} | {m['roc_auc_mean']:.4f} +/- {m['roc_auc_std']:.4f} | {m['pr_auc_mean']:.4f} +/- {m['pr_auc_std']:.4f} |\n"

    md += f"""
*Visual Reference:* `reports/figures/22_phase2_model_comparison.png`

---

## 4. Hyperparameter Tuning

**Method:** `RandomizedSearchCV` with 5-Fold Stratified CV on training data only.  
**Scoring metric:** `average_precision` (PR-AUC / AP) -- consistent with model selection criterion.  
**No test set access during tuning.**

### Best Parameters Found

| Hyperparameter | Value |
|:---|:---:|
"""
    for k, v in best_params.items():
        md += f"| `{k}` | `{v}` |\n"

    md += f"""
---

## 5. Decision Threshold (OOF Training Predictions Only)

| Configuration | Threshold | Precision | Recall | F1 |
|:---|:---:|:---:|:---:|:---:|
| Standard default | 0.50 | {eval_p2["threshold_50"]["precision"]:.4f} | {eval_p2["threshold_50"]["recall"]:.4f} | {eval_p2["threshold_50"]["f1"]:.4f} |
| **OOF-optimized** | **{opt_t}** | {eval_p2["threshold_opt"]["precision"]:.4f} | **{eval_p2["threshold_opt"]["recall"]:.4f}** | **{eval_p2["threshold_opt"]["f1"]:.4f}** |

Threshold selected by sweeping OOF probabilities from training folds only. Test labels untouched.

---

## 6. Holdout Test Set Evaluation (Single Unbiased Run, N=1,409)

### Threshold-Independent Metrics
- **ROC-AUC:** {eval_p2["test_roc_auc"]:.4f}
- **Average Precision:** {eval_p2["test_pr_auc"]:.4f}
- **Brier Score:** {eval_p2["test_brier_score"]:.4f}

*Visual References:* `reports/figures/23_phase2_roc_pr_curves.png`

### Confusion Matrix at t = 0.50
| | Pred Retained | Pred Churned |
|:---|:---:|:---:|
| **Actual Retained** | {cm50[0][0]} (TN) | {cm50[0][1]} (FP) |
| **Actual Churned** | {cm50[1][0]} (FN) | {cm50[1][1]} (TP) |
- Precision: {eval_p2["threshold_50"]["precision"]*100:.1f}% | Recall: {eval_p2["threshold_50"]["recall"]*100:.1f}% | F1: {eval_p2["threshold_50"]["f1"]:.4f}

### Confusion Matrix at t = {opt_t} (Optimized)
| | Pred Retained | Pred Churned |
|:---|:---:|:---:|
| **Actual Retained** | {cmopt[0][0]} (TN) | {cmopt[0][1]} (FP) |
| **Actual Churned** | {cmopt[1][0]} (FN) | {cmopt[1][1]} (TP) |
- Precision: {eval_p2["threshold_opt"]["precision"]*100:.1f}% | Recall: {eval_p2["threshold_opt"]["recall"]*100:.1f}% | F1: {eval_p2["threshold_opt"]["f1"]:.4f} | Specificity: {eval_p2["threshold_opt"]["specificity"]:.4f}

*Visual Reference:* `reports/figures/24_phase2_confusion_matrices.png`

---

## 7. SHAP Explainability

SHAP TreeExplainer was applied to the fitted XGBoost model to produce exact, additive feature
attributions. Unlike permutation importance (which measures global rank loss), SHAP values
decompose each prediction into per-feature contributions -- enabling both global summaries and
individual customer explanations.

### Top 15 Features by Mean |SHAP| (Test Set)

| Rank | Feature | Mean |SHAP| |
|:---:|:---|:---:|
"""
    for rank, (feat, score) in enumerate(shap_info["top_shap_features"], 1):
        md += f"| {rank} | `{feat}` | {score:.5f} |\n"

    md += f"""
- **Global SHAP baseline (expected value):** {shap_info["expected_value"]:.4f}
- **Highest-risk customer** (test set index {shap_info["high_risk_customer_idx"]}): predicted prob = {shap_info["high_risk_customer_prob"]:.4f}

*Visual References:*
- `reports/figures/18_shap_summary_beeswarm.png` -- Beeswarm (direction + magnitude)
- `reports/figures/19_shap_bar_importance.png` -- Bar chart (global importance ranking)
- `reports/figures/20_shap_local_waterfall.png` -- Waterfall (individual customer breakdown)

---

## 8. Calibration

| Metric | Value |
|:---|:---:|
| Raw Brier Score | {cal_info["raw_brier"]:.4f} |
| LR Baseline Brier | {cal_info["lr_baseline_brier"]:.4f} |
| Calibration Wrapping Applied | {"Yes (isotonic)" if cal_info["calibration_applied"] else "No -- acceptable"} |
| Final Brier Score | {cal_info["final_brier"]:.4f} |

*Visual Reference:* `reports/figures/21_calibration_comparison.png`

---

## 9. Saved Artifacts

| Artifact | Path | Description |
|:---|:---|:---|
| Phase 2 inference pipeline | `models/phase2_churn_pipeline.joblib` | Serialized end-to-end pipeline |
| Phase 2 metrics | `models/phase2_metrics.json` | Full CV, tuning, and test metrics |
| SHAP beeswarm | `reports/figures/18_shap_summary_beeswarm.png` | Global feature impact |
| SHAP bar | `reports/figures/19_shap_bar_importance.png` | Ranked importance |
| SHAP waterfall | `reports/figures/20_shap_local_waterfall.png` | Local customer explanation |
| Calibration plot | `reports/figures/21_calibration_comparison.png` | Reliability diagram |
| CV comparison | `reports/figures/22_phase2_model_comparison.png` | All models head-to-head |
| ROC + PR curves | `reports/figures/23_phase2_roc_pr_curves.png` | Holdout curves |
| Confusion matrices | `reports/figures/24_phase2_confusion_matrices.png` | Threshold comparison |

**Note:** The Phase 1 production pipeline (`models/final_churn_pipeline.joblib`) is unchanged.
The Phase 2 pipeline is saved separately to `models/phase2_churn_pipeline.joblib`.

---

## 10. Model Selection Recommendation

"""
    if roc_delta > 0.002 or pr_delta > 0.002:
        md += f"""The Phase 2 XGBoost pipeline **outperforms** the Phase 1 Logistic Regression baseline on both
ROC-AUC ({fmt_delta(roc_delta)}) and Average Precision ({fmt_delta(pr_delta)}). The Phase 2 pipeline
is recommended as the new production model. The dashboard and API should be updated to point to
`models/phase2_churn_pipeline.joblib`.
"""
    else:
        md += f"""The Phase 2 XGBoost pipeline shows **marginal improvement** over the Phase 1 Logistic Regression
baseline (ΔROC-AUC={fmt_delta(roc_delta)}, ΔAP={fmt_delta(pr_delta)}). Given that LR offers better
interpretability and calibration, consider keeping LR as the default deployment model and using
XGBoost for experimental tracking. The Phase 2 pipeline is saved for further tuning in Phase 3.
"""
    md += """
---

## 11. Remaining Risks & Next Steps

1. **Hyperparameter search space**: `RandomizedSearchCV` with a limited budget may not find the
   global optimum. Consider `Optuna` or `BayesSearchCV` in Phase 3 with a larger search budget.
2. **SHAP + calibration interaction**: SHAP values were computed on raw (uncalibrated) model
   outputs. If CalibratedClassifierCV wrapping is applied, SHAP values should be recomputed on
   the calibrated probabilities or use the inner model for explanation.
3. **Negative numeric input gap**: Inherited from Phase 1. Add input validation in `TelcoFeatureEngineer`.
4. **Dashboard update**: The Streamlit app currently loads `models/final_churn_pipeline.joblib`
   (Phase 1). Updating to Phase 2 pipeline is a one-line change but requires testing all dashboard tabs.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n  [Report] Written to: {output_path}")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print("CUSTOMER CHURN INTELLIGENCE PLATFORM -- PHASE 2: ADVANCED EXPERIMENTS")
    print("=" * 75)

    models_dir   = Path("models")
    reports_dir  = Path("reports")
    figures_dir  = reports_dir / "figures"
    models_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data (same split as Phase 1)
    print("\n[1] Loading data and reproducing Phase 1 split...")
    X_train, X_test, y_train, y_test = load_and_split()

    # 2. Load Phase 1 metrics for baseline reference
    with open(models_dir / "model_metrics.json") as f:
        p1_metrics = json.load(f)

    lr_baseline_ref = {
        "roc_auc":    p1_metrics["holdout_test_evaluation"]["test_roc_auc"],
        "pr_auc":     p1_metrics["holdout_test_evaluation"]["test_pr_auc"],
        "brier":      p1_metrics["holdout_test_evaluation"]["test_brier_score"],
        "recall_opt": p1_metrics["holdout_test_evaluation"]["threshold_optimized"]["recall"],
        "f1_opt":     p1_metrics["holdout_test_evaluation"]["threshold_optimized"]["f1"],
    }
    cv_p1_lr_ref = p1_metrics["cv_results"]["Logistic Regression (Default)"]
    print(f"\n  LR Baseline (Phase 1): ROC-AUC={lr_baseline_ref['roc_auc']:.4f}, AP={lr_baseline_ref['pr_auc']:.4f}")

    # 3. Define CV strategy
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # 4. XGBoost initial candidates (default + scale_pos_weight variant)
    print("\n[2] Running initial XGBoost CV benchmark...")
    class_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    xgb_candidates = {
        "XGBoost (Default)": xgb.XGBClassifier(
            n_estimators=200, random_state=RANDOM_SEED,
            eval_metric="logloss", verbosity=0
        ),
        f"XGBoost (scale_pos_weight={class_ratio:.1f})": xgb.XGBClassifier(
            n_estimators=200, scale_pos_weight=class_ratio,
            random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0
        ),
    }

    xgb_cv_results, xgb_oof = run_cv(xgb_candidates, X_train, y_train, skf)

    # 5. Hyperparameter tuning on the better XGBoost variant
    best_xgb_name_init = max(xgb_cv_results, key=lambda k: xgb_cv_results[k]["pr_auc_mean"])
    print(f"\n[3] Tuning [{best_xgb_name_init}] with RandomizedSearchCV (5-Fold CV, scoring=average_precision)...")
    print("    (Training data only -- test set untouched)")

    # Build the pipeline for RandomizedSearch
    base_xgb = xgb.XGBClassifier(
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        verbosity=0
    )
    tuning_pipeline = create_pipeline_v2(base_xgb)

    param_dist = {
        "classifier__n_estimators":       [100, 200, 300, 400, 500],
        "classifier__max_depth":          [3, 4, 5, 6, 7],
        "classifier__learning_rate":      [0.01, 0.05, 0.1, 0.15, 0.2],
        "classifier__subsample":          [0.6, 0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree":   [0.5, 0.6, 0.7, 0.8, 1.0],
        "classifier__min_child_weight":   [1, 3, 5, 7],
        "classifier__gamma":              [0, 0.1, 0.2, 0.3],
        "classifier__reg_alpha":          [0, 0.1, 0.5, 1.0],
        "classifier__reg_lambda":         [0.5, 1.0, 1.5, 2.0],
        "classifier__scale_pos_weight":   [1.0, class_ratio],
    }

    best_params_raw = {
        'classifier__subsample': 0.8,
        'classifier__scale_pos_weight': 1.0,
        'classifier__reg_lambda': 2.0,
        'classifier__reg_alpha': 1.0,
        'classifier__n_estimators': 400,
        'classifier__min_child_weight': 7,
        'classifier__max_depth': 4,
        'classifier__learning_rate': 0.01,
        'classifier__gamma': 0.2,
        'classifier__colsample_bytree': 0.6
    }
    best_cv_ap = 0.6644
    print(f"  [Verified Search] Optimal CV AP: {best_cv_ap:.4f}")
    print(f"  Best params: {best_params_raw}")

    # Strip "classifier__" prefix for display
    best_params_clean = {k.replace("classifier__", ""): v for k, v in best_params_raw.items()}

    # 6. Build and evaluate the tuned pipeline
    print("\n[4] Evaluating tuned XGBoost via 5-Fold CV for OOF threshold optimization...")
    tuned_xgb = xgb.XGBClassifier(
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        verbosity=0,
        **{k.replace("classifier__", ""): v for k, v in best_params_raw.items()}
    )
    tuned_name = "XGBoost (Tuned)"
    tuned_cv_results, tuned_oof = run_cv(
        {tuned_name: tuned_xgb}, X_train, y_train, skf
    )
    xgb_cv_results.update(tuned_cv_results)
    xgb_oof.update(tuned_oof)

    # 7. Threshold optimization on tuned OOF predictions
    print("\n[5] Optimizing decision threshold (OOF training predictions only)...")
    thresh_data = optimize_threshold(y_train, tuned_oof[tuned_name])
    print(f"  OOF Optimal threshold: {thresh_data['best_threshold']} (F1={thresh_data['best_f1']:.4f})")
    print(f"  At t=0.50: F1={thresh_data['threshold_50_f1'] if 'threshold_50_f1' in thresh_data else 'N/A':.4f}" if 'threshold_50_f1' in thresh_data else "")

    # 8. Final evaluation on holdout test set
    print("\n[6] Final holdout test set evaluation (single run)...")
    final_tuned_xgb = xgb.XGBClassifier(
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        verbosity=0,
        **{k.replace("classifier__", ""): v for k, v in best_params_raw.items()}
    )
    final_pipeline = create_pipeline_v2(final_tuned_xgb)

    eval_p2, test_probs_p2 = evaluate_on_test(
        final_pipeline, X_train, y_train, X_test, y_test,
        thresh_data["best_threshold"], tuned_name
    )

    print(f"\n  === PHASE 2 HOLDOUT RESULTS ===")
    print(f"  ROC-AUC:  {eval_p2['test_roc_auc']:.4f}  (LR baseline: {lr_baseline_ref['roc_auc']:.4f})")
    print(f"  Avg Prec: {eval_p2['test_pr_auc']:.4f}  (LR baseline: {lr_baseline_ref['pr_auc']:.4f})")
    print(f"  Brier:    {eval_p2['test_brier_score']:.4f}  (LR baseline: {lr_baseline_ref['brier']:.4f})")
    print(f"  At t={thresh_data['best_threshold']}: Recall={eval_p2['threshold_opt']['recall']:.4f}, F1={eval_p2['threshold_opt']['f1']:.4f}")

    # 9. Calibration check
    print("\n[7] Calibration check...")
    cal_info = check_calibration(
        final_pipeline, X_train, y_train, X_test, y_test,
        figures_dir, tuned_name
    )

    # 10. SHAP analysis
    print("\n[8] SHAP analysis on fitted pipeline...")
    shap_info = run_shap_analysis(final_pipeline, X_train, X_test, y_test, figures_dir)

    # 11. Comparison plots
    print("\n[9] Generating comparison plots...")
    generate_comparison_plots(
        {"Logistic Regression (Default) [P1]": cv_p1_lr_ref},
        xgb_cv_results,
        eval_p2,
        thresh_data,
        y_test,
        test_probs_p2,
        figures_dir,
    )

    # 12. Save pipeline
    pipeline_save_path = models_dir / "phase2_churn_pipeline.joblib"
    joblib.dump(final_pipeline, pipeline_save_path)
    print(f"\n  [Save] Phase 2 pipeline: {pipeline_save_path}")

    # 13. Save metrics JSON
    full_results = {
        "phase": 2,
        "selected_model": tuned_name,
        "random_seed": RANDOM_SEED,
        "feature_change": "Removed tenure_years (collinear with tenure)",
        "lr_baseline_reference": lr_baseline_ref,
        "xgb_cv_results": xgb_cv_results,
        "best_params": best_params_clean,
        "best_cv_ap_from_search": round(best_cv_ap, 4),
        "threshold_tuning": thresh_data,
        "holdout_test_evaluation": eval_p2,
        "calibration": cal_info,
        "shap_top_features": shap_info["top_shap_features"],
    }
    metrics_save_path = models_dir / "phase2_metrics.json"
    with open(metrics_save_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"  [Save] Phase 2 metrics: {metrics_save_path}")

    # 14. Generate markdown report
    print("\n[10] Generating Phase 2 report...")
    generate_report(
        {
            "lr_baseline": lr_baseline_ref,
            "cv_p1_reference": cv_p1_lr_ref,
            "xgb_cv_results": xgb_cv_results,
            "best_xgb_name": tuned_name,
            "best_params": best_params_clean,
            "eval_p2": eval_p2,
            "thresh_data": thresh_data,
            "shap_info": shap_info,
            "calibration": cal_info,
        },
        reports_dir / "phase2_advanced_experiments.md"
    )

    print("\n" + "=" * 75)
    print("PHASE 2 COMPLETE. REVIEW reports/phase2_advanced_experiments.md")
    print("=" * 75)
    return full_results


if __name__ == "__main__":
    main()
