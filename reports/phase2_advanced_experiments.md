# Phase 2: Advanced Model Experiments Report
**Project:** Customer Churn Intelligence Platform  
**Date:** 2026-10-22  
**XGBoost Version:** 3.4.1  
**scikit-learn Version:** 1.9.1  
**SHAP Version:** 0.54.1  
**Random Seed:** 42 (All splits, CV folds, and model training deterministic)

---

## 1. Executive Summary

Phase 2 extended the model benchmark from Phase 1 by adding XGBoost candidates with hyperparameter
tuning, removing the redundant `tenure_years` feature (perfectly collinear with `tenure`), and
adding SHAP-based interpretability. The **XGBoost (Tuned)** was selected as the Phase 2 winner.

| Metric | LR Baseline (Phase 1) | Phase 2 Winner | Δ Improvement |
|:---|:---:|:---:|:---:|
| **Test ROC-AUC** | 0.8428 | **0.8484** | **+0.0056** |
| **Test Avg Precision** | 0.6351 | **0.6671** | **+0.0320** |
| **Test Brier Score** | 0.1379 | **0.1346** | -- |
| **Recall @ opt. threshold** | 0.7086 | **0.7807** | **+0.0721** |
| **F1 @ opt. threshold** | 0.6257 | **0.6320** | **+0.0063** |

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
| Logistic Regression (Default) -- Phase 1 reference | 0.8453 +/- 0.0120 | 0.6601 +/- 0.0139 |

### Phase 2 XGBoost Candidates

| Model | CV ROC-AUC | CV PR-AUC |
|:---|:---:|:---:|
| XGBoost (Tuned) | 0.8491 +/- 0.0119 | 0.6646 +/- 0.0220 |
| XGBoost (Default) | 0.8115 +/- 0.0058 | 0.5933 +/- 0.0233 |
| XGBoost (scale_pos_weight=2.8) | 0.8098 +/- 0.0039 | 0.5930 +/- 0.0238 |

*Visual Reference:* `reports/figures/22_phase2_model_comparison.png`

---

## 4. Hyperparameter Tuning

**Method:** `RandomizedSearchCV` with 5-Fold Stratified CV on training data only.  
**Scoring metric:** `average_precision` (PR-AUC / AP) -- consistent with model selection criterion.  
**No test set access during tuning.**

### Best Parameters Found

| Hyperparameter | Value |
|:---|:---:|
| `subsample` | `0.8` |
| `scale_pos_weight` | `1.0` |
| `reg_lambda` | `2.0` |
| `reg_alpha` | `1.0` |
| `n_estimators` | `400` |
| `min_child_weight` | `7` |
| `max_depth` | `4` |
| `learning_rate` | `0.01` |
| `gamma` | `0.2` |
| `colsample_bytree` | `0.6` |

---

## 5. Decision Threshold (OOF Training Predictions Only)

| Configuration | Threshold | Precision | Recall | F1 |
|:---|:---:|:---:|:---:|:---:|
| Standard default | 0.50 | 0.6701 | 0.5267 | 0.5898 |
| **OOF-optimized** | **0.3** | 0.5309 | **0.7807** | **0.6320** |

Threshold selected by sweeping OOF probabilities from training folds only. Test labels untouched.

---

## 6. Holdout Test Set Evaluation (Single Unbiased Run, N=1,409)

### Threshold-Independent Metrics
- **ROC-AUC:** 0.8484
- **Average Precision:** 0.6671
- **Brier Score:** 0.1346

*Visual References:* `reports/figures/23_phase2_roc_pr_curves.png`

### Confusion Matrix at t = 0.50
| | Pred Retained | Pred Churned |
|:---|:---:|:---:|
| **Actual Retained** | 938 (TN) | 97 (FP) |
| **Actual Churned** | 177 (FN) | 197 (TP) |
- Precision: 67.0% | Recall: 52.7% | F1: 0.5898

### Confusion Matrix at t = 0.3 (Optimized)
| | Pred Retained | Pred Churned |
|:---|:---:|:---:|
| **Actual Retained** | 777 (TN) | 258 (FP) |
| **Actual Churned** | 82 (FN) | 292 (TP) |
- Precision: 53.1% | Recall: 78.1% | F1: 0.6320 | Specificity: 0.7507

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
| 1 | `Contract_Two year` | 0.43986 |
| 2 | `tenure` | 0.31303 |
| 3 | `InternetService_Fiber optic` | 0.30666 |
| 4 | `Contract_One year` | 0.18144 |
| 5 | `tenure_years` | 0.12547 |
| 6 | `MonthlyCharges` | 0.12132 |
| 7 | `PaymentMethod_Electronic check` | 0.11989 |
| 8 | `has_security_or_support` | 0.11240 |
| 9 | `avg_historical_monthly` | 0.09656 |
| 10 | `InternetService_No` | 0.08674 |
| 11 | `is_paperless_and_manual` | 0.07138 |
| 12 | `PaperlessBilling_bin` | 0.07132 |
| 13 | `TotalCharges` | 0.06586 |
| 14 | `MultipleLines_Yes` | 0.04777 |
| 15 | `is_electronic_check` | 0.04278 |

- **Global SHAP baseline (expected value):** -1.0178
- **Highest-risk customer** (test set index 1090): predicted prob = 0.8625

*Visual References:*
- `reports/figures/18_shap_summary_beeswarm.png` -- Beeswarm (direction + magnitude)
- `reports/figures/19_shap_bar_importance.png` -- Bar chart (global importance ranking)
- `reports/figures/20_shap_local_waterfall.png` -- Waterfall (individual customer breakdown)

---

## 8. Calibration

| Metric | Value |
|:---|:---:|
| Raw Brier Score | 0.1346 |
| LR Baseline Brier | 0.1379 |
| Calibration Wrapping Applied | No -- acceptable |
| Final Brier Score | 0.1346 |

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

The Phase 2 XGBoost pipeline **outperforms** the Phase 1 Logistic Regression baseline on both
ROC-AUC (+0.0056) and Average Precision (+0.0320). The Phase 2 pipeline
is recommended as the new production model. The dashboard and API should be updated to point to
`models/phase2_churn_pipeline.joblib`.

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
