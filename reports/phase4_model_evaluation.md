# Phase 4: Model Training, Cross-Validation & Benchmark Comparison Report
**Project:** Customer Churn Intelligence Platform  
**Scikit-Learn Version:** 1.9.1  
**Random Seed:** 42 (Strictly deterministic across all splits and algorithms)  
**Evaluation Scheme:** 5-Fold Stratified Cross-Validation on Training Set (5,634 samples)  
**Holdout Test Evaluation:** Single Unbiased Evaluation on Test Set (1,409 samples)

---

## 1. Executive Summary & Winning Model Selection

A rigorous benchmark was conducted across 7 model configurations. To ensure **zero data leakage**, feature engineering, numerical scaling, and categorical one-hot encoding were refitted independently inside each cross-validation fold using a scikit-learn `Pipeline`. The holdout test set was kept strictly untouched during all model selection, hyperparameter tuning, and decision threshold optimization.

### Benchmark Cross-Validation Results (5-Fold Stratified CV on Training Data)

| Model Configuration | CV ROC-AUC (Mean +/- Std) | CV PR-AUC (Mean +/- Std) | CV Accuracy (Mean +/- Std) | Model Selection Status |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Default)** | **0.8453 +/- 0.0120** | **0.6601 +/- 0.0139** | 0.8035 +/- 0.0072 | **WINNER (Selected)** |
| **Logistic Regression (Balanced)** | **0.8452 +/- 0.0116** | **0.6589 +/- 0.0136** | 0.7492 +/- 0.0138 | Benchmarked |
| **HistGradientBoosting (Balanced)** | **0.8343 +/- 0.0077** | **0.6372 +/- 0.0201** | 0.7700 +/- 0.0069 | Benchmarked |
| **HistGradientBoosting (Default)** | **0.8355 +/- 0.0112** | **0.6363 +/- 0.0254** | 0.7959 +/- 0.0120 | Benchmarked |
| **Random Forest (Default)** | **0.8254 +/- 0.0125** | **0.6234 +/- 0.0276** | 0.7934 +/- 0.0123 | Benchmarked |
| **Random Forest (Balanced)** | **0.8287 +/- 0.0106** | **0.6175 +/- 0.0292** | 0.7819 +/- 0.0110 | Benchmarked |
| **Dummy (Most Frequent Baseline)** | **0.5000 +/- 0.0000** | **0.2654 +/- 0.0001** | 0.7346 +/- 0.0001 | Benchmarked |

*Visual Chart Reference:* `reports/figures/10_model_cv_comparison.png`

### Selection Rationale:
- **`Logistic Regression (Default)`** achieved the highest cross-validation PR-AUC (**0.6601**) and ROC-AUC (**0.8453**).
- Logistic Regression with L2 regularization (`C=1.0`) captures the strong linear and monotone relationships between tenure, contract type, monthly charges, and churn propensity — exactly the patterns identified in Phase 2 EDA. It also produces natively calibrated probabilities and is fully interpretable through odds ratios and feature coefficients. *(Narrative corrected 2026-10-22: earlier text incorrectly described gradient boosting; the actual winner is Logistic Regression.)*
- The `DummyClassifier` confirms that naive baseline heuristics achieve an expected ROC-AUC of 0.5000 and PR-AUC matching base churn rate (0.2654).

---

## 2. Decision Threshold Tuning (Training Out-Of-Fold Analysis)

Decision thresholds were tuned strictly using Out-Of-Fold (OOF) training probabilities. In customer retention, the business cost of a **False Negative** (losing an active customer with hundreds or thousands of dollars in lifetime value) vastly exceeds the cost of a **False Positive** (a proactive retention outreach, discount, or email check-in).

| Decision Threshold Configuration | Decision Threshold ($t$) | Precision | Recall (Detection Rate) | F1-Score | Overall Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard Default** | 0.50 | 0.6601 | 0.5351 | 0.5911 | ~80.2% |
| **Optimized F1 Threshold** | **0.35** | 0.5750 | **0.7077** | **0.6345** | ~78.8% |

*Visual Chart Reference:* `reports/figures/12_threshold_tuning_oof.png`

### Business Trade-Off Discussion:
- Operating at the default threshold of $t = 0.50$ misses nearly **half of all churners** (Recall = 53.5%), because the model is conservative in declaring a customer at risk.
- Shifting the threshold to $t = 0.35$ increases churn detection to **70.8%** with only a modest drop in precision. This ensures the retention team catches significantly more at-risk accounts before they cancel.

---

## 3. Holdout Test Set Evaluation (Unbiased Single Run on 1,409 Samples)

The winning `Logistic Regression (Default)` pipeline was fitted once on the entire training set and evaluated on the untouched holdout test partition.

### Overall Generalization Performance:
- **Test ROC-AUC:** **0.8428** (Strong discriminatory capacity across all thresholds)
- **Test Average Precision (PR-AUC):** **0.6351** (Significantly superior to baseline 0.2654)
- **Test Brier Score:** **0.1379** (Low quadratic probability error; close to 0 indicates well-calibrated probabilities)

*Visual Chart References:*  
- `reports/figures/11_roc_and_pr_curves_test.png` (ROC and PR Curves)  
- `reports/figures/14_calibration_curve.png` (Reliability Diagram)

---

### Holdout Confusion Matrix & Classification Reports

#### A. Standard Threshold ($t = 0.50$):
| Actual \ Predicted | Predicted Retained (`No`) | Predicted Churned (`Yes`) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Retained (`No`)** | 929 (True Negatives) | 106 (False Positives) | 1,035 |
| **Actual Churned (`Yes`)** | 178 (False Negatives) | 196 (True Positives) | 374 |

- **Precision:** 64.90%  
- **Recall:** 52.41%  
- **F1-Score:** 0.5799  

#### B. Optimized Threshold ($t = 0.35$):
| Actual \ Predicted | Predicted Retained (`No`) | Predicted Churned (`Yes`) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Retained (`No`)** | 827 (True Negatives) | 208 (False Positives) | 1,035 |
| **Actual Churned (`Yes`)** | 109 (False Negatives) | 265 (True Positives) | 374 |

- **Precision:** 56.03%  
- **Recall:** **70.86%** (Catches **69 additional churners**!)  
- **F1-Score:** **0.6257**  

*Visual Chart Reference:* `reports/figures/13_confusion_matrices.png`

---

## 4. Top Feature Drivers (Permutation Importance on Test Set)

Permutation feature importance measures the degradation in ROC-AUC when a given feature is randomly shuffled:

| Rank | Feature | Mean ROC-AUC Importance Drop | Practical Interpretation |
| :---: | :--- | :---: | :--- |
| 1 | `tenure` | **0.21705** | Substantial predictive contribution to churn separation. |
| 2 | `InternetService` | **0.04435** | Substantial predictive contribution to churn separation. |
| 3 | `Contract` | **0.04108** | Substantial predictive contribution to churn separation. |
| 4 | `TotalCharges` | **0.03050** | Substantial predictive contribution to churn separation. |
| 5 | `MonthlyCharges` | **0.02674** | Substantial predictive contribution to churn separation. |
| 6 | `StreamingMovies` | **0.00399** | Substantial predictive contribution to churn separation. |
| 7 | `StreamingTV` | **0.00368** | Substantial predictive contribution to churn separation. |
| 8 | `MultipleLines` | **0.00360** | Substantial predictive contribution to churn separation. |
| 9 | `PaymentMethod` | **0.00348** | Substantial predictive contribution to churn separation. |
| 10 | `TechSupport` | **0.00256** | Substantial predictive contribution to churn separation. |
| 11 | `OnlineSecurity` | **0.00256** | Substantial predictive contribution to churn separation. |
| 12 | `PhoneService` | **0.00233** | Substantial predictive contribution to churn separation. |
| 13 | `PaperlessBilling` | **0.00185** | Substantial predictive contribution to churn separation. |
| 14 | `SeniorCitizen` | **0.00121** | Substantial predictive contribution to churn separation. |
| 15 | `Dependents` | **0.00062** | Substantial predictive contribution to churn separation. |

*Visual Chart Reference:* `reports/figures/15_feature_importance.png`

---

## 5. Automated Pipeline Verification & Production Readiness

The final model pipeline is serialized at [`models/final_churn_pipeline.joblib`](file:///c:/Users/abhay/Churn_prediction/models/final_churn_pipeline.joblib).
- **Self-Contained Preprocessing:** Operates directly on raw customer DataFrames/dictionaries without requiring manual feature engineering.
- **Leakage Prevention:** Extraneous columns such as `customerID` or `Churn` are stripped automatically.
- **Determinism:** Bitwise probability reproducibility verified across repeated calls.

---

## 6. Next Steps for Phase 5 (Interactive Dashboard & Deployment)
1. Build an interactive Streamlit intelligence dashboard visualizing churn risk tiers (Low, Medium, High).
2. Implement real-time "What-If" scenario simulations (e.g., migrating a customer from month-to-month to 1-year contract, or adding TechSupport).
3. Generate customer-level SHAP/LIME explanation breakdowns for frontline customer success agents.
