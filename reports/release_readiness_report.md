# Final Release-Readiness Audit Report
**Project:** Customer Churn Intelligence Platform  
**Audit Date:** 2026-10-22  
**Audit Scope:** End-to-end verification of models, inference pipelines, Streamlit dashboard, FastAPI service, automated tests, explainability, security, and portability.  
**Raw Dataset Integrity:** 100% Intact & Read-Only (`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`)

---

## Executive Audit Scorecard

| Check # | Audit Dimension | Status | Key Verification Evidence |
| :---: | :--- | :---: | :--- |
| **1** | Model & Inference Pipeline Loading | **PASS** | Serialized pipeline loads cleanly as `sklearn.pipeline.Pipeline` with steps `['feature_engineer', 'preprocessor', 'classifier']` and classifier `LogisticRegression`. |
| **2** | Cross-Module Pipeline Alignment | **PASS** | `app.py`, `src/api.py`, and `src/batch_score.py` all consume the exact same `models/final_churn_pipeline.joblib` asset. |
| **3** | Automated Test Suite Execution | **PASS** | `pytest tests/test_pipeline.py -v` executed: **6 passed** in 2.86s (100% passing). |
| **4** | Metric Consistency Verification | **PASS** | Exact match with documented evaluation artifacts: 5-Fold CV PR-AUC = **0.6601 ± 0.0139**, CV ROC-AUC = **0.8453 ± 0.0120**. |
| **5** | Holdout Test Performance | **PASS** | Holdout ROC-AUC = **0.8428**, PR-AUC = **0.6351**, Brier Score = **0.1379**, Selected Threshold = **0.35**. |
| **6** | Zero Test-Set Leakage in Thresholding | **PASS** | Decision threshold was chosen strictly via Out-Of-Fold (OOF) cross-validation predictions on `X_train_raw`. Holdout test labels were untouched. |
| **7** | Model Robustness & Edge Cases | **PASS** | Scrambled column orders, novel categories, zero-tenure whitespace strings, and missing optional values all predict successfully without errors. |
| **8** | Live Service Verification | **PASS** | **Streamlit** active on `http://localhost:8501` (HTTP 200). **FastAPI** verified: `GET /` (200), `POST /predict` valid (200), `POST /predict` invalid schema (422). |
| **9** | Explainability Non-Causal Audit | **PASS** | Odds Ratios and coefficients are characterized strictly as *multivariable statistical associations*, avoiding unsupported causal assertions. |
| **10** | Portability & Requirements Packaging | **PASS** | `requirements.txt`, `Dockerfile`, and `README.md` verified. Zero hardcoded absolute paths in production code files. |
| **11** | Privacy & Security Review | **PASS** | Zero PII columns detected; customer IDs are anonymized alphanumeric tokens. Security recommendations documented for production API hosting. |
| **12** | Release Readiness Verdict | **PASS** | **Platform is fully validated and ready for production portfolio deployment.** |

---

## Detailed Check-by-Check Audit Evidence

### Check 1: Model Pipeline Loading
- **File:** `models/final_churn_pipeline.joblib` (8.4 KB)
- **Pipeline Architecture:**
  1. `feature_engineer`: Custom `TelcoFeatureEngineer` (scikit-learn `BaseEstimator` / `TransformerMixin`)
  2. `preprocessor`: `ColumnTransformer` (`StandardScaler` on 7 continuous features + `OneHotEncoder(drop='first')` on 10 categorical features + passthrough on 15 binary indicators)
  3. `classifier`: `LogisticRegression(C=1.0, max_iter=1000, random_state=42)`
- **Loading Test:** Instantiated and verified via `joblib.load()`. Status: **PASS**.

---

### Check 2: Pipeline Alignment Across Applications
- **Streamlit Application (`app.py`):** Loads `models/final_churn_pipeline.joblib`.
- **FastAPI REST API (`src/api.py`):** Loads `models/final_churn_pipeline.joblib`.
- **CLI Batch Scorer (`src/batch_score.py`):** Loads `models/final_churn_pipeline.joblib`.
- **Verification:** All interfaces execute predictions through the exact same unified pipeline, guaranteeing 100% consistency across UI, API, and batch jobs. Status: **PASS**.

---

### Check 3: Automated Test Suite
- **Command:** `pytest tests/test_pipeline.py -v`
- **Results:**
  - `test_tenure_zero_handling`: **PASSED** (Confirms zero-tenure and blank whitespace TotalCharges do not cause division-by-zero or crash)
  - `test_feature_engineering_deterministic`: **PASSED** (Bitwise reproducibility verified)
  - `test_pipeline_prediction_shape_and_range`: **PASSED** (Shape `(1, 2)` and probabilities in $[0, 1]$ summing to 1.0)
  - `test_pipeline_isolation_of_extra_columns`: **PASSED** (Extraneous columns like `customerID` or `Churn` safely stripped)
  - `test_unseen_category_handling`: **PASSED** (Unknown categories gracefully encoded as zeros via `handle_unknown='ignore'`)
  - `test_threshold_decision_logic`: **PASSED** (Decision boundaries verified)
- **Summary:** **6 passed in 2.86s**. Status: **PASS**.

---

### Check 4 & 5: Verified Cross-Validation & Holdout Metrics

#### 5-Fold Stratified Cross-Validation Benchmark (Training Set, N=5,634):
- **Selected Model:** `Logistic Regression (Default)`
- **CV ROC-AUC:** **0.8453 ± 0.0120**
- **CV PR-AUC:** **0.6601 ± 0.0139** (2.5× higher than 0.2654 baseline)
- **CV Accuracy:** **79.84%**

#### Holdout Test Set Generalization (N=1,409):
- **ROC-AUC:** **0.8428**
- **PR-AUC (Average Precision):** **0.6351**
- **Brier Score:** **0.1379** (Reliability assessment: well-calibrated probabilities)
- **Decision Threshold:** **0.35**

#### Holdout Confusion Matrix Breakdown ($t = 0.35$ vs. $t = 0.50$):
```
Standard Threshold (t = 0.50):
                 Predicted Retained   Predicted Churned
Actual Retained         929                  106
Actual Churned          178                  196
Metrics: Precision = 64.90%, Recall = 52.41%, F1 = 0.5799

Optimized Threshold (t = 0.35):
                 Predicted Retained   Predicted Churned
Actual Retained         827                  208
Actual Churned          109                  265
Metrics: Precision = 56.03%, Recall = 70.86%, F1 = 0.6257
```

#### Threshold Selection Rule:
The threshold $t = 0.35$ was selected to maximize the F1-score across Out-Of-Fold training predictions, recognizing the asymmetry in retention economics: catching an additional churner (protecting ~$1,800 LTV) outweighs the nominal cost of proactive retention outreach (~$25).

---

### Check 6: Threshold Selection Isolation
- Audited `src/train_models.py` lines 615–620:
  `optimize_threshold(y_train, selected_oof_probs)`
- The function receives strictly `y_train` and out-of-fold cross-validation probabilities. Neither holdout test features (`X_test`) nor holdout test labels (`y_test`) were accessed. Status: **PASS**.

---

### Check 7: Input Robustness & Permutation Invariance
- **Scrambled Column Ordering:** Shuffling dictionary keys produced identical output ($p = True$).
- **Unseen Categories:** Input with `InternetService = "Quantum Fiber 9000"` and `PaymentMethod = "Cryptocurrency Wallet"` executed safely without raising exceptions ($p \in [0, 1]$).
- **Zero Tenure & Whitespace Strings:** Handled gracefully ($p \in [0, 1]$). Status: **PASS**.

---

### Check 8: Live Services Verification
- **Streamlit Application:** Active and returning HTTP 200 on `http://localhost:8501`.
- **FastAPI Endpoints:**
  - `GET /`: Returned HTTP 200 (`{"status": "Operational", "model_loaded": true}`).
  - `POST /predict` (valid customer): Returned HTTP 200 with calibrated probability, risk tier, and recommended action.
  - `POST /predict` (invalid schema: `SeniorCitizen = 99`): Returned HTTP 422 Unprocessable Entity with clear validation errors. Status: **PASS**.

---

### Check 9: Explainability Language Audit
- Audited `src/explainability.py`, `models/model_coefficients.csv`, and documentation.
- All odds ratios and coefficients are described strictly as *statistical associations* controlling for other predictors, rather than causal mechanisms. Status: **PASS**.

---

### Check 10: Portability & Clean Installation
- `requirements.txt` specifies pinned versions for pandas, numpy, scikit-learn, joblib, matplotlib, seaborn, streamlit, fastapi, uvicorn, pydantic, and pytest.
- Zero machine-specific absolute Windows paths (`C:/Users/...`) exist in production code files. All paths utilize Python `pathlib.Path` relative paths. Status: **PASS**.

---

### Check 11: Privacy & Security
- **PII Audit:** Dataset contains zero direct personal identifiers (no names, telephone numbers, emails, physical addresses, or payment card numbers).
- **Anonymization:** `customerID` values are synthetic alphanumeric tokens (`dddd-XXXXX`).
- **Production API Hardening Guidance:**
  - For public cloud deployment, wrap FastAPI endpoints with API Key headers or OAuth2 Bearer tokens.
  - Deploy behind an Nginx reverse proxy with HTTPS/TLS encryption. Status: **PASS**.

---

## Portfolio Demonstration Quick-Start Guide

To demonstrate the platform on any machine:

1. **Clone & Setup Environment:**
   ```bash
   git clone <repo-url>
   cd Churn_prediction
   python -m venv .venv
   source .venv/bin/activate  # Or on Windows: .venv\Scriptsctivate
   pip install -r requirements.txt
   ```

2. **Launch Interactive Dashboard:**
   ```bash
   streamlit run app.py
   ```
   Open **http://localhost:8501** to demonstrate:
   - Single Customer Risk Scoring & Real-Time "What-If" Counterfactual Simulator.
   - Cohort Analytics & Permutation Feature Importances.
   - Decision Threshold Simulator with Financial ROI Modeling.
   - Strategic Retention Playbook.

3. **Launch REST API Server:**
   ```bash
   uvicorn src.api:app --reload --port 8000
   ```
   Explore Swagger UI at **http://localhost:8000/docs**.

4. **Run Automated Test Suite:**
   ```bash
   pytest tests/test_pipeline.py -v
   ```
