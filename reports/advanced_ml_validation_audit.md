# Advanced ML Validation Audit Report
**Project:** Customer Churn Intelligence Platform  
**Audit Date:** 2026-10-22  
**Auditor:** Phase 1 Strict ML Validation  
**Python:** 3.14.0 | **scikit-learn:** ≥ 1.5.0 | **pytest:** 9.1.1  
**Audit Scope:** End-to-end inspection — data integrity, leakage, metric reproduction, inference consistency, automated tests.

---

## 1. Executive Summary

All primary claims in the project's existing reports were independently verified against actual code execution. The model metrics match exactly to 4 decimal places. No data leakage was found in the train/test split or the CV procedure. The inference pipeline is consistent across all three entry points. The automated test suite passes 6/6. Two documentation issues and one missing input-validation gap were identified, neither of which invalidates the model or its metrics.

**Final Verdict: ✅ READY FOR ADVANCED EXPERIMENTS**  
*(after one minor documentation fix — see Section 9)*

---

## 2. PASS / WARNING / FAIL Summary Table

| # | Check | Status | Evidence |
|:---:|:---|:---:|:---|
| 1 | Target definition: `Churn` binary (Yes→1, No→0) | ✅ PASS | `train_models.py:72` |
| 2 | `customerID` excluded from feature matrix | ✅ PASS | Dropped at `load_raw_dataset()` and `TelcoFeatureEngineer.transform()` |
| 3 | No duplicate rows in raw dataset | ✅ PASS | `df.duplicated().sum() == 0` |
| 4 | No null values in raw CSV | ✅ PASS | `df.isnull().sum()` all zero |
| 5 | `TotalCharges` whitespace handling (11 rows) | ✅ PASS | Coerced to `0.0` at feature engineer, matched to `tenure==0` rows |
| 6 | Stratified 80/20 split integrity | ✅ PASS | Train 26.54%, Test 26.54% churn rate; zero customerID overlap |
| 7 | Preprocessor fitted on training data only | ✅ PASS | `StandardScaler`/`OneHotEncoder` fitted inside CV folds and on full training set |
| 8 | Threshold selected on OOF training predictions only | ✅ PASS | `optimize_threshold(y_train, selected_oof_probs)` — test labels never touched |
| 9 | ROC-AUC reproduced | ✅ PASS | Reproduced: 0.8428 (reported: 0.8428) |
| 10 | Average Precision (PR-AUC) reproduced | ✅ PASS | Reproduced: 0.6351 (reported: 0.6351) |
| 11 | Brier Score reproduced | ✅ PASS | Reproduced: 0.1379 (reported: 0.1379) |
| 12 | All threshold metrics match | ✅ PASS | All 8 metrics at t=0.50 and t=0.35 reproduced exactly |
| 13 | Inference consistency across Streamlit/API/Batch | ✅ PASS | All three load the same `models/final_churn_pipeline.joblib` |
| 14 | Pipeline handles unseen categories gracefully | ✅ PASS | `handle_unknown='ignore'` encodes as all-zeros |
| 15 | Pipeline handles missing `TotalCharges` (None) | ✅ PASS | `pd.to_numeric(..., errors='coerce').fillna(0.0)` |
| 16 | Extra columns (customerID, Churn) stripped | ✅ PASS | `TelcoFeatureEngineer.transform()` drops them |
| 17 | Missing required column raises error | ⚠️ WARNING | Missing `Contract` raises `ValueError` — expected behavior but not explicitly tested or documented |
| 18 | Negative numeric inputs accepted silently | ⚠️ WARNING | Pipeline has no numeric guard; produces high-confidence erroneous output (prob=0.952 for tenure=-5, MonthlyCharges=-99) |
| 19 | Phase4 report narrative inconsistency | ⚠️ WARNING | Report text says "Gradient boosting captures nonlinear interactions" but winner is Logistic Regression |
| 20 | Redundant collinear features (`tenure` + `tenure_years`) | ℹ️ NOTE | Both in numeric features; perfectly collinear (ratio 12×). Harmless for LR with L2 regularization; worth noting for tree-based future models |
| 21 | Automated test suite: 6/6 passed | ✅ PASS | `pytest tests/test_pipeline.py -v` |
| 22 | No hardcoded absolute paths in production code | ✅ PASS | All paths use `pathlib.Path` relative construction |

---

## 3. Data Leakage & Split Integrity

### 3.1 Raw Dataset
| Property | Value | Status |
|:---|:---|:---:|
| Total rows | 7,043 | — |
| Total columns | 21 | — |
| Duplicate rows | **0** | ✅ |
| Null values (any column) | **None** | ✅ |
| `TotalCharges` blank strings (whitespace) | 11 rows | ✅ handled |
| Rows with `tenure == 0` | 11 rows | ✅ matched correctly |
| `customerID` uniqueness | 7,043 / 7,043 | ✅ |
| `SeniorCitizen` type | `int64` (0/1, not Yes/No) | ✅ handled |

> [!NOTE]
> The 11 `TotalCharges` whitespace rows all correspond to `tenure == 0` customers. The `TelcoFeatureEngineer` converts whitespace via `pd.to_numeric(..., errors='coerce').fillna(0.0)`, and then correctly sets `avg_historical_monthly = MonthlyCharges` when `tenure == 0`. This is the correct behavior and verified by `test_tenure_zero_handling`.

### 3.2 Train/Test Split
| Property | Value | Status |
|:---|:---|:---:|
| Split ratio | 80/20 (5,634 / 1,409) | ✅ |
| Stratified | Yes (`stratify=y`, `random_state=42`) | ✅ |
| Train churn rate | 26.54% | ✅ |
| Test churn rate | 26.54% | ✅ |
| `customerID` overlap between splits | **0** | ✅ |
| Index overlap between splits | **0** | ✅ |
| Saved CSV sizes match expected | y_train: 5,634 / y_test: 1,409 | ✅ |
| Saved CSV customerID overlap | **0** | ✅ |

### 3.3 Leakage Assessment

**Target leakage:** None detected. `Churn` is removed via `load_raw_dataset()` at line 73 of `train_models.py` and also stripped by `TelcoFeatureEngineer` (`Churn_bin` drop-guard at line 45 of `pipeline_components.py`).

**Identifier leakage:** None. `customerID` is excluded from `X` at loading and additionally stripped in the transformer.

**Preprocessing leakage:** None. `StandardScaler` and `OneHotEncoder` are embedded in the `sklearn.Pipeline` and fitted inside each CV fold separately (manual loop, lines 99–112 of `train_models.py`). This guarantees fold-safe cross-validation with no statistics computed from fold validation data.

**Threshold leakage:** None. `optimize_threshold(y_train, selected_oof_probs)` at line 618 of `train_models.py` only uses `y_train` and out-of-fold probabilities collected strictly during training folds. Test labels (`y_test`) are not accessed until the final single holdout evaluation.

**Feature engineering leakage risk:** All engineered features are **row-level** computations (within-row ratios and categorical comparisons). No cross-row statistics (means, variances, ranks) are computed prior to the split. This is consistent with the preprocessing report's documented architecture.

**Collinearity flag (informational, not a leak):** `tenure_years = tenure / 12.0` and `tenure` are both in the numeric features list. They are exactly linearly related. For `LogisticRegression` with L2 regularization (`C=1.0`), this is handled gracefully — the L2 penalty distributes coefficient weight across collinear features. However, this introduces unnecessary redundancy. Similarly, `TotalCharges` and `avg_historical_monthly = TotalCharges / tenure` are near-collinear. **Recommendation:** For documentation clarity, note this as a known design choice. For future tree-based models, consider dropping `tenure_years` to reduce feature count without information loss.

---

## 4. Reproduced Baseline Metrics

All metrics were reproduced by loading the saved `models/final_churn_pipeline.joblib` and calling `predict_proba(X_test)` **without any refitting**. The test set was reconstructed using the identical `train_test_split` parameters (same seed, same raw data, same stratify flag).

### 4.1 Threshold-Independent Metrics

| Metric | Reproduced | Reported | Match |
|:---|:---:|:---:|:---:|
| **ROC-AUC** | **0.8428** | 0.8428 | ✅ EXACT |
| **Average Precision (AP / PR-AUC)** | **0.6351** | 0.6351 | ✅ EXACT |
| **Brier Score** | **0.1379** | 0.1379 | ✅ EXACT |

> [!IMPORTANT]
> **AP vs. Trapezoidal PR-AUC:** `sklearn.metrics.average_precision_score` computes the interpolated **step-function** area under the Precision-Recall curve, not the trapezoidal integral. These differ numerically. The value 0.6351 is the **Average Precision (AP)** — correctly identified in the existing codebase as `average_precision_score`. All reports should use the term "Average Precision" or "AP" rather than "PR-AUC (trapezoidal)" to avoid ambiguity.

### 4.2 Dummy Baseline Comparison

| Metric | Logistic Regression | Dummy (Most Frequent) | Improvement |
|:---|:---:|:---:|:---:|
| ROC-AUC | 0.8428 | 0.5000 | **+0.3428** |
| Average Precision | 0.6351 | 0.2654 (prevalence) | **+0.3697** |
| Brier Score | 0.1379 | ~0.1949 (naive) | **better** |

The model is substantially above chance on both threshold-independent metrics.

---

## 5. Threshold Comparison

Decision threshold was selected by sweeping OOF training predictions to maximize F1 on training data only. The optimal threshold found was **t = 0.35**.

### 5.1 Metrics at Both Thresholds (Holdout Test Set, N=1,409)

| Metric | t = 0.50 | t = 0.35 | Notes |
|:---|:---:|:---:|:---|
| **Accuracy** | 0.7984 | 0.7750 | Slight drop expected with lower threshold |
| **Precision** | 0.6490 | 0.5603 | More false positives |
| **Recall** | 0.5241 | 0.7086 | **69 additional churners caught** |
| **F1-Score** | 0.5799 | **0.6257** | Optimized threshold wins |
| **Specificity** | 0.8976 | 0.7990 | More retained customers flagged |
| TN | 929 | 827 | |
| FP | 106 | 208 | |
| FN | 178 | 109 | |
| TP | 196 | 265 | |

**Threshold selection integrity:** VERIFIED. The threshold sweep function `optimize_threshold()` receives only `y_train` (5,634 labels) and `oof_probs` (accumulated from training folds). The `y_test` array is not read or referenced until `evaluate_on_test_set()` is called after threshold selection is complete.

**Business rationale:** At t=0.35, the model catches 70.86% of churners vs. 52.41% at t=0.50. This 18-point recall gain (+69 churners detected on a 1,409-sample test set) is justified by the cost asymmetry: preventing churn protects customer LTV (~$1,800) vs. cost of proactive retention outreach (~$25).

---

## 6. Calibration Findings

| Metric | Value | Interpretation |
|:---|:---:|:---|
| Brier Score | 0.1379 | Good calibration (lower is better; random = ~0.195) |
| Predicted prob range | [0.0033, 0.8894] | Full range, no extreme predictions |
| Calibration method | None applied (raw LR probabilities) | Logistic Regression is inherently well-calibrated |

> [!NOTE]
> Logistic Regression produces naturally calibrated probabilities due to its log-loss training objective. A `calibration_curve` plot was generated (`reports/figures/14_calibration_curve.png`) and the Brier score of 0.1379 confirms reasonable probability reliability. For the existing use case (risk tiering into Low/Moderate/High), raw LR probabilities are appropriate. If exact probability calibration is needed for downstream financial models, Platt scaling or isotonic regression could be added without touching the decision threshold logic.

---

## 7. Inference Consistency Results

All three inference entry points load the same `models/final_churn_pipeline.joblib`:

| Interface | Model File | Threshold | Status |
|:---|:---|:---:|:---:|
| `app.py` (Streamlit) | `models/final_churn_pipeline.joblib` | 0.35 (from `model_metrics.json`) | ✅ |
| `src/api.py` (FastAPI) | `models/final_churn_pipeline.joblib` | 0.35 (from `model_metrics.json`) | ✅ |
| `src/batch_score.py` (CLI) | `models/final_churn_pipeline.joblib` | 0.35 (default arg) | ✅ |

**Prediction consistency test:** Identical probability (0.634569) from all three-loaded pipeline instances for the same valid input. ✅

### Inference Edge Case Tests

| Test | Input | Result | Status |
|:---|:---|:---|:---:|
| Valid record | Standard customer | prob=0.6346, in [0,1] | ✅ PASS |
| Unseen categories | `InternetService='Quantum Fiber 9000'`, `PaymentMethod='Cryptocurrency Wallet'` | prob=0.6340, no crash | ✅ PASS |
| Missing `TotalCharges` (None) | `TotalCharges=None` | prob=0.5970, no crash | ✅ PASS |
| Missing required column | No `Contract` column | `ValueError: columns are missing: {'Contract'}` — expected | ⚠️ NOTE |
| Extra columns (Churn, random) | Added `Churn='Yes'`, `random_metric=9999` | prob=0.6346, unchanged | ✅ PASS |
| Scrambled column order | Reversed column list | prob=0.6346, identical | ✅ PASS |
| Zero tenure + whitespace TotalCharges | `tenure=0`, `TotalCharges='   '` | prob=0.6427, no crash | ✅ PASS |
| Negative numerics | `tenure=-5`, `MonthlyCharges=-99` | prob=0.9520, no crash — **no guard** | ⚠️ WARNING |

> [!WARNING]
> **No numeric validation in the ML pipeline itself.** The `TelcoFeatureEngineer` and `ColumnTransformer` do not validate that `tenure >= 0` or `MonthlyCharges >= 0`. The FastAPI layer uses Pydantic's `ge=0` constraint for HTTP API calls, but the Streamlit dashboard and batch CLI scorer accept any numeric value. A negative tenure customer would produce nonsensical feature engineering outputs (negative `tenure_years`, broken `avg_historical_monthly`) and a high churn probability (0.952 in the test). For the current portfolio use case this is low risk, but worth flagging for a production deployment hardening checklist.

### API Validation Boundary

The FastAPI Pydantic schema (`src/api.py`, `CustomerRecord`) enforces:
- `SeniorCitizen: int = Field(..., ge=0, le=1)` — only 0 or 1 accepted
- `tenure: int = Field(..., ge=0, le=100)` — non-negative, max 100
- `MonthlyCharges: float = Field(..., ge=0.0)` — non-negative
- Invalid schema returns HTTP 422 Unprocessable Entity ✅

This validation layer is **only active on the HTTP API path**. The Streamlit UI and batch scorer do not have equivalent guards.

---

## 8. Automated Test Results

**Command:** `python -m pytest tests/test_pipeline.py -v`  
**Platform:** win32, Python 3.14.0, pytest-9.1.1  
**Working directory:** `c:\Users\abhay\Churn_prediction`

```
============================= test session starts =============================
platform win32 -- Python 3.14.0, pytest-9.1.1, pluggy-1.7.0
collected 6 items

tests/test_pipeline.py::test_tenure_zero_handling                PASSED [ 16%]
tests/test_pipeline.py::test_feature_engineering_deterministic   PASSED [ 33%]
tests/test_pipeline.py::test_pipeline_prediction_shape_and_range PASSED [ 50%]
tests/test_pipeline.py::test_pipeline_isolation_of_extra_columns PASSED [ 66%]
tests/test_pipeline.py::test_unseen_category_handling            PASSED [ 83%]
tests/test_pipeline.py::test_threshold_decision_logic            PASSED [100%]

======================== 6 passed, 1 warning in 1.58s =========================
```

**Warning observed:** `sklearn` issues a `UserWarning` during `test_unseen_category_handling` when unseen categories are encountered in columns `[1, 9]` (InternetService, PaymentMethod). This is expected behavior with `handle_unknown='ignore'` and confirms the graceful fallback is working. The test passes correctly.

### Test Coverage Gaps Identified

| Gap | Severity | Recommendation |
|:---|:---:|:---|
| No test for negative `tenure` or `MonthlyCharges` | Low | Add guard + test for production hardening |
| No test for missing required column (e.g., `Contract`) | Low | Add test asserting `ValueError` is raised with a useful message |
| No test validating the saved model's holdout ROC-AUC matches expected (regression test) | Medium | Add metric regression test loading the saved model and asserting ROC-AUC ≥ 0.84 |
| No test for `app.py` or `api.py` integration | Low | Out of scope for Phase 1, but valuable for Phase 5 |

> [!TIP]
> A metric regression test (`test_holdout_roc_auc_above_baseline`) would catch model degradation if the pipeline is ever retrained and the `.joblib` artifact is replaced. Recommended for Phase 2.

---

## 9. Issues Found & Files Changed

### Issues Found

#### Issue 1 — Documentation Inconsistency in Phase 4 Report (WARNING)
**File:** [`reports/phase4_model_evaluation.md`](file:///c:/Users/abhay/Churn_prediction/reports/phase4_model_evaluation.md)  
**Lines:** Selection Rationale section  
**Description:** The narrative reads *"Gradient boosting captures nonlinear interactions between contract duration, fiber optic architecture, and monthly charges without overfitting."* However, the actual winning model (verified by `models/model_metrics.json`) is **Logistic Regression (Default)**, not a gradient boosting model. Logistic Regression had the highest CV PR-AUC (0.6601) vs. HistGradientBoosting Default (0.6363).  
**Risk:** Misleading to a technical reviewer who reads the narrative alongside the table.  
**Fix:** Update the narrative in `phase4_model_evaluation.md`.  
**Severity:** ⚠️ WARNING — documentation only, no impact on model or metrics.

#### Issue 2 — `average_precision_score` Labelled as "PR-AUC" (NOTE)
**File:** Multiple reports and `train_models.py`  
**Description:** `sklearn.metrics.average_precision_score` computes **Average Precision (AP)** using a step-function (not trapezoidal) summation. Calling it "PR-AUC" is technically ambiguous — trapezoidal PR-AUC would differ slightly. The value (0.6351) is correct; only the label is potentially misleading.  
**Fix:** Clarify in comments that "PR-AUC" in this project refers to Average Precision (AP) as computed by `sklearn.metrics.average_precision_score`.  
**Severity:** ℹ️ NOTE — no numerical impact.

#### Issue 3 — No Input Validation for Negative Numeric Values at Pipeline Level (WARNING)
**Files:** [`src/pipeline_components.py`](file:///c:/Users/abhay/Churn_prediction/src/pipeline_components.py), [`src/batch_score.py`](file:///c:/Users/abhay/Churn_prediction/src/batch_score.py)  
**Description:** `TelcoFeatureEngineer` and the downstream `ColumnTransformer` accept negative values for `tenure` and `MonthlyCharges` without raising an error. A customer with `tenure=-5, MonthlyCharges=-99` receives prob=0.952. The FastAPI layer validates these fields, but Streamlit and batch scorer do not.  
**Fix (recommended):** Add value clipping or assertion in `TelcoFeatureEngineer.transform()` for `tenure >= 0` and `MonthlyCharges >= 0`. Not applied during this audit per the no-unnecessary-changes rule.  
**Severity:** ⚠️ WARNING — acceptable for current portfolio use; should be addressed before production deployment.

### Files Changed During This Audit

| Action | File | Reason |
|:---|:---|:---|
| Created | `reports/advanced_ml_validation_audit.md` | This report |
| Created (scratch, safe to delete) | `audit_tasks.py` | Audit script — not a production file |
| Created (scratch, safe to delete) | `audit_feature_check.py` | Audit script — not a production file |

**No production files were modified.** The original model, pipeline, data, and application code are unchanged.

### Fix Applied: Phase 4 Report Narrative Correction

The selection rationale in [`reports/phase4_model_evaluation.md`](file:///c:/Users/abhay/Churn_prediction/reports/phase4_model_evaluation.md) contains a copy-paste error describing gradient boosting as the winner. A corrected version of the relevant paragraph is provided below for the user to review and apply:

> **Corrected text (Selection Rationale):**  
> `Logistic Regression (Default)` achieved the highest cross-validation PR-AUC (**0.6601**) and ROC-AUC (**0.8453**). Despite being a linear model, Logistic Regression with L2 regularization captures the strong linear and monotone relationships between tenure, contract type, monthly charges, and churn probability — as evidenced by the EDA findings. It also produces well-calibrated probabilities natively and is highly interpretable through odds ratios and feature coefficients.

---

## 10. Remaining Risks & Recommended Next Steps

### Remaining Risks

| Risk | Severity | Category |
|:---|:---:|:---|
| No input validation for negative numerics in pipeline / Streamlit / batch | Low-Medium | Robustness |
| Collinear features (`tenure` + `tenure_years`, `TotalCharges` + `avg_historical_monthly`) | Low | Feature Design |
| No metric regression test to detect model degradation if retrained | Medium | Testing |
| `PreprocessingReport` was built on a separate pipeline (`src/preprocessing.py`) that is NOT the inference pipeline — the inference pipeline is `pipeline_components.py`. The saved `models/preprocessor.joblib` is a standalone artifact not used by `final_churn_pipeline.joblib`. | Low | Documentation |
| Streamlit dashboard uses `train_test_split` to reconstruct test set at runtime — correct but fragile if raw data changes | Low | Architecture |
| `batch_score.py` hardcodes threshold=0.35 as default argument rather than reading from `model_metrics.json` | Low | Consistency |

> [!NOTE]
> The `models/preprocessor.joblib` artifact (saved by `src/preprocessing.py`) is a standalone `ColumnTransformer` that is **not used** by the production inference pipeline (`final_churn_pipeline.joblib`). The production pipeline embeds its own `ColumnTransformer` via `get_column_transformer()` in `pipeline_components.py`. The standalone preprocessor and the processed CSV files in `data/processed/` are legacy Phase 3 artifacts. This is not a bug — but it means the `preprocessor.joblib` file could mislead a reviewer into thinking it is the active preprocessing artifact.

### Recommended Next Steps (Phase 2)

1. **Fix Phase 4 report narrative** — correct "Gradient boosting" to "Logistic Regression" (5 minutes).
2. **Add metric regression test** — assert that the saved model's holdout ROC-AUC ≥ 0.84 to catch future regressions.
3. **Add missing-column and negative-input tests** — two additional pytest cases.
4. **Clarify preprocessor artifact** — add a comment to `models/preprocessor.joblib` usage noting it is a standalone Phase 3 artifact, not the active inference preprocessor.
5. **Update `batch_score.py`** — read threshold from `model_metrics.json` rather than hardcoding 0.35 as default, for full consistency.
6. **Phase 2 work:** Explore tree-based models (XGBoost, LightGBM) or hyperparameter tuning. Remove `tenure_years` from the feature set before re-benchmarking to reduce redundancy and improve interpretability.

---

## 11. Audit Decision

> [!IMPORTANT]
> **✅ READY FOR ADVANCED EXPERIMENTS**
>
> All model metrics are exactly reproduced. No data leakage found. The training, CV, threshold selection, and holdout evaluation procedures are correctly implemented and correctly isolated. The inference pipeline is consistent across all entry points. The automated test suite passes 6/6. The two documentation issues identified are minor and do not affect model correctness. The one missing validation gap (negative numeric inputs) is noted for production hardening but does not compromise the validity of the existing results.
>
> **The model and its reported metrics are trustworthy and suitable as a foundation for advanced experimentation.**
