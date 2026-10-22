# Phase 3 Preprocessing & Pipeline Audit Report
**Project:** Customer Churn Intelligence Platform  
**Audit Scope:** Verification of `src/preprocessing.py`, `reports/preprocessing_report.md`, `models/feature_metadata.json`, and generated train/test artifacts.  
**Raw Dataset Status:** Preserved and 100% Unmodified (`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`)  
**Audit Script:** [`src/audit_phase3.py`](file:///c:/Users/abhay/Churn_prediction/src/audit_phase3.py)

---

## Executive Audit Summary

| Check # | Audit Dimension | Status | Key Evidence / Findings |
| :---: | :--- | :---: | :--- |
| **1** | Target (`Churn`) & `customerID` Exclusion | **PASS** | Neither `customerID` nor `Churn` exists in `X_train.csv` or `X_test.csv` (both have exactly 43 predictor columns). |
| **2** | Prediction-Time Feature Availability | **PASS** | All 43 features are strictly row-level transformations derived from variables known at active customer status. |
| **3** | Formula Audit: `avg_historical_monthly` | **PASS** *(Corrected)* | Flaw identified in legacy `TotalCharges / (tenure + 1)` (caused 50% underestimation for month-1 customers). Safely corrected to `np.where(tenure == 0, MonthlyCharges, TotalCharges / tenure)`. |
| **4** | Characterization of `charge_discrepancy` | **PASS** | Audited to ensure discrepancy is documented as *rate divergence* (promotions ending, tier changes, add-on additions/removals) rather than definitive proof of unilateral price hikes. |
| **5** | Recalculation of Reported Churn Benchmarks | **PASS** | Exact recalculation directly from raw CSV matches: First-year (`tenure <= 12`) = 1,037 / 2,186 = **47.4382%** (47.44%); Electronic check = 1,071 / 2,365 = **45.2854%** (45.29%). |
| **6** | Stratified 80/20 Split & ID Alignment | **PASS** | 5,634 train (79.99%) vs. 1,409 test (20.01%). Train churn = 26.5353%; Test churn = 26.5436%. Zero ID overlap. Customer IDs 100% aligned with labels. |
| **7** | Preprocessing Fitting Strictly on Training Data | **PASS** | `StandardScaler` (7 means/scales) and `OneHotEncoder` (10 category sets) fitted exclusively on `X_train`. Holdout test set transformed without refitting. |
| **8** | Feature Matrix & Preprocessor Column Alignment | **PASS** | Column lists for `X_train`, `X_test`, `preprocessor.get_feature_names_out()`, and `feature_metadata.json` match identically in count (43) and order. |
| **9** | Consistent Handling of 11 Blank `TotalCharges` | **PASS** | All 11 blank values have `tenure = 0` (8 in train, 3 in test). Consistently imputed with `0.0` in both splits. |
| **10** | Systemic Integrity & Reusable Pipeline | **PASS** | Legacy formula defect corrected, pipeline re-fitted, artifacts synchronized, and raw CSV preserved. |

---

## Detailed Audit Findings

### Check 1: Target (`Churn`) and `customerID` Exclusion
- **Objective:** Ensure no identity tokens or target values leak into the predictor feature matrices.
- **Verification:**
  - `customerID` in `X_train.columns`: `False`
  - `customerID` in `X_test.columns`: `False`
  - `Churn` (or `Churn_bin`) in `X_train.columns`: `False`
  - `Churn` (or `Churn_bin`) in `X_test.columns`: `False`
- **Verdict:** **PASS**. Both `X_train` and `X_test` contain exactly 43 feature columns, strictly excluding identifiers and ground-truth targets.

---

### Check 2: Prediction-Time Availability & Data Leakage Review
- **Objective:** Verify that every engineered feature relies solely on information observable while the customer is active.
- **Verification:**
  - All engineered attributes (`tenure_years`, `is_first_year`, `is_long_term`, `avg_historical_monthly`, `charge_discrepancy`, `total_addons`, `has_security_or_support`, `has_streaming_services`, `is_electronic_check`, `is_auto_payment`, `is_paperless_and_manual`, `senior_living_alone`, `family_support_household`) are row-level functions.
  - No forward-looking target information or post-cancellation metrics are utilized.
- **Verdict:** **PASS**.

---

### Check 3: Audit of `avg_historical_monthly` Formula
- **Objective:** Evaluate `avg_historical_monthly = TotalCharges / (tenure + 1)`.
- **Finding & Defect Analysis:**
  - While adding `+ 1` avoided a division-by-zero exception when `tenure == 0`, it introduced an unintended mathematical distortion across all active tenures:
    - For customers with `tenure = 1`, `TotalCharges / 2` halved their historical monthly rate. The average for month-1 customers was computed as **$25.24** instead of the actual **$50.49** (which matches their true monthly charge of **$50.49**).
    - Across all active tenures (`tenure > 0`), the formula caused a **10.14% average downward bias**.
- **Corrected & Implemented Formulation:**
  ```python
  safe_tenure = np.where(df["tenure"] == 0, 1, df["tenure"])
  df["avg_historical_monthly"] = np.where(
      df["tenure"] == 0,
      df["MonthlyCharges"],
      (df["TotalCharges"] / safe_tenure)
  ).round(2)
  ```
- **Validation:**
  - For `tenure == 0`, historical monthly rate defaults to their contract baseline (`MonthlyCharges`).
  - For `tenure >= 1`, historical monthly rate equals `TotalCharges / tenure`, giving an unbiased estimate of past billing.
- **Verdict:** **PASS** *(Issue discovered, documented, and corrected in code and saved splits)*.

---

### Check 4: Audit of `charge_discrepancy` Characterization
- **Objective:** Ensure `charge_discrepancy` is not portrayed as proof of unilateral price hikes.
- **Verification:**
  - `charge_discrepancy` is defined as `MonthlyCharges - avg_historical_monthly`.
  - With the corrected formula:
    - Minimum discrepancy: **-$18.90** (customers who recently downgraded services or dropped add-ons)
    - Maximum discrepancy: **+$19.13** (customers who added services, lost promotional discounts, or upgraded)
    - Mean discrepancy: **$0.00** across the population.
  - The documentation in `reports/preprocessing_report.md` has been verified to characterize this feature as *rate divergence* (which can stem from plan upgrades, add-on additions/removals, or expiration of promotional introductory rates), rather than conclusive proof of company-wide price increases.
- **Verdict:** **PASS**.

---

### Check 5: Recalculation of Reported Benchmark Churn Rates
- **Objective:** Validate the reported 47.44% first-year churn rate and 45.29% electronic-check churn rate directly from the raw dataset (`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`).
- **Calculation 1: First-Year Churn Rate (`tenure <= 12` months)**
  - Churned Customers (Numerator): **1,037**
  - Total Customers (Denominator): **2,186**
  - Exact Calculation: $\frac{1037}{2186} = 0.4743824 \rightarrow \mathbf{47.44\%}$
  - *Context:* If defined strictly as months 0 through 11 (`tenure < 12`), the numerator is 999 and denominator is 2,069, yielding **48.28%**. The reported 47.44% correctly reflects the cohort with tenure through month 12.
- **Calculation 2: Electronic Check Churn Rate**
  - Churned Customers (Numerator): **1,071**
  - Total Customers (Denominator): **2,365**
  - Exact Calculation: $\frac{1071}{2365} = 0.4528541 \rightarrow \mathbf{45.29\%}$
- **Verdict:** **PASS**. Both rates match the raw source data with 100% precision.

---

### Check 6: Stratified 80/20 Split & Customer ID Alignment
- **Objective:** Verify partition proportions, class representation, and customer alignment.
- **Verification Metrics:**
  - Total Samples: 7,043
  - Training Samples: 5,634 (79.994%)
  - Test Samples: 1,409 (20.006%)
  - Training Churn Count: 1,495 (**26.5353%**)
  - Testing Churn Count: 374 (**26.5436%**)
  - Global Churn Rate: 26.5370% (Proportion difference between splits: **0.0083%**)
  - Set Intersection (`train_ids` $\cap$ `test_ids`): **0 records** (Strictly disjoint holdout partition)
  - Index & ID Alignment: Every row in `X_train.csv` corresponds to the exact same customer ID in `y_train.csv` and `train_full.csv`.
- **Verdict:** **PASS**.

---

### Check 7: Preprocessing Pipeline Fitting Strictly on Training Data
- **Objective:** Ensure no test-set information leaked into scaling or encoding parameters.
- **Verification:**
  - `preprocessor = joblib.load('models/preprocessor.joblib')`
  - `StandardScaler`: Fitted exclusively on the 5,634 training records (7 numerical features). Training means stored: `[32.49, 64.93, 2299.33, 2.71, 64.90, -0.01, 2.06]`.
  - `OneHotEncoder`: Categories learned exclusively on `X_train`. Unseen categories are safely ignored via `handle_unknown='ignore'`.
  - Holdout test split was transformed using the training parameters without refitting.
- **Verdict:** **PASS**.

---

### Check 8: Column Alignment Across Artifacts
- **Objective:** Confirm that feature columns match across training, test, and pipeline schema.
- **Verification:**
  - Columns in `X_train.csv`: 43
  - Columns in `X_test.csv`: 43
  - Columns from `preprocessor.get_feature_names_out()`: 43
  - Columns in `models/feature_metadata.json`: 43
  - Direct equality check: `train_cols == test_cols == prep_cols == meta_cols` evaluated to **True**.
- **Verdict:** **PASS**.

---

### Check 9: Consistency in Handling 11 Blank `TotalCharges` Values
- **Objective:** Verify the location and imputation of the 11 blank `TotalCharges` records.
- **Verification:**
  - Exactly 11 whitespace strings in raw CSV.
  - Distribution across splits:
    - **8 records** landed in the training split.
    - **3 records** landed in the test split.
  - Tenure verification: All 8 records in train have `tenure == 0`. All 3 records in test have `tenure == 0`.
  - Imputation consistency: Both splits had these records imputed with `0.0`.
- **Verdict:** **PASS**.

---

### Check 10: Overall Systemic Integrity & Reusability
- **Objective:** Audit system readiness for Phase 4 modeling.
- **Verification:**
  - The `tenure + 1` formula defect was identified, documented, and safely updated in `src/preprocessing.py`.
  - Processed CSVs (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, `train_full.csv`, `test_full.csv`), `models/preprocessor.joblib`, and `models/feature_metadata.json` were regenerated and validated.
  - Raw CSV `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` remains completely unmodified.
- **Verdict:** **PASS**.

---

## Conclusion & Readiness for Phase 4

The Phase 3 preprocessing architecture has undergone an end-to-end audit. All 10 checkpoints have passed verification. Zero data leakage exists, formulas are mathematically sound, class stratification is exact, and all pipeline assets are serialized and ready for model training.
