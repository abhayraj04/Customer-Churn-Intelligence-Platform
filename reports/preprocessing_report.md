# Phase 3: Data Preprocessing & Feature Engineering Report
**Project:** Customer Churn Intelligence Platform  
**Status:** Completed & Validated  
**Raw Dataset Integrity:** 100% Unmodified (`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`)

---

## 1. Executive Summary

Phase 3 established the end-to-end data preparation architecture for the Customer Churn Intelligence Platform. The pipeline handles missing values, applies domain-specific feature engineering inspired by Phase 2 EDA, performs an 80/20 stratified split, and encapsulates numerical scaling and categorical one-hot encoding in an exported scikit-learn `ColumnTransformer`.

### Pipeline Summary Metrics
| Pipeline Dimension | Value | Operational Notes |
| :--- | :--- | :--- |
| **Total Original Records** | 7,043 | Zero records dropped |
| **Training Records (80%)** | 5,634 | Class-stratified split |
| **Testing Records (20%)** | 1,409 | Holdout set strictly unseen |
| **Training Churn Rate** | 26.54% | 1,495 churned / 5,634 total |
| **Testing Churn Rate** | 26.54% | 374 churned / 1,409 total |
| **Transformed Feature Space** | **43 features** | Continuous, encoded, and binary features |
| **Serialized Pipeline Asset** | `models/preprocessor.joblib` | Production-ready transformer |

---

## 2. Feature Engineering Rationale & Formulations

Nine specialized features were engineered based on statistical signals discovered during Phase 2 EDA:

| Feature Name | Calculation / Formula | Business & Statistical Hypothesis |
| :--- | :--- | :--- |
| **`is_first_year`** | `tenure <= 12` | Directly targets the **47.44% first-year churn fragility window**. |
| **`is_long_term`** | `tenure >= 48` | Flags entrenched customers with sub-10% churn propensity. |
| **`tenure_years`** | `tenure / 12.0` | Provides linear scale of customer lifecycle longevity. |
| **`avg_historical_monthly`** | `TotalCharges / tenure` (or `MonthlyCharges` if `tenure==0`) | Historical average monthly billing rate without low-tenure denominator distortion. |
| **`charge_discrepancy`** | `MonthlyCharges - avg_historical_monthly` | Divergence between current billing rate and historical average (e.g., promotional expiration, plan migrations, or add-on adjustments). |
| **`total_addons`** | Count of active add-on services | Measures customer ecosystem engagement across 6 add-ons (higher depth = lower churn). |
| **`has_security_or_support`**| `(TechSupport == 'Yes') or (OnlineSecurity == 'Yes')` | Flags retention anchor services (shown in EDA to reduce churn ~3x). |
| **`is_electronic_check`** | `PaymentMethod == 'Electronic check'` | Explicitly isolates the highest-risk payment channel (**45.29% churn rate**). |
| **`is_auto_payment`** | `PaymentMethod contains 'automatic'` | Flags customers using hands-off automated renewal billing. |
| **`is_paperless_and_manual`**| `PaperlessBilling == 'Yes' & is_auto_payment == 0` | Customers receiving bills digitally but needing to pay manually each cycle (high cognitive friction). |
| **`senior_living_alone`** | `SeniorCitizen == 1 & Partner == 'No' & Dependents == 'No'` | Captures isolated senior demographics exhibiting elevated churn in EDA. |

---

## 3. Data Leakage Prevention Architecture

Data leakage is a critical point of failure in real-world ML systems. The following architecture safeguards data integrity:

```
Raw CSV (7,043)
       │
       ▼
Domain Feature Engineering (Row-level operations only)
       │
       ▼
Stratified Train/Test Split (80/20, stratify=y, seed=42)
      ┌┴─────────────────────────────┐
      ▼                              ▼
Training Set (5,634)           Test Set (1,409)
      │                              │
      ▼                              │
Fit ColumnTransformer                │
(StandardScaler + OneHotEncoder)      │
      │                              │
      ├──────────────────────────────┘
      ▼
Transform Both Splits Separately
      │
      ▼
Save X_train, y_train, X_test, y_test & preprocessor.joblib
```

1. **Row-Level Transformations Only Prior to Split:** All feature engineering relies strictly on within-row attributes (e.g., `tenure / 12`, `TotalCharges / (tenure + 1)`). No dataset-wide means, variances, or target correlations were calculated prior to the split.
2. **ColumnTransformer Fitting Strictly on `X_train`:** The `StandardScaler` calculates mean and standard deviation solely on the 5,634 training records. `X_test` is transformed using training statistics without recalculating metrics.
3. **OneHotEncoder Vocabulary:** The category dictionary is learned strictly on `X_train`. Unknown categories in future data are gracefully handled via `handle_unknown='ignore'`.

---

## 4. Final Feature Schema (Transformed Feature Space: 43 Columns)

### Numerical Features (Standardized with `StandardScaler`):
`tenure, MonthlyCharges, TotalCharges, tenure_years, avg_historical_monthly, charge_discrepancy, total_addons`

### Categorical Features (One-Hot Encoded with `drop='first'`):
`MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaymentMethod`

### Binary & Indicator Features (0/1 Passthrough):
`SeniorCitizen, gender_bin, Partner_bin, Dependents_bin, PhoneService_bin, PaperlessBilling_bin, is_first_year, is_long_term, has_security_or_support, has_streaming_services, is_electronic_check, is_auto_payment, is_paperless_and_manual, senior_living_alone, family_support_household`

---

## 5. Artifact Verification

All generated data splits and pipeline objects have been written and verified:
- `data/processed/X_train.csv` (5,634 rows, 43 columns)
- `data/processed/X_test.csv` (1,409 rows, 43 columns)
- `data/processed/y_train.csv` (5,634 rows)
- `data/processed/y_test.csv` (1,409 rows)
- `data/processed/train_full.csv` (includes `customerID`, features, and `Churn`)
- `data/processed/test_full.csv` (includes `customerID`, features, and `Churn`)
- `models/preprocessor.joblib` (Fitted ColumnTransformer object)
- `models/feature_metadata.json` (Structured pipeline schema)
