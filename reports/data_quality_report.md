# Data Quality & Preprocessing Audit Report

## 1. Dataset Dimensions & Schema Validation
- **Source File:** `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Total Record Count:** 7,043 rows
- **Feature Space:** 21 raw attributes (1 primary key, 19 predictive signals, 1 binary outcome `Churn`)
- **Memory Footprint:** 977.5 KB on disk
- **Raw File Integrity:** 100% Unmodified and intact.

## 2. Integrity Checks & Missingness Analysis

| Column | Data Type | Null Count | Unique Count | Audit Findings |
| :--- | :---: | :---: | :---: | :--- |
| `customerID` | object | 0 | 7,043 | 100% unique alphanumeric ID. Excluded from predictive modeling. |
| `gender` | object | 0 | 2 | Balanced: Male (50.5%), Female (49.5%). Encoded as `gender_bin`. |
| `SeniorCitizen` | int64 | 0 | 2 | Binary flag (16.2% seniors). Maintained as numeric indicator. |
| `Partner` | object | 0 | 2 | 48.3% Yes, 51.7% No. Encoded as `Partner_bin`. |
| `Dependents` | object | 0 | 2 | 29.9% Yes, 70.1% No. Encoded as `Dependents_bin`. |
| `tenure` | int64 | 0 | 73 | Range: 0 to 72 months. Median: 29.0 months. 11 rows have `tenure == 0`. |
| `PhoneService` | object | 0 | 2 | 90.3% Yes, 9.7% No. |
| `MultipleLines` | object | 0 | 3 | 'No phone service' (9.7%), 'No' (48.1%), 'Yes' (42.2%). |
| `InternetService`| object | 0 | 3 | 'DSL' (34.4%), 'Fiber optic' (44.0%), 'No' (21.7%). |
| `OnlineSecurity`| object | 0 | 3 | 28.7% Yes, 49.7% No, 21.7% No internet service. |
| `OnlineBackup` | object | 0 | 3 | 34.5% Yes, 43.8% No, 21.7% No internet service. |
| `DeviceProtection`| object | 0 | 3 | 34.4% Yes, 43.9% No, 21.7% No internet service. |
| `TechSupport` | object | 0 | 3 | 29.0% Yes, 49.3% No, 21.7% No internet service. |
| `StreamingTV` | object | 0 | 3 | 38.4% Yes, 39.9% No, 21.7% No internet service. |
| `StreamingMovies`| object | 0 | 3 | 38.8% Yes, 39.5% No, 21.7% No internet service. |
| `Contract` | object | 0 | 3 | 'Month-to-month' (55.0%), 'One year' (20.9%), 'Two year' (24.1%). |
| `PaperlessBilling`| object | 0 | 2 | 59.2% Yes, 40.8% No. |
| `PaymentMethod` | object | 0 | 4 | 'Electronic check' (33.6%), 'Mailed check' (22.9%), Bank transfer (21.9%), Credit card (21.6%). |
| `MonthlyCharges`| float64 | 0 | 1,585 | Range: $18.25 to $118.75. Mean: $64.76. |
| `TotalCharges` | object | 0 (11 blank) | 6,531 | **Data anomaly:** Stored as string with 11 whitespace records (`" "`). Coerced to numeric `0.0`. |
| `Churn` | object | 0 | 2 | Target: 'No' = 5,174 (73.46%), 'Yes' = 1,869 (26.54%). |

## 3. Handling of Identified Data Anomalies
1. **TotalCharges Whitespace Handling:** Exactly 11 customer records contained blank string spaces `" "` instead of numeric figures. Cross-tabulation revealed that 100% of these records had `tenure == 0` (brand-new accounts signed up in the current billing cycle who have not received their first monthly statement).
   - **Resolution:** Coerced via `pd.to_numeric(..., errors='coerce').fillna(0.0)`.
   - **Historical Rate Resolution:** In `TelcoFeatureEngineer`, `avg_historical_monthly` is safely derived using `MonthlyCharges` as the baseline whenever `tenure == 0`, avoiding division-by-zero errors.
2. **Duplicate Records:** 0 duplicate rows detected across all 21 columns.
3. **Split Partitioning:** Stratified 80/20 train/test split created 5,634 training records and 1,409 holdout test records with 0 overlap in customer identities.
