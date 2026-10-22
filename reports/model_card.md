# Model Card: Customer Churn Intelligence System

## 1. Model Details
- **Model Name:** Customer Churn Predictor (XGBoost Tuned Pipeline)
- **Model Version:** 2.0.0 (Production Champion)
- **Model Architecture:** scikit-learn Pipeline with:
  1. `TelcoFeatureEngineer`: Deterministic domain transformations and row-level features.
  2. `ColumnTransformer`: StandardScaler on 7 continuous features + OneHotEncoder (`drop='first'`, `handle_unknown='ignore'`) on 10 categorical features + binary passthroughs.
  3. `classifier`: `xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.01, subsample=0.8, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=2.0, min_child_weight=7, gamma=0.2, scale_pos_weight=1.0, random_state=42)`
- **Rollback Baseline:** `LogisticRegression(C=1.0, max_iter=1000, random_state=42)` (Pipeline v1.0.0 in `models/final_churn_pipeline.joblib`)
- **Release Date:** October 2026
- **License:** MIT License

## 2. Intended Use
- **Primary Use Case:** Proactive identification of at-risk subscription customers in telecommunication accounts.
- **Intended Users:** Retention Specialists, Customer Success Teams, and Growth Product Managers.
- **Decision Workflow:** Model assigns a calibrated continuous churn probability and risk tier (`LOW RISK`, `MODERATE RISK`, `HIGH RISK`) based on the OOF-optimized decision boundary ($t = 0.30$).
- **Out-of-Scope & Prohibited Uses:**
  - Automated termination, pricing discrimination, credit scoring, or unilateral denial of essential telecommunication services.
  - Evaluation of employees or vendor performance.

## 3. Factors and Cohort Attributes
- **Demographics:** Gender, SeniorCitizen status, Partner, Dependents.
- **Account Logistics:** Tenure, Contract type (Month-to-month, One year, Two year), PaperlessBilling, PaymentMethod.
- **Services Subscribed:** PhoneService, MultipleLines, InternetService (DSL, Fiber optic, No), OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies.
- **Charges:** MonthlyCharges, TotalCharges, derived average historical monthly rates and pricing discrepancies.

## 4. Evaluation Data & Protocol
- **Dataset:** IBM Telco Customer Churn (7,043 unique customer accounts).
- **Split Scheme:** 80% Training ($N = 5,634$) / 20% Holdout Test ($N = 1,409$), Stratified by target `Churn`.
- **Validation Isolation:** Zero test set leakage. Hyperparameter search (60 iterations) and decision threshold selection were performed strictly on 5-Fold Stratified Training Cross-Validation. Holdout test set evaluated exactly once.

## 5. Performance Metrics (Holdout Test Set)

| Metric | Champion (XGBoost Tuned) | Baseline (Logistic Regression) | Direction |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | **0.8484** | 0.8428 | +0.0056 |
| **Average Precision (PR-AUC)** | **0.6671** | 0.6351 | +0.0320 |
| **Brier Score (Reliability)** | **0.1346** | 0.1379 | -0.0033 (Better) |
| **Operational Threshold ($t$)** | **0.30** | 0.35 | Customized |
| **Recall @ Operational Threshold**| **78.07%** | 70.86% | +7.21% |
| **F1-Score @ Operational Threshold**| **0.6320** | 0.6257 | +0.0063 |
| **Precision @ Operational Threshold**| **53.09%** | 56.03% | -2.94% |
| **Specificity** | **75.07%** | 79.90% | -4.83% |

## 6. Interpretability & Explainability
- **Global Feature Impact:** Assessed via TreeExplainer SHAP summary and mean $|SHAP|$ attributions. Top drivers: `Contract_Two year`, `tenure`, `InternetService_Fiber optic`, `Contract_One year`, and `MonthlyCharges`.
- **Local Explanations:** Real-time customer waterfall decomposes log-odds contribution per factor.
- **Non-Causal Notice:** All attributions represent statistical associations controlling for multi-feature correlations; they do not prove direct causal mechanisms.

## 7. Limitations & Caveats
1. **Cross-Sectional Dataset:** Dataset records a single point-in-time snapshot. Dynamic time-series signals (e.g., changes in call drops, support ticket escalations in the last 14 days) are absent.
2. **Economic Environment Changes:** Inflationary adjustments, competitor promotional rollouts, or macro-economic pressures can shift baseline churn propensity beyond observed training distributions.
3. **Fiber Optic Cohort Risk:** High churn rates associated with Fiber Optic are coupled with higher monthly charges ($80-$100+). Outreach should focus on service reliability and bundled value rather than simple price reductions.
