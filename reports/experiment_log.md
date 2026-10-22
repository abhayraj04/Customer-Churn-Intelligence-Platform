# Machine Learning Experiment & Model Evaluation Log

## Experiment Summary Timeline
- **Random Seed:** 42 across all experiments, CV fold splits, and stochastic estimators.
- **Evaluation Split:** 5,634 Train / 1,409 Test (Stratified 80/20, untouched test set).
- **Primary Optimization Metric:** PR-AUC / Average Precision (due to 26.54% class imbalance).

---

## Benchmark Cross-Validation Log (5-Fold Stratified on Training Data)

| Model Architecture | Hyperparameters / Setup | CV ROC-AUC | CV PR-AUC | CV Accuracy | Selection Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **DummyClassifier** | `strategy="most_frequent"` | 0.5000 ± 0.0000 | 0.2654 ± 0.0001 | 73.46% | Unskilled chance baseline. |
| **Logistic Regression (Default)** | `C=1.0, max_iter=1000` | 0.8453 ± 0.0120 | 0.6601 ± 0.0139 | 80.35% | High linear benchmark; well-calibrated probabilities. |
| **Logistic Regression (Balanced)**| `class_weight="balanced"` | 0.8452 ± 0.0116 | 0.6589 ± 0.0136 | 74.92% | Shifts threshold internally; slight precision degradation. |
| **Random Forest (Default)** | `n_estimators=100, max_depth=None` | 0.8254 ± 0.0125 | 0.6234 ± 0.0276 | 79.34% | Tendency to overfit noisy tabular categories. |
| **Random Forest (Balanced)** | `class_weight="balanced"` | 0.8287 ± 0.0106 | 0.6175 ± 0.0292 | 78.19% | Marginally improved recall, PR-AUC unchanged. |
| **HistGradientBoosting (Default)**| `learning_rate=0.1, max_iter=100` | 0.8355 ± 0.0112 | 0.6363 ± 0.0254 | 79.59% | Fast tree ensemble; competitive baseline. |
| **HistGradientBoosting (Balanced)**| `class_weight="balanced"` | 0.8343 ± 0.0077 | 0.6372 ± 0.0201 | 77.00% | High recall configuration. |
| **XGBoost (Default)** | `n_estimators=200, max_depth=6` | 0.8115 ± 0.0058 | 0.5933 ± 0.0233 | 77.60% | Overfitting without regularization. |
| **XGBoost (Tuned Champion)** | `n_estimators=400, max_depth=4, lr=0.01, subsample=0.8, colsample=0.6, reg_alpha=1.0, reg_lambda=2.0` | **0.8491 ± 0.0119** | **0.6646 ± 0.0220** | **80.46%** | **Top CV Performer. Selected as Champion.** |

---

## Holdout Test Set Evaluation (Unbiased Single Evaluation on 1,409 Samples)

| Evaluation Metric | Baseline: Logistic Regression (v1) | Champion: Tuned XGBoost (v2) | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Holdout ROC-AUC** | 0.8428 | **0.8484** | **+0.0056** |
| **Holdout PR-AUC (Average Precision)**| 0.6351 | **0.6671** | **+0.0320** |
| **Brier Score (Probability Error)** | 0.1379 | **0.1346** | **-0.0033 (Better)** |
| **Operational Threshold ($t$)** | 0.35 | **0.30** | Selected via OOF Max-F1 |
| **Holdout Accuracy** | 77.50% | 75.87% | -1.63% |
| **Holdout Precision** | 56.03% | 53.09% | -2.94% |
| **Holdout Recall (Churn Coverage)** | 70.86% (265 / 374) | **78.07% (292 / 374)** | **+7.21% (+27 churners caught)** |
| **Holdout F1-Score** | 0.6257 | **0.6320** | **+0.0063** |
| **Holdout Specificity** | 79.90% | 75.07% | -4.83% |
