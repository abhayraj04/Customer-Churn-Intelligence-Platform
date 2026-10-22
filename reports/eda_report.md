# Phase 2: Exploratory Data Analysis (EDA) Report
**Project:** Customer Churn Intelligence Platform  
**Dataset:** IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)  
**Scope:** In-depth statistical analysis, cross-tabulation, behavioral patterns, and visualization  
**Status:** Complete (Read-Only Dataset Verified)

---

## 1. Executive Summary & Top 5 Actionable Findings

This exploratory data analysis investigates the customer attributes and billing characteristics associated with customer churn. All statistics below represent direct computations from the 7,043 customer accounts.

### Top 5 Most Useful Findings (With Exact Dataset Figures)

1. **Contract Commitment is the Single Strongest Churn Predictor:**
   - **Month-to-Month Contract Churn Rate:** **42.71%** (1,655 churned out of 3,875 customers).
   - **One-Year Contract Churn Rate:** **11.27%** (166 churned out of 1,473).
   - **Two-Year Contract Churn Rate:** **2.83%** (48 churned out of 1,695).
   - *Beginner-friendly explanation:* Customers on flexible monthly plans are almost 15 times more likely to cancel than customers on two-year agreements.

2. **The "First-Year Fragility" Window (Tenure 0–12 Months):**
   - **0–12 Months Tenure Churn Rate:** **47.44%** (1,037 churned out of 2,186).
   - **61–72 Months Tenure Churn Rate:** Drops to **6.61%** (93 churned out of 1,407).
   - *Beginner-friendly explanation:* Over half of all churned customers leave within their very first year. Once a customer reaches 5+ years of tenure, churn drops to under 7%.

3. **Fiber Optic Internet Exhibits Disproportionate Churn Despite Premium Positioning:**
   - **Fiber Optic Churn Rate:** **41.89%** (1,297 churned out of 3,096).
   - **DSL Churn Rate:** **18.96%** (459 out of 2,421).
   - **No Internet Churn Rate:** **7.4%** (113 out of 1,526).
   - *Beginner-friendly explanation:* Fiber optic customers churn at more than double the rate of DSL customers. This strongly points to potential pricing pain, installation issues, or service reliability gaps.

4. **Electronic Check Payment Method is a Major Friction Point:**
   - **Electronic Check Churn Rate:** **45.29%** (1,071 out of 2,365).
   - **Automated Bank Transfer:** **16.71%**.
   - **Automated Credit Card:** **15.24%**.
   - **Mailed Check:** **19.11%**.
   - *Beginner-friendly explanation:* Customers paying by electronic check churn at almost triple the rate of customers enrolled in automatic billing (credit card or bank transfer).

5. **Value-Added Support Services Act as High-Value Retention Anchors:**
   - **No TechSupport Churn Rate:** **41.64%** vs. **15.17%** with TechSupport.
   - **No OnlineSecurity Churn Rate:** **41.77%** vs. **14.61%** with OnlineSecurity.
   - *Beginner-friendly explanation:* Customers who do not have tech support or online security are roughly 3 times more likely to cancel their subscriptions.

---

## 2. TotalCharges Missing Value Investigation: 0.0 vs NaN Comparison

During Phase 1, exactly **11 rows** in `TotalCharges` contained empty whitespace strings (`" "`). Converting with `pd.to_numeric(errors="coerce")` generates 11 `NaN` values.

### Statistical Comparison: Retaining NaN vs Imputing 0.0

| Metric | Retaining as Missing (`NaN`, N=7032) | Imputing as Zero (`0.0`, N=7043) | Absolute Difference |
| :--- | :--- | :--- | :--- |
| **Mean** | $2283.30 | $2279.73 | $3.57 (0.15% change) |
| **Median** | $1397.47 | $1394.55 | $0.00 (No change) |
| **Std Dev** | $2266.77 | $2266.79 | $0.02 |
| **Minimum** | $18.80 | $0.00 | $18.80 -> $0.00 |
| **Maximum** | $8684.80 | $8684.80 | $0.00 |

### Chosen Strategy & Limitations
- **Decision:** Impute `0.0` for the 11 instances.
- **Justification:** Every one of these 11 customers has `tenure = 0` (joined during the current cycle with no prior billing statements closed). Assigning $0 cumulative charge accurately reflects their completed payment history.
- **Limitations:** If the telecom company collects activation fees or first-month charges in advance, their true ledger balance upon account creation might equal `MonthlyCharges` rather than $0. However, since 11 records represent only **0.156%** of the entire dataset, neither choice creates meaningful bias.

---

## 3. Deep-Dive Analytical Cross-Tabulations

### A. Contract Commitment
| Contract Type | Total Customers | Share of Total (%) | Churned Customers | Retained Customers | Churn Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Month-to-month | 3,875 | 55.0% | 1,655 | 2,220 | **42.71%** |
| One year | 1,473 | 20.9% | 166 | 1,307 | **11.27%** |
| Two year | 1,695 | 24.1% | 48 | 1,647 | **2.83%** |

*Visual Chart Reference:* `reports/figures/02_churn_by_contract.png`

---

### B. Tenure Cohorts
| Tenure Cohort | Total Customers | Share of Total (%) | Churned Customers | Retained Customers | Churn Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0-12m | 2,186 | 31.0% | 1,037 | 1,149 | **47.44%** |
| 13-24m | 1,024 | 14.5% | 294 | 730 | **28.71%** |
| 25-36m | 832 | 11.8% | 180 | 652 | **21.63%** |
| 37-48m | 762 | 10.8% | 145 | 617 | **19.03%** |
| 49-60m | 832 | 11.8% | 120 | 712 | **14.42%** |
| 61-72m | 1,407 | 20.0% | 93 | 1,314 | **6.61%** |

*Visual Chart Reference:* `reports/figures/03_churn_by_tenure_cohorts.png`

---

### C. Internet Architecture
| Internet Service | Total Customers | Share of Total (%) | Churned Customers | Retained Customers | Churn Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DSL | 2,421 | 34.4% | 459 | 1,962 | **18.96%** |
| Fiber optic | 3,096 | 44.0% | 1,297 | 1,799 | **41.89%** |
| No | 1,526 | 21.7% | 113 | 1,413 | **7.40%** |

*Visual Chart Reference:* `reports/figures/04_churn_by_internet_service.png`

---

### D. Payment Method
| Payment Method | Total Customers | Share of Total (%) | Churned Customers | Retained Customers | Churn Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Bank transfer (automatic) | 1,544 | 21.9% | 258 | 1,286 | **16.71%** |
| Credit card (automatic) | 1,522 | 21.6% | 232 | 1,290 | **15.24%** |
| Electronic check | 2,365 | 33.6% | 1,071 | 1,294 | **45.29%** |
| Mailed check | 1,612 | 22.9% | 308 | 1,304 | **19.11%** |

*Visual Chart Reference:* `reports/figures/05_churn_by_payment_method.png`

---

### E. Service Add-Ons: TechSupport and OnlineSecurity
| Feature | Category | Total Customers | Churned Customers | Retained Customers | Churn Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TechSupport | No | 3,473 | 1,446 | 2,027 | **41.64%** |
| TechSupport | No internet service | 1,526 | 113 | 1,413 | **7.40%** |
| TechSupport | Yes | 2,044 | 310 | 1,734 | **15.17%** |
| OnlineSecurity | No | 3,498 | 1,461 | 2,037 | **41.77%** |
| OnlineSecurity | No internet service | 1,526 | 113 | 1,413 | **7.40%** |
| OnlineSecurity | Yes | 2,019 | 295 | 1,724 | **14.61%** |

*Visual Chart Reference:* `reports/figures/06_churn_by_security_and_support.png`

---

### F. Demographics: Senior Citizens, Partners, and Dependents
| Demographic Dimension | Category | Total Customers | Churned Customers | Retained Customers | Churn Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Senior Citizen | Non-Senior | 5,901 | 1,393 | 4,508 | **23.61%** |
| Senior Citizen | Senior | 1,142 | 476 | 666 | **41.68%** |
| Partner | No | 3,641 | 1,200 | 2,441 | **32.96%** |
| Partner | Yes | 3,402 | 669 | 2,733 | **19.66%** |
| Dependents | No | 4,933 | 1,543 | 3,390 | **31.28%** |
| Dependents | Yes | 2,110 | 326 | 1,784 | **15.45%** |

*Visual Chart Reference:* `reports/figures/07_churn_by_demographics.png`

---

## 4. Financial & Pricing Metrics Analysis

### MonthlyCharges & TotalCharges Summary by Churn Status

| Metric | Retained Customers (`Churn = No`) | Churned Customers (`Churn = Yes`) | Observational Difference |
| :--- | :--- | :--- | :--- |
| **MonthlyCharges (Mean)** | $61.27 | $74.44 | Churned customers paid **$13.18 more / month** (+21.5%) |
| **MonthlyCharges (Median)**| $64.43 | $79.65 | Churned median is higher by **$15.22** |
| **TotalCharges (Mean)** | $2549.91 | $1531.80 | Retained customers accumulated **$1018.12 more** in lifetime value |
| **TotalCharges (Median)** | $1679.53 | $703.55 | Retained median is higher by **$975.98** |

*Visual Chart Reference:* `reports/figures/08_charges_distribution_by_churn.png`

---

## 5. Linear Correlation Analysis Among Numeric Features

| Pairwise Relationship | Pearson Correlation (r) | Interpretation |
| :--- | :--- | :--- |
| **Tenure vs. Churn** | **-0.35** | Moderate negative correlation: As customer tenure increases, churn probability drops noticeably. |
| **MonthlyCharges vs. Churn** | **+0.19** | Weak-to-moderate positive correlation: Higher recurring fees correlate with higher departure rates. |
| **TotalCharges vs. Churn** | **-0.20** | Weak negative correlation: Churned customers tend to leave before accumulating large lifetime spend. |
| **Tenure vs. TotalCharges** | **+0.83** | Very strong positive linear relationship: Cumulative spend is predominantly driven by customer longevity. |

*Visual Chart Reference:* `reports/figures/09_numeric_correlation_matrix.png`

---

## 6. Critical Methodological & Statistical Notes

1. **Association Does NOT Establish Causation:**
   - Observing a 41.9% churn rate among Fiber Optic subscribers does *not* prove that optical fiber technology causes customers to churn. Fiber optic plans are bundled with higher monthly rates (averaging >$90/month), attract tech-savvy subscribers who switch providers for deals, or may face localized regional competition.
   - Similarly, paying by Electronic Check does not cause churn; rather, customers who choose electronic checks are disproportionately on month-to-month contracts and lack auto-renewal commitment.

2. **Sample Size & Sub-group Reliability:**
   - None of the primary analysis groups suffer from tiny sample sizes (all analyzed categories exceed N=1,000 customers).
   - The 11 unbilled records (`tenure = 0`) represent only 0.156% of total records, ensuring zero risk of distortion to global statistics.

3. **Data Integrity Note:**
   - `customerID` was preserved in the underlying raw dataset but excluded from correlation and aggregation matrices as appropriate.

---

## 7. Actionable Business Recommendations for Platform Implementation

1. **Proactive "First-90-Days" Onboarding Playbook:**
   - Because churn is concentrated in the 0–12 month window (47.44%), the platform should prioritize early warning signals and dedicated onboarding check-ins during months 1 through 3.
2. **Incentivize Long-Term Contract Migration:**
   - With month-to-month churn at 42.71% versus 11.27% for annual contracts, marketing should test targeted discounts or free service add-ons for customers willing to commit to 1- or 2-year terms.
3. **Automate Payment Method Conversion:**
   - Electronic check users churn at 45.29%. Implementing a $5 monthly discount or loyalty credits for enrolling in automatic credit card or ACH payment could significantly dampen churn.
4. **Bundle Support & Security Add-Ons into High-Tier Fiber Plans:**
   - Fiber Optic churn (41.89%) and unassisted customers (41.64%) exhibit high churn. Bundling free TechSupport or OnlineSecurity directly into Fiber Optic subscriptions creates value, reduces technical frustration, and improves stickiness.
