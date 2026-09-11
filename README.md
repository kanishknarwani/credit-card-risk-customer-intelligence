# Credit Card Risk & Customer Intelligence Platform

An end-to-end credit risk analytics platform built on 30,000 customer records from the UCI Default of Credit Card Clients dataset. The project simulates the analytical workflow of a fintech credit risk team — from raw data exploration to default prediction, customer segmentation, A/B testing, data drift monitoring, and spend forecasting.

---

## Business Question

> How can a credit card company decide which customers are at risk of defaulting, how to segment them by behavior, and how to monitor portfolio health over time?

---

## Dataset

**Source:** [UCI Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

| Property | Value |
|----------|-------|
| Rows | 30,000 customers |
| Columns | 25 (raw) → 31 (after feature engineering) |
| Target | default.payment.next.month (1 = defaulted, 0 = did not) |
| Period | April 2005 – September 2005, Taiwan |
| Class Balance | 77.88% non-default / 22.12% default |

---

## Project Structure

```text
credit-card-risk-customer-intelligence/
│
├── Datasets/
│   └── UCI_Credit_Card.csv
│
├── Plots/
│   ├── [generated visualizations]
│   └── ...
│
├── project4.py
└── README.md
```

---

## Modules

### Module 1 — Exploratory Data Analysis
- Distribution analysis of credit limits, age, demographics
- Default rate by gender, education, marital status
- Correlation heatmap across all 25 features
- Probability analysis — P(Default), P(Default | Payment Delayed), P(Default | High Utilization), P(Default | Low Credit Limit)
- 95% Confidence intervals on credit limit, age, and default rate

**Key finding:** P(Default | Payment Delayed) = 50% vs 22% baseline — a 2.3x risk escalation

---

### Module 2 — Data Cleaning
- Fixed undocumented EDUCATION values (0, 5, 6) → mapped to 'Others' (4)
- Fixed undocumented MARRIAGE value (0) → mapped to 'Others' (3)
- Dropped ID column (no analytical value)
- Clipped credit utilization ratio to [0, 1] to handle negative bills and over-limit fees

---

### Module 3 — Feature Engineering
Six behavioral features engineered from raw columns:

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| credit_utilization_ratio | BILL_AMT1 / LIMIT_BAL (clipped 0–1) | What % of limit is being used |
| avg_bill_amt | Mean of BILL_AMT1–6 | Average monthly debt burden |
| avg_pay_amt | Mean of PAY_AMT1–6 | Average monthly payment made |
| repayment_ratio | avg_pay_amt / (avg_bill_amt + 1) | How much of bill is actually paid |
| max_delay | Max of PAY_0, PAY_2–6 | Worst payment delay in 6 months |
| delay_count | Count of PAY columns > 0 | How many months had any delay |

**Validation:** max_delay ranked #1 and delay_count ranked #2 in XGBoost feature importance — confirming engineering was worthwhile.

---

### Module 4 — Customer Segmentation

**PCA — Dimensionality Reduction:**
- Input: 8 features (raw + engineered)
- Output: 6 principal components
- Variance preserved: 94.7%

**KMeans Clustering:**
- Optimal K selected via Elbow Method (K=2 to K=10)
- K=4 chosen — curve showed diminishing returns beyond 4

**4 Customer Risk Profiles:**

| Cluster | Profile | Description |
|---------|---------|-------------|
| 0 | Safe Revolvers | Largest group (46%) — moderate utilization, low delays, consistent repayment |
| 1 | High Risk Defaulters | High utilization, frequent payment delays, high max_delay |
| 2 | High Spenders — Low Risk | High bill amounts, high limits, strong repayment behavior |
| 3 | Outliers | Tiny group (0.01%) — extreme values, likely anomalies |

---

### Module 5 — Credit Default Prediction

**Train-Test Split:** 80/20 stratified split (24,000 train / 6,000 test)

**Class Imbalance Handling:** SMOTE applied on training data only
- Before SMOTE: 18,691 non-default / 5,309 default
- After SMOTE: 18,691 non-default / 18,691 default (balanced)

**Cross Validation:** Stratified K-Fold

**Models compared:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.7352 | 0.4296 | **0.6029** | 0.5017 | 0.7339 |
| Random Forest | **0.7745** | **0.4899** | 0.4748 | 0.4822 | 0.7419 |
| XGBoost | 0.7648 | 0.4698 | 0.4928 | 0.4811 | **0.7430** |

**Model chosen for production recommendation:** Logistic Regression — highest Recall (60%) minimizes missed defaulters which is the most costly error in credit risk.

**Why Recall over Accuracy:** Missing an actual defaulter (false negative) costs the bank the full loan amount. A false positive just triggers extra manual review. With 78:22 class imbalance, a model predicting "no default" for everyone achieves 78% accuracy while catching zero defaulters — making accuracy a misleading metric.

---

### Module 6 — Feature Importance

Top 5 features by XGBoost importance:

| Rank | Feature | Importance Score |
|------|---------|-----------------|
| 1 | max_delay | ~0.33 |
| 2 | delay_count | ~0.11 |
| 3 | MARRIAGE | ~0.08 |
| 4 | PAY_2 | ~0.07 |
| 5 | SEX | ~0.06 |

**Key insight:** Both top features are engineered — validating that behavioral pattern features outperform raw demographic features for credit default prediction.

---

### Module 7 — A/B Testing

**Business question:** Does a 10% cashback offer drive higher spending than a 5% offer?

| Group | Size | Mean Spend |
|-------|------|------------|
| Control (5% cashback) | 14,990 | 51,565 NT$ |
| Treatment (10% cashback) | 15,010 | 50,882 NT$ |

**Test:** Welch's two-sample t-test (equal_var=False)

**Results:**
- T-statistic: 0.8037
- P-value: 0.4216
- 95% CI on difference: (−2,349 to +983) — contains zero
- **Conclusion: Fail to reject H0 — no statistically significant difference**

**Note:** This is a simulated experiment on historical data. In production, real A/B test data with actual experiment logs would be required. A null result here is a legitimate finding — doubling the cashback offer does not significantly move spending in this dataset.

---

### Module 8 — Data Drift Detection (PSI)

**Setup:** First 20,000 rows as "2022 training population", last 10,000 as "2025 current population"

**Population Stability Index results:**

| Feature | PSI | Status |
|---------|-----|--------|
| LIMIT_BAL | 0.0111 | Stable |
| AGE | 0.0067 | Stable |
| credit_utilization_ratio | 0.0046 | Stable |
| max_delay | 0.0332 | Stable |
| delay_count | 0.0111 | Stable |
| repayment_ratio | 0.0014 | Stable |

**PSI thresholds:**
- PSI < 0.10 → Stable (no action)
- PSI 0.10–0.25 → Monitor (watch closely)
- PSI > 0.25 → Retrain (model retraining required)

All features stable — expected since both populations are drawn from the same homogeneous 2005 dataset. In production, real drift would be detected when scoring 2025 customers with a model trained on 2022 data.

---

### Module 9 — Time Series Forecasting

**Setup:** Reshaped 6 monthly bill columns into a portfolio-level time series

**Monthly average spend:**

| Month | Avg Spend (NT$) | Growth Rate |
|-------|----------------|-------------|
| April | 38,871 | — |
| May | 40,311 | +3.70% |
| June | 43,262 | +7.32% |
| July | 47,013 | +8.67% |
| August | 49,179 | +4.61% |
| September | 51,223 | +4.16% |

**Features created:** 2-month MA, 3-month MA, Lag-1, Lag-2, monthly growth rate

**October 2005 Forecast:**

| Method | Forecast |
|--------|---------|
| 3-Month Moving Average | 49,138 |
| Linear Trend | 54,188 |
| Growth Rate Extrapolation | 54,138 |
| **Ensemble Average** | **52,488** |

**Portfolio trend:** 31.8% spend growth from April to September 2005

---

## Key Numbers

```
Dataset         : 30,000 customers, 25 raw columns
Features        : 17 model features (11 raw + 6 engineered)
PCA             : 8 → 6 components, 94.73% variance preserved
Clusters        : 4 customer risk profiles
ROC-AUC         : 0.743 (XGBoost)
Recall          : 60% (Logistic Regression — chosen model)
P(Default)      : 22.12% baseline
P(Default|Delay): 50% — 2.3x risk escalation
PSI             : All features < 0.10 (stable)
Oct Forecast    : 52,488 NT dollars
A/B p-value     : 0.42 (not significant)
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost |
| Imbalance | imbalanced-learn (SMOTE) |
| Statistics | SciPy |
| Dimensionality | Scikit-learn PCA |
| Clustering | Scikit-learn KMeans |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/kanishknarwani/credit-card-risk-customer-intelligence.git

# Navigate into the project
cd credit-card-risk-customer-intelligence

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn scipy

# Download dataset
# Place UCI_Credit_Card.csv in the Datasets/ folder
# Download from: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

# Run the analysis
python project4.py
```

---

## Limitations

- Dataset is from Taiwan 2005 — behavioral patterns may not translate directly to modern Indian credit markets
- Customer-level aggregated data, not transaction-level — intra-month behavioral signals are lost
- No reject inference — dataset contains only approved customers, so the model is blind to customers who were never given credit (known bias in credit risk modeling)
- A/B test is simulated on historical data — not a real randomized controlled trial
- Time series has only 6 data points — insufficient for complex forecasting models like Prophet or ARIMA

---

## Author

**Kanishk**
B.Com | Masters in Economics
Gokhale Institute of Politics and Economics, Pune

[GitHub](https://github.com/kanishknarwani) | [LinkedIn](https://linkedin.com/in/kanishk-narwani)
