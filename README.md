# Customer Churn Prediction — Exploratory Data Analysis & Feature Insights

## Executive Summary & Dataset Overview
This document summarizes the **Exploratory Data Analysis (EDA)** phase for predicting customer churn using `synthetic_data_204.csv`. The business goal is to identify churn-prone customers to optimize proactive retention campaigns.

* **Dataset Dimensions:** 1,000 rows, 6 predictor attributes, 1 target variable (`customer_churn`).
* **Class Balance:** 850 `Retained` (85.0%) vs. 150 `Churned` (15.0%).
* **Missing Values & Duplicates:** 0 missing values across all columns; 0 duplicate records.

---

## Key Analytical Insights & Statistical Hypotheses

### 1. Customer Satisfaction Score ($p = 7.11 	imes 10^{-31}$)
* **Finding:** Churning customers exhibit a statistically significantly higher median satisfaction score ($0.908$) compared to retained customers ($0.772$).
* **Business Interpretation:** Indicates a potential artifact in survey collection or high expectation levels among dissatisfied churners.

### 2. Customer Complaints ($p = 2.96 	imes 10^{-23}$)
* **Finding:** Retained customers log higher average complaints ($5.27$) than churned customers ($2.75$).
* **Business Interpretation:** Active customers communicate complaints through support channels, whereas churned customers leave silently without lodging formal complaints.

### 3. Monthly Spend ($p = 7.15 	imes 10^{-12}$)
* **Finding:** Churning customers have a higher median spend ($72.21$) than retained customers ($52.04$). High-value customers are at higher risk of switching providers.

### 4. Null Finding: Discount Usage Rate ($p = 0.7527$)
* **Finding:** Discount usage rate shows no statistically significant difference between retained ($0.278$) and churned ($0.284$) customers ($p = 0.7527$).
* **Pre-selection Action:** `discount_usage_rate` offers low standalone discriminative power for linear baselines.

---

## Statistical Testing & Multicollinearity Summary

| Feature | Data Type | Subgroup Comparison (Retained vs Churned) | Statistical Test | $p$-value | Significance ($alpha=0.05$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`customer_satisfaction_score`** | Continuous | Median: $0.772$ vs $0.908$ | Mann-Whitney U | $7.11 	imes 10^{-31}$ | **Statistically Significant** |
| **`number_of_complaints`** | Continuous | Mean: $5.27$ vs $2.75$ | Mann-Whitney U | $2.96 	imes 10^{-23}$ | **Statistically Significant** |
| **`monthly_spend`** | Continuous | Median: $\$52.04$ vs $\$72.21$ | Mann-Whitney U | $7.15 	imes 10^{-12}$ | **Statistically Significant** |
| **`membership_tier`** | Categorical | Churn Rate: Basic ($19.4\%$), Premium ($12.8\%$), Pro ($4.8\%$) | Chi-Square Contingency | $2.62 	imes 10^{-05}$ | **Statistically Significant** |
| **`number_of_store_visits`** | Continuous | Mean: $14.57$ vs $16.97$ | Mann-Whitney U | $1.92 	imes 10^{-03}$ | **Statistically Significant** |
| **`discount_usage_rate`** | Continuous | Mean: $0.278$ vs $0.284$ | Mann-Whitney U | $0.7527$ | **Not Significant** |

### Multicollinearity Analysis (VIF)
* All Variance Inflation Factor (VIF) values remain under $7.3$ (`customer_satisfaction_score`: $7.30$, `monthly_spend`: $5.87$, `number_of_store_visits_per_month`: $4.58$, `discount_usage_rate`: $4.59$, `number_of_complaints`: $3.01$). Multicollinearity is within acceptable modeling bounds.

---

## Preprocessing & Feature Engineering Strategy
1. **Derived Feature Construction:** `complaint_to_visit_ratio` ($p = 2.51 	imes 10^{-21}$) engineered as a high-impact interaction predictor.
2. **Stratified Splitting:** Train/Val/Test splitting must preserve the $15.0\%$ churn ratio (`stratify=y`).
3. **Scaling & Encoding:** Apply `StandardScaler` to continuous predictors and `OneHotEncoder(handle_unknown='ignore')` to `membership_tier`.
