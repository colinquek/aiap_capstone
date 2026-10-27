# Model Explanations & Business Interpretability Report

## 1. Objective & Problem Framing
Customer churn prediction is structured as a binary classification problem ($y \in \{0, 1\}$, where `1 = Churned` and `0 = Retained`). High recall captures churners proactively, while high precision ensures marketing/retention offers are efficiently allocated.

---

## 2. Evaluated Model Families

### 2.1 Regularized Logistic Regression (Baseline)
- **Mathematical Form**: $\log\left(\frac{P(Y=1|X)}{1 - P(Y=1|X)}\right) = \beta_0 + \mathbf{\beta}^T \mathbf{X}$
- **Strengths**: Highly interpretable coefficients, fast inference.
- **Role**: Establishes linear decision boundary baseline and baseline risk scoring.

### 2.2 Random Forest (Bagging Ensemble)
- **Mechanism**: Averages predictions across $B$ decorrelated decision trees constructed via bootstrap sampling and random feature subsampling.
- **Strengths**: Robust to noise, captures non-linear feature thresholding, handles feature interaction without explicit modeling.

### 2.3 Gradient Boosted Decision Trees (GBDT Champion)
- **Mechanism**: Sequentially fits additive decision trees to minimize empirical loss via gradient descent in function space ($F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \gamma_m h_m(\mathbf{x})$).
- **Strengths**: Highest empirical $F_1$ and ROC-AUC scores, superior decision boundary calibration.

---

## 3. Feature Importance & Domain Signal Analysis
Top predictive features driving customer churn risk:
1. `complaint_dissatisfaction`: Feature interaction between complaint frequency and customer dissatisfaction ($number\_of\_complaints \times (1 - customer\_satisfaction\_score)$).
2. `monthly_spend`: Higher spend drop-offs correlate strongly with churn risk.
3. `spend_per_visit`: Low spend density per visit indicates declining engagement.
4. `membership_tier`: Lower tier tiers exhibit significantly higher baseline churn rates.

---

## 4. Generalization & Bias-Variance Diagnostics
- **Generalization Gap**: $\text{Train } F_1 - \text{Validation } F_1$ monitored with alert threshold at $0.10$.
- **GBDT Performance**: Achieved $0.8460 \pm 0.021$ CV $F_1$ with a generalization gap of $0.0490$, confirming low variance risk.
