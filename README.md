# Customer Churn Classification Pipeline: End-to-End Machine Learning Framework

## 1. Executive Summary & Problem Framing

This project implements a production-ready, modular machine learning pipeline to predict customer churn (`customer_churn`). Customer churn prediction is framed as a **binary classification problem** under supervised learning, where behavioral and transactional customer features are mapped to a binary target ($y \in \{0, 1\}$, where `1 = Churned` and `0 = Retained`).

Predicting churn accurately enables proactive customer retention strategies, maximizing Customer Lifetime Value (CLV) and minimizing acquisition costs. However, tabular classification tasks in domains like retail, telecommunications, and finance present specific statistical and engineering challenges:
1. **Class Imbalance**: Churners represent 15.0% of the dataset (Imbalance Ratio = $5.67:1$). Standard metrics like Accuracy are misleading ("Accuracy Paradox"); evaluation focuses on **PR-AUC**, **$F_1$-score**, **Precision**, and **Recall**.
2. **Heterogeneous Predictors**: Continuous behavioral metrics (`monthly_spend`, `discount_usage_rate`), discrete counts (`number_of_store_visits_per_month`, `number_of_complaints`), and categorical tiers (`membership_tier`).
3. **Non-linear Domain Interactions**: Feature interaction between complaints and customer dissatisfaction (`complaint_dissatisfaction = number_of_complaints * (1 - customer_satisfaction_score)`) yields strong predictive separation ($p = 6.91 \times 10^{-32}$).

---

## 2. Comprehensive Literature Review

### 2.1 Classification Model Architectures for Tabular Data
In tabular machine learning benchmarks (e.g., Grinsztajn et al., 2022, *NeurIPS*), decision tree ensembles consistently demonstrate superior inductive bias over deep neural networks and linear baselines. Four candidate model families are evaluated for this capstone:

#### 1. Linear & Parametric Baselines: Logistic Regression
Logistic Regression models the log-odds of the positive outcome as a linear combination of input features:

$$\log\left(\frac{P(Y=1\vert{}X)}{1 - P(Y=1\vert{}X)}\right) = \beta_0 + \mathbf{\beta}^T \mathbf{X}$$

* **Strengths**: Highly interpretable, computationally lightweight, serves as an essential baseline for pipeline validation.
* **Limitations**: Assumes linear decision boundaries and independent predictors (susceptible to multicollinearity). Requires feature scaling and encoding.
* **Regularization**: $L_1$ (Lasso) for explicit feature selection, $L_2$ (Ridge) for shrinkage under multicollinearity, or ElasticNet ($\alpha L_1 + (1-\alpha) L_2$).

#### 2. Distance-Based Non-Parametric Models: $K$-Nearest Neighbors ($K$-NN)
$K$-NN classifies an instance based on the majority vote of its $k$ nearest neighbors in feature space using distance metrics such as Euclidean or Manhattan distance:

$$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

* **Strengths**: Non-parametric, capable of capturing complex non-linear decision boundaries without assuming feature distributions.
* **Limitations**: Highly sensitive to feature scales (requires standardization/normalization), computationally expensive during inference $\mathcal{O}(N \cdot P)$, and vulnerable to the curse of dimensionality.

#### 3. Tree-Based Interpretability: Decision Trees
Decision Trees recursively partition feature space into orthogonal hyperplanes by maximizing information gain or minimizing impurity metrics such as Gini Impurity:

$$I_G(p) = 1 - \sum_{i=1}^C p_i^2$$

* **Strengths**: Scale-invariant to monotonic transformations, natural handling of mixed feature types, high interpretability via decision paths.
* **Limitations**: Prone to severe overfitting and variance; sensitive to minor perturbations in training data without tree pruning (`max_depth`, `min_samples_leaf`).

#### 4. Tree Ensembles & Gradient Boosting (Random Forest, XGBoost, LightGBM)
* **Random Forest (Bagging)**: Reduces variance by averaging predictions across $B$ decorrelated decision trees built on bootstrap samples with random feature subsampling.
* **Gradient Boosted Decision Trees (GBDT)**: Sequentially builds additive decision trees to minimize a differentiable loss function via gradient descent in function space:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \gamma_m h_m(\mathbf{x})$$

* **Empirical Justification**: Ensemble methods effectively capture higher-order feature interactions, accommodate unscaled/mixed features, and consistently achieve state-of-the-art ROC-AUC and PR-AUC scores on imbalanced tabular benchmarks.

---

### 2.2 Feature Engineering & Preprocessing Best Practices
All data transformations must be fitted exclusively on training folds and subsequently applied to validation/test sets to prevent data leakage.
* **Missing Data Imputation Strategy**: Analyze missingness prior to cleaning (MCAR vs. MAR vs. MNAR). Subgroup-level median imputation retains distribution structures.
* **Numerical Scaling**: Robust Scaling (using median and Interquartile Range) is deployed when continuous variables present extreme outliers (`monthly_spend`).
* **Categorical Encoding**: Ordinal encoding for hierarchical features (`membership_tier`: Basic $\rightarrow$ Pro $\rightarrow$ Premium). One-hot encoding for nominal categories.
* **Domain Feature Interactions**: Composite ratios and multiplicative terms (e.g., `spend_per_visit`, `complaint_dissatisfaction`).

---

### 2.3 Class Imbalance Mitigation Methodologies
Imbalanced class distributions bias classifier decision boundaries toward the majority class:
1. **Algorithmic Cost-Sensitive Learning**: Modifies loss functions to penalize misclassifications of the minority class proportionally to class frequencies (`class_weight='balanced'`).
2. **Data-Level Resampling Techniques**: Random Undersampling (RUS), Synthetic Minority Over-sampling Technique (SMOTE), and combined resampling (SMOTEENN / SMOTETomek). Resampling must only be performed within cross-validation training folds.

---

### 2.4 Comprehensive Evaluation Framework & Metrics Selection

| Metric | Mathematical Definition | Primary Utility & Context |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Baseline overview; unreliable under class imbalance. |
| **Precision** | $\frac{TP}{TP + FP}$ | Measures false positive rate; critical when retention interventions are costly. |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Measures false negative rate; critical when losing a customer incurs high opportunity cost. |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean balancing Precision and Recall at a fixed 0.5 threshold. |
| **ROC-AUC** | $\int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt$ | Measures ranking capability across all classification thresholds. |
| **PR-AUC (Average Precision)** | Area under Precision-Recall curve | Preferred metric for heavily imbalanced datasets; focuses explicitly on minority class performance. |

#### Decision Threshold Optimization
Optimal threshold $\tau^*$ is determined empirically on validation folds by maximizing expected financial utility or maximizing $F_\beta$-score ($\beta > 1$ prioritizing Recall):

$$\tau^* = \arg\max_\tau \left( \text{Gain}(TP(\tau)) - \text{Cost}(FP(\tau)) \right)$$

---

## 3. Project Architecture & Folder Structure

The project follows the production modular design pattern specified in Chapter 1.6 & 2.4 of the AIAP Foundation Learning Companion:

```text
your-repo-name/
│-- README.md                      # Comprehensive project documentation & literature review
│-- eda.ipynb                      # Interactive exploratory data analysis notebook
│-- main.py                        # Pipeline orchestration entry point
│-- requirements.txt               # Pinned dependencies for environment reproducibility
│-- data/
│   └── synthetic_data_204.csv     # Raw customer churn dataset
└── src/
    │-- __init__.py
    │-- config.yaml                # Centralized parameters, seeds, paths, and feature grids
    └── preprocessing.py           # Preprocessing pipeline, schema safeguards, & transformers
```

## Sprint 4: Baseline Model Development

### Benchmarks
- **Model Type**: Logistic Regression (L2 Penalty)
- **Evaluation Metrics**:

| Split | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 0.82 | 0.78 | 0.75 | 0.76 | 0.86 |
| **Test** | 0.80 | 0.76 | 0.73 | 0.74 | 0.84 |

### Usage
Run the baseline model training & evaluation pipeline:
```bash
python main.py src/config.yaml