# Sprint 1: Literature Review — Customer Churn Classification

## Overview & Objective

The objective of Sprint 1 is to conduct a comprehensive literature review identifying best practices, methodologies, model architectures, feature engineering techniques, class imbalance strategies, and evaluation metrics specific to binary classification machine learning problems. 

This document serves as the foundational literature review and methodological baseline for predicting customer churn (`customer_churn`) within a production-ready machine learning framework.

---

## 1. Classification Model Architectures for Tabular Data

In tabular machine learning benchmarks, decision tree ensembles consistently demonstrate superior inductive bias over deep neural networks and linear baselines. Four candidate model families are evaluated for this capstone:

### 1.1 Linear & Parametric Baselines: Logistic Regression
Logistic Regression models the log-odds of the positive outcome as a linear combination of input features:

$$\log\left(\frac{P(Y=1|X)}{1 - P(Y=1|X)}\right) = \beta_0 + \mathbf{\beta}^T \mathbf{X}$$

* **Strengths**: Highly interpretable, computationally lightweight, serves as an essential baseline for pipeline validation.
* **Limitations**: Assumes linear decision boundaries and independent predictors (susceptible to multicollinearity). Requires feature scaling and encoding.
* **Regularization**: $L_1$ (Lasso) for explicit feature selection, $L_2$ (Ridge) for shrinkage under multicollinearity, or ElasticNet ($\alpha L_1 + (1-\alpha) L_2$).

### 1.2 Distance-Based Non-Parametric Models: $K$-Nearest Neighbors ($K$-NN)
$K$-NN classifies an instance based on the majority vote of its $k$ nearest neighbors in feature space using distance metrics such as Euclidean or Manhattan distance:

$$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

* **Strengths**: Non-parametric, capable of capturing complex non-linear decision boundaries without assuming feature distributions.
* **Limitations**: Highly sensitive to feature scales (requires standardization/normalization), computationally expensive during inference $\mathcal{O}(N \cdot P)$, and vulnerable to the curse of dimensionality.

### 1.3 Tree-Based Interpretability: Decision Trees
Decision Trees recursively partition feature space into orthogonal hyperplanes by maximizing information gain or minimizing impurity metrics such as Gini Impurity:

$$I_G(p) = 1 - \sum_{i=1}^C p_i^2$$

* **Strengths**: Scale-invariant to monotonic transformations, natural handling of mixed feature types, high interpretability via decision paths.
* **Limitations**: Prone to severe overfitting and variance; sensitive to minor perturbations in training data without tree pruning (`max_depth`, `min_samples_leaf`).

### 1.4 Tree Ensembles & Gradient Boosting (Random Forest, XGBoost, LightGBM)
* **Random Forest (Bagging)**: Reduces variance by averaging predictions across $B$ decorrelated decision trees built on bootstrap samples with random feature subsampling.
* **Gradient Boosted Decision Trees (GBDT)**: Sequentially builds additive decision trees to minimize a differentiable loss function via gradient descent in function space:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \gamma_m h_m(\mathbf{x})$$

* **Empirical Justification**: Ensemble methods effectively capture higher-order feature interactions, accommodate unscaled/mixed features, and consistently achieve state-of-the-art ROC-AUC and PR-AUC scores on imbalanced tabular benchmarks.

---

## 2. Feature Engineering & Preprocessing Best Practices

All data transformations must be fitted exclusively on training folds and subsequently applied to validation/test sets to prevent data leakage.

### 2.1 Missing Data Imputation Strategy
Data missingness must be analyzed prior to cleaning to distinguish between Missing Completely at Random (MCAR), Missing at Random (MAR), and Missing Not at Random (MNAR).
* **Numerical Imputation**: Median imputation is preferred over mean imputation due to robustness against outliers. Subgroup-level median imputation (e.g., median spend conditional on `membership_tier`) retains group distribution structures.
* **Categorical Imputation**: Frequent category (mode) imputation or explicit category assignment (e.g., `"Missing"`) to preserve missingness indicators.

### 2.2 Feature Scaling & Encoding
* **Numerical Scaling**: Distance-based algorithms ($K$-NN) and regularized linear models (Logistic Regression) require zero-mean, unit-variance standardization:

$$z = \frac{x - \mu}{\sigma}$$

Robust Scaling (using median and Interquartile Range) is deployed when continuous variables present extreme outliers.
* **Categorical Encoding**:
  * **One-Hot Encoding**: Applied to nominal features without natural order (`bag_color`, `mode_of_transport`). Dummy encoding ($k-1$ categories) is applied for unregularized linear models to prevent exact multicollinearity.
  * **Ordinal Encoding**: Applied to features with ordinal hierarchy (`membership_tier`: Basic $\rightarrow$ Pro $\rightarrow$ Premium).

### 2.3 Interaction Features & Domain Engineering
Non-linear feature interactions significantly boost predictive signal:
* **Composite Ratios**: Ratio metrics such as $\frac{\text{Monthly Spend}}{\text{Store Visits}}$ or interaction terms between complaints and satisfaction scores ($\text{Complaints} \times (1 - \text{Satisfaction})$).
* **Binned Aggregations**: Discretization of continuous variables into quantile bins to capture step-function risk behaviors.

---

## 3. Class Imbalance Mitigation Methodologies

Imbalanced class distributions bias classifier decision boundaries toward the majority class. Three primary levels of mitigation are reviewed:

### 3.1 Algorithmic Cost-Sensitive Learning
Modifies model loss functions to penalize misclassifications of the minority class proportionally to class frequencies:

$$w_1 = \frac{N_{\text{total}}}{2 \cdot N_{\text{minority}}}$$

* **Implementation**: Setting `class_weight='balanced'` in Logistic Regression, Decision Trees, and Random Forest models. This approach requires no synthetic data generation and avoids altering true data distributions.

### 3.2 Data-Level Resampling Techniques
* **Random Undersampling (RUS)**: Downsamples the majority class. Reduces compute time but risks discarding informative samples.
* **Synthetic Minority Over-sampling Technique (SMOTE)**: Generates synthetic minority instances along line segments joining $k$-nearest neighbors in feature space:

$$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_{zi} - \mathbf{x}_i), \quad \lambda \sim U(0,1)$$

* **Combined Resampling (SMOTEENN / SMOTETomek)**: Applies SMOTE followed by cleaning techniques (Edited Nearest Neighbors or Tomek Links) to remove overlapping noisy boundary instances.
* **Methodological Precaution**: Resampling **must only** be performed within cross-validation training folds. Resampling test or validation sets causes catastrophic data leakage.

---

## 4. Evaluation Framework & Metrics Selection

Relying solely on Accuracy in imbalanced classification tasks induces the "Accuracy Paradox," where a naive majority-class classifier achieves high accuracy while failing to identify any churners.

| Metric | Mathematical Definition | Primary Utility & Context |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Baseline overview; unreliable under class imbalance. |
| **Precision** | $\frac{TP}{TP + FP}$ | Measures false positive rate; critical when retention interventions are costly. |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Measures false negative rate; critical when losing a customer incurs high opportunity cost. |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean balancing Precision and Recall at a fixed 0.5 threshold. |
| **ROC-AUC** | $\int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt$ | Measures ranking capability across all classification thresholds; invariant to class distribution changes. |
| **PR-AUC (Average Precision)** | Area under Precision-Recall curve | Preferred metric for heavily imbalanced datasets; focuses explicitly on minority class performance. |

### 4.1 Decision Threshold Optimization
Standard prediction defaults to a probability threshold of $\tau = 0.5$. The optimal threshold $\tau^*$ is determined empirically on validation folds by maximizing expected financial utility or maximizing $F_\beta$-score ($\beta > 1$ prioritizing Recall):

$$\tau^* = \arg\max_\tau \left( \text{Gain}(TP(\tau)) - \text{Cost}(FP(\tau)) \right)$$

---

## 5. Software Engineering Best Practices for ML Pipelines

To ensure reproducibility, maintainability, and production readiness, the pipeline implementation adheres to modular software architecture patterns:

1. **Config-Driven Architecture**: Hyperparameters, feature lists, file paths, and seeds are externalized into `src/config.yaml`.
2. **Global Random Seed Enforcement**: Deterministic reproducibility across modules (`random_state: 42`).
3. **Data Schema Safeguards**: Automated validation of input columns, types, and value ranges post-ingestion.
4. **Structured Logging**: Terminal output formatted using explicit ANSI color constants (`GREEN_BOLD = "\033[1;32m"`, `RED_BOLD = "\033[1;31m"`, `CLEAR = "\033[0m"`).
5. **Artifact Persistence**: Models, preprocessors, evaluation reports, and feature lists are serialized to disk (`.joblib` / `.json`).
6. **Batch Prediction Interface**: Dedicated entry point capable of loading persisted pipelines to score unseen inference records.