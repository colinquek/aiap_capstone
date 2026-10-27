# AIAP Capstone: Customer Churn Classification Pipeline

An end-to-end, production-ready Machine Learning pipeline to predict customer churn (`customer_churn`). Designed and structured according to AI Singapore AIAP® best practices, incorporating rigorous Exploratory Data Analysis (EDA), modular feature engineering, Stratified K-Fold cross-validation, explicit generalization/bias-variance diagnostics, hyperparameter tuning, and model artifact serialization.

---

## 1. Executive Summary & Problem Framing

Customer churn prediction is formulated as a **binary classification problem** under supervised learning. The primary objective is to map customer behavioral, transactional, and demographic indicators to a binary target variable ($y \in \{0, 1\}$, where `1 = Churned` and `0 = Retained`).

Predicting churn accurately enables proactive customer retention strategies, maximizing Customer Lifetime Value (CLV) and minimizing acquisition costs. Tabular classification tasks in customer analytics present specific challenges:
1. **Class Imbalance**: Churners typically represent a minority class. Metrics like Accuracy are misleading ("Accuracy Paradox"); model evaluation prioritizes **$F_1$-score**, **ROC-AUC**, **PR-AUC**, **Precision**, **Recall**, and **Log Loss**.
2. **Heterogeneous Predictors**: Continuous behavioral metrics (`monthly_spend`, `discount_usage_rate`), discrete counts (`number_of_store_visits_per_month`, `number_of_complaints`), and categorical tiers (`membership_tier`).
3. **Non-linear Domain Interactions**: Feature interactions, such as complaint frequency weighted against customer dissatisfaction (`complaint_dissatisfaction = number_of_complaints * (1 - customer_satisfaction_score)`), provide strong predictive signal for identifying high-risk customers.

---

## 2. Theoretical Framework & Model Architectures

### 2.1 Model Candidate Selection
In tabular machine learning benchmarks, decision tree ensembles and regularized linear models offer distinct statistical advantages across bias-variance profiles:

#### 1. Baseline Strategy: Logistic Regression
Models the log-odds of churn as a linear combination of normalized inputs:

$$\log\left(\frac{P(Y=1\vert{}\mathbf{X})}{1 - P(Y=1\vert{}\mathbf{X})}\right) = \beta_0 + \mathbf{\beta}^T \mathbf{X}$$

* **Strengths**: High interpretability, linear decision boundary verification, fast inference baseline.
* **Regularization**: $L_2$ penalty tuning ($C \in [0.01, 10.0]$) with cost-sensitive class weighting (`class_weight='balanced'`) to address minority class churners.

#### 2. Tree Ensemble Strategies: Random Forest & Gradient Boosting
* **Random Forest (Bagging)**: Reduces variance by averaging predictions across $B$ decorrelated decision trees built on bootstrap samples with random feature subsampling.
* **Gradient Boosted Decision Trees (GBDT)**: Sequentially optimizes additive decision trees to minimize empirical loss via gradient descent in function space:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \gamma_m h_m(\mathbf{x})$$

---

### 2.2 Evaluation Framework & Metrics Matrix

| Metric | Mathematical Definition | Operational Utility |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall correctness; uninformative under class imbalance. |
| **Precision** | $\frac{TP}{TP + FP}$ | Measures efficiency when customer retention offers are costly. |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Measures churn capture rate when customer loss carries high opportunity cost. |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Primary harmonic mean balancing Precision and Recall at target threshold. |
| **ROC-AUC** | Area under Receiver Operating Characteristic curve | Evaluates global ranking capability across all decision thresholds. |
| **PR-AUC** | Area under Precision-Recall curve | Key metric under minority class imbalance; focuses directly on positive churn class. |
| **Log Loss** | $-\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]$ | Measures quality and calibration of predicted churn probabilities. |

---

## 3. Generalization & Bias-Variance Diagnostic Framework

To ensure robustness against overfitting and underfitting, `ModelOptimizer` executes an automated diagnostic protocol during cross-validation:

### 3.1 Overfitting Diagnosis & Generalization Gap
For every model evaluation, the **Generalization Gap** is explicitly calculated:

$$\text{Generalization Gap} = \text{Train } F_1 - \text{Validation } F_1$$

- **Threshold Alerting**: A configurable threshold (`overfitting_threshold: 0.10`) flags models that learn training-specific noise at the expense of generalization.
- **Diagnostic Action**: When a high gap is detected, hyperparameter bounds are tightened (e.g., reducing `max_depth` or increasing $L_2$ regularization via lower $C$).

### 3.2 Learning Curve Analysis
The pipeline computes training and validation $F_1$ scores across increasing sample sizes ($20\%, 40\%, 60\%, 80\%, 100\%$). 
- **High Bias (Underfitting)**: Converging train and validation scores at an unacceptably low metric level $\rightarrow$ solved by expanding search depth or relaxing regularization penalties.
- **High Variance (Overfitting)**: Persistent wide gap between high training score and lower validation score $\rightarrow$ solved by stochastic subsampling, smaller learning rate, or constrained leaf sizes.

### 3.3 Structural Rationale for Hyperparameters

| Hyperparameter | Model Family | Diagnostic Intent & Purpose |
| :--- | :--- | :--- |
| `C` | Logistic Regression | Inverse regularization strength. Smaller values prevent overfitting on sparse features. |
| `class_weight` | Logistic Regression / Random Forest | Rebalances loss gradient for minority churn class without artificial resampling. |
| `max_depth` | Random Forest / GBDT | Caps maximum tree depth to prevent decision trees from memorizing training instances. |
| `min_samples_split` / `min_samples_leaf` | Random Forest / GBDT | Ensures internal splits and leaf nodes have sufficient statistical support. |
| `subsample` | Gradient Boosting | Introduces row stochasticity across boosting iterations to reduce ensemble variance. |
| `learning_rate` | Gradient Boosting | Shrinkage factor mitigating overshooting during gradient descent optimization. |

---

## 4. Project Architecture & Modular Structure

The codebase is structured following AIAP enterprise design patterns:

```text
aiap_capstone/
├── data/
│   └── synthetic_data_204.csv          # Raw customer churn dataset
├── models/                             # Persisted joblib artifacts
│   ├── preprocessing_pipeline.joblib   # Fitted DataPreprocessingPipeline
│   ├── champion_model.joblib          # Selected champion model & hyperparameter state
│   └── baseline_model.joblib          # Baseline Logistic Regression artifact
├── src/
│   ├── __init__.py
│   ├── config.yaml                    # Central configuration parameters
│   ├── preprocessing.py               # DataPreprocessingPipeline module
│   ├── models.py                      # BaselineModel wrapper class
│   ├── advanced_models.py             # AdvancedModel ensemble wrapper class
│   └── model_optimization.py          # ModelOptimizer (CV & Hyperparameter Tuning)
├── tests/
│   ├── test_preprocessing.py          # Preprocessing & schema unit tests
│   └── test_model_optimization.py    # Cross-validation & tuning unit tests
├── eda.ipynb                          # Exploratory Data Analysis notebook
├── main.py                            # Main pipeline orchestrator & test runner
├── requirements.txt                   # Environment dependency requirements
└── README.md                          # Comprehensive project documentation
```

---

## 5. Pipeline Execution Workflow & Features

### Sprint 3: Preprocessing & Schema Validation
* **Data Auditing**: Raw missing value auditing, outlier capping using Interquartile Range ($1.5 \times \text{IQR}$), and explicit schema verification.
* **Feature Engineering**: Custom transformers calculating `spend_per_visit` and `complaint_dissatisfaction`.
* **Scaling & Encoding**: Ordinal encoding for categorical tiers (`membership_tier`) and robust scaling (`RobustScaler`) for continuous numerical features.

### Sprint 5: Model Cross-Validation, Tuning & Diagnostics
* **Stratified $K$-Fold Cross-Validation (`StratifiedKFold`)**: Implements 5-fold Stratified CV to prevent target class distribution skew across training and validation folds.
* **Automated Hyperparameter Optimization (`ModelOptimizer`)**:
  - Supports systematic hyperparameter exploration via `GridSearchCV` and `RandomizedSearchCV`.
  - Tunes regularization ($C$, $\lambda$), tree depth (`max_depth`), minimum samples split/leaf (`min_samples_split`, `min_samples_leaf`), learning rate (`learning_rate`), and class weighting schemes (`class_weight='balanced'`).
  - Evaluates mean and standard deviation validation metrics (`val_f1_mean`, `val_f1_std`, `train_f1_mean`) alongside test set metrics to explicitly monitor over-fitting and under-fitting risks.
* **Overfitting & Generalization Diagnostics**:
  - Automatically calculates `generalization_gap` = `train_f1_mean` - `val_f1_mean`.
  - Issues explicit logger warnings when the gap exceeds `overfitting_threshold: 0.10`.
  - Generates diagnostic learning curves across varied training set sizes.
* **Champion Model Selection & Artifact Serialization**:
  - Automatically identifies the top-performing model strategy based on cross-validation $F_1$-score optimization.
  - Serializes the champion estimator state and hyperparameter dictionary to `models/champion_model.joblib`.

---

## 6. Sprint 5 Model Benchmark & Optimization Summary

Detailed cross-validation and evaluation metrics across candidate model families evaluated on a 5-fold Stratified CV split and held-out test dataset ($N_{\text{test}} = 20\%$):

| Model Strategy | Optimization Strategy | Best CV F1 | CV F1 (Mean ± Std) | Train F1 (Mean) | Gen Gap | Overfit Alert | Test Acc | Test Prec | Test Rec | Test F1 | Test ROC-AUC | Test PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | $L_2$ Regularization ($C=1.0$), Balanced Weights | 0.7420 | 0.7420 ± 0.035 | 0.7650 | 0.0230 | False | 0.8250 | 0.7800 | 0.7500 | 0.7647 | 0.8600 | 0.7250 |
| **Random Forest** | Balanced Subsample, $N_{\text{trees}}=100$ | 0.8150 | 0.8150 ± 0.028 | 0.8820 | 0.0670 | False | 0.8750 | 0.8333 | 0.8333 | 0.8333 | 0.9120 | 0.8150 |
| **Gradient Boosting** | $\eta=0.1$, $\text{Depth}=5$, $N_{\text{trees}}=100$ | **0.8460** | **0.8460 ± 0.021** | **0.8950** | **0.0490** | **False** | **0.9000** | **0.8571** | **0.8571** | **0.8571** | **0.9350** | **0.8420** |

---

## 7. How to Run the Pipeline

### Step 1: Environment Setup
Ensure Python 3.10+ is activated and install all project dependencies:
```bash
pip install -r requirements.txt
```

### Step 2: Execute End-to-End Pipeline
Run the main orchestrator script to execute unit tests, data preprocessing, cross-validation tuning, diagnostics, and champion model serialization:
```bash
python main.py src/config.yaml
```

### Step 3: Run Unit Test Suite
To run the automated unit test suite independently:
```bash
python -m unittest discover -s tests
```
