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
│   ├── feature_engineering.py         # Custom transformers & FeatureSelector
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
* **Feature Engineering (after imputation)**: Raw numerical source columns are first imputed and IQR-capped by a dedicated `source_cleaner` stage, and only then do the custom transformers derive 7 domain features (see `src/feature_engineering.py` below) — the reordered flow ensures derived features are computed from clean values, and every stage is fitted on the training split only.
* **Skew Transformation**: `PowerTransformer` (Yeo-Johnson) applied to numerical features when `transformation.skew: "power"` is set in `config.yaml` — active in the current configuration (`"power"`); set `"none"` to disable.
* **Scaling & Encoding**: Robust scaling (`RobustScaler`) for continuous numerical features; one-hot encoding for categorical tiers (`membership_tier`) — tier churn is non-monotonic (Basic 0.194 > Premium 0.131 > Pro 0.047), so ordinal encoding would impose a false ordering.
* **Feature Selection**: Seeded `SelectKBest` wrapper (see below) first removes near-duplicate features via a Pearson-correlation redundancy filter (`redundancy_threshold: 0.90`), then ranks the full engineered feature set and keeps the top `feature_selection.k_best` (currently `5`).

#### `src/feature_engineering.py` — Custom Transformers & Feature Selection
Seven stateless domain-feature transformers (`requires_fit=False`), each grounded in observed churn statistics from the EDA:
* **`SpendPerVisitTransformer`**: `spend_per_visit = monthly_spend / (visits + ε)`.
* **`ComplaintDissatisfactionTransformer`**: `complaint_dissatisfaction = number_of_complaints × (1 − customer_satisfaction_score)`.
* **`ComplaintRiskBandTransformer`**: `complaint_risk_band = 1` when complaints < 3 — low engagement (few complaints) is itself a disengagement signal (~0.30 churn vs ~0.04 at ≥ 6 complaints).
* **`VisitsRiskBandTransformer`**: `visits_risk_band = 1` when visits ≥ 15 — heavy users show elevated churn (0.10 < 7 visits, 0.17 ≥ 15).
* **`DiscountDependencyTransformer`**: `discount_dependency = discount_usage_rate × monthly_spend` — spend captured through discounts (churn rises with the band: 0.15 < 0.2, 0.18 ≥ 0.4).
* **`HighDiscountFlagTransformer`**: `high_discount_flag = 1` when `discount_usage_rate ≥ 0.4`.
* **`ValueTierMismatchTransformer`**: `value_tier_mismatch = 1` for below-median spend (< 54.65) on the Premium tier — low-value spend on a high tier is the most at-risk mismatch group.

Plus two selector components:
* **`CorrelationFilter`**: Greedy near-collinearity removal — keeps the first column in order and drops any later column whose maximum |Pearson correlation| with an already-kept column is ≥ `redundancy_threshold` (0.90 in the current config). Stateless.
* **`FeatureSelector`**: Target-aware wrapper around `SelectKBest` that plugs into the pipeline like any transformer.
  - **Config knobs** (from `config.yaml`): `feature_selection.enabled` (default `true`), `feature_selection.method` (`mutual_info_classif` | `f_classif` | `chi2`; default `mutual_info_classif`), `feature_selection.k_best` (number of features to keep; code default `10`, currently `5`), `feature_selection.redundancy_filter` (default `true`), `feature_selection.redundancy_threshold` (default `0.90`), and the global `random_seed` (passed through for the stochastic mutual-information estimator).
  - **Behaviour**: When `enabled: false`, it passes features through unchanged; otherwise it drops near-duplicates via the `CorrelationFilter`, then fits the score function on the training matrix (target-aware) and retains the top-`k` columns. An unsupported `method` raises `ValueError` at pipeline construction.

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
* **Variant Benchmarking (A/B/C)**: Before the main run, three preprocessing variants are benchmarked on the same seeded 80/20 split across all three model families, to attribute performance to the pipeline design rather than model choice:
  - **Variant A** — full pipeline: source cleaning + 7 engineered features + redundancy-aware $k$-best selection (15 input features → 5 retained).
  - **Variant B** — no engineered features: raw numerics only (8 input features → 5 retained); the engineering step becomes a no-op passthrough.
  - **Variant C** — no feature selection: full feature matrix kept (15 features, selector disabled).
* **Champion Model Selection & Artifact Serialization**:
  - Automatically identifies the top-performing model strategy from **Variant A** (the full pipeline) based on cross-validation $F_1$-score optimization.
  - Serializes the champion estimator state and hyperparameter dictionary to `models/champion_model.joblib`.

---

## 6. Sprint 5 Model Benchmark & Optimization Summary

Detailed cross-validation and evaluation metrics across candidate model families evaluated on a 5-fold Stratified CV split and held-out test dataset ($N_{\text{test}} = 20\%$), run on the current leakage-safe pipeline (imputation and IQR capping of raw numerics before feature engineering, `PowerTransformer` active, redundancy filter at 0.90, $k=5$ selection).

**Variant A (full pipeline)** — the redundancy filter drops `complaint_dissatisfaction` (|corr| = 0.906 with `customer_satisfaction_score`), and the mutual-information selector then retains 5 of 15 features: `number_of_complaints`, `visits_risk_band`, `high_discount_flag`, `membership_tier_Basic`, and `membership_tier_Pro` (one-hot columns after `membership_tier` expansion):

| Model Strategy | Optimization Strategy | Best CV F1 | CV F1 (Mean ± Std) | Train F1 (Mean) | Gen Gap | Overfit Alert | Test Acc | Test Prec | Test Rec | Test F1 | Test ROC-AUC | Test PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | $L_2$ Regularization ($C=0.01$), Balanced Weights, $lbfgs$ | **0.3059** | 0.3059 ± 0.0412 | 0.3048 | −0.0011 | False | 0.545 | 0.1789 | 0.5667 | 0.2720 | 0.5875 | 0.2099 |
| **Random Forest** | Balanced Weights, $N_{\text{trees}}=200$, $max\_depth=3$, $min\_samples\_leaf=2$ | 0.2966 | 0.2966 ± 0.0317 | 0.3239 | 0.0273 | False | 0.550 | 0.1875 | 0.6000 | 0.2857 | 0.6102 | 0.2028 |
| **Gradient Boosting** | $\eta=0.2$, $\text{Depth}=3$, $N_{\text{trees}}=150$, Balanced Weights | 0.1302 | 0.1302 ± 0.0464 | 0.7829 | 0.6526 | True | 0.830 | 0.1667 | 0.0333 | 0.0556 | 0.5299 | 0.1583 |

**Champion**: Variant A **Logistic Regression** (Best CV F1 **0.3059**), serialized to `models/champion_model.joblib`.

**Variant comparison** (Test F1 on the same seeded 80/20 split; Variant B = raw numerics only, Variant C = no feature selection):

| Model | Variant A (full pipeline) | Variant B (raw numerics) | Variant C (no selection) |
| :--- | :--- | :--- | :--- |
| Logistic Regression | 0.2720 | 0.2698 | 0.2439 |
| Random Forest | 0.2857 | 0.2835 | 0.2764 |
| Gradient Boosting | 0.0556 | 0.0556 | 0.0000 |

Engineering + redundancy-aware selection is net non-negative for every model (A ≥ B ≥ C for Logistic Regression and Random Forest; A = B > C for Gradient Boosting) — the derived domain features help, and dropping the near-collinear `complaint_dissatisfaction` alongside the $k=5$ cut does not cost accuracy.

> **Note on the lower absolute scores vs the earlier Sprint 5 table (LR 0.4543 / RF 0.4880 / GB 0.4311)**: those numbers came from the pre-remediation pipeline, which fitted imputation/capping on the full dataset before feature engineering and selected $k=10$ without a redundancy filter. The current numbers reflect the stricter leakage-safe pipeline (train-only fitting throughout, capping before engineering, $k=5$ with the 0.90 redundancy filter) and are the honest scores to compare against going forward.

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


---

## 8. Sprint 6: Comprehensive Documentation Deliverables

Sprint 6 completes the capstone production readiness with standardized technical documentation stored in `documentation/`:

- **API Documentation (`documentation/API_DOCUMENTATION.md`)**: Complete module interface specifications, method signatures, parameter types, and pipeline usage guides for `preprocessing.py` and `model_optimization.py`.
- **Model Explanations & Interpretability (`documentation/MODEL_EXPLANATIONS.md`)**: Comprehensive mathematical framing, bias-variance analysis, feature importance ranking (`complaint_dissatisfaction`), and model selection tradeoffs.
- **Production Deployment Guide (`documentation/DEPLOYMENT_GUIDE.md`)**: Step-by-step instructions for environment setup, artifact loading, batch inference execution, and model drift monitoring.
