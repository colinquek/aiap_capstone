# Customer Churn Prediction — AIAP Capstone Project

## 1. Executive Summary & Problem Objective
This project addresses customer churn prediction using the dataset (`synthetic_data_204.csv`). Customer churn represents a critical operational loss in retail and membership services. The business objective is to identify churn-prone customers early to allow proactive, cost-effective retention strategies.

### Mathematical Framing
* **Task Type:** Binary Classification
* **Target Variable ($y$):** `customer_churn`
  * `1` = `Churned` (Positive Class — Target of Interest)
  * `0` = `Retained` (Negative Class — Baseline State)
* **Dataset Scale:** 1,000 records, 6 predictor attributes, 1 target attribute.
* **Class Balance:** ~15% `Churned` (150 rows) vs ~85% `Retained` (850 rows).

---

## 2. Literature Review & Methodology Selection

### 2.1 Candidate Model Architectures
To establish a rigorous comparative framework, three distinct model families are evaluated:

1. **Logistic Regression (Linear Baseline):**
   - Serves as the interpretable baseline model.
   - Fits log-odds ratio: $\ln\left(\frac{P(y=1)}{1-P(y=1)}\right) = \beta_0 + \beta_1 X_1 + \dots + \beta_p X_p$.
   - Requires continuous normalization (`StandardScaler`) and categorical encoding (`OneHotEncoder`).

2. **K-Nearest Neighbors (KNN - Distance-Based Classifier):**
   - Evaluates local feature space density using Minkowski/Euclidean distance metrics.
   - Extremely sensitive to feature scales; requires strict preprocessing pipelines.

3. **Tree-Based Ensembles (Random Forest / Gradient Boosted Trees):**
   - Non-parametric models capable of capturing non-linear interactions (e.g., interaction between high complaint counts and low store visits).
   - Immune to feature scaling differences; provides tree-based feature importances.

### 2.2 Evaluation Metrics Strategy
Because the dataset exhibits a 15:85 class imbalance, standard **Accuracy** is misleading (a naive classifier predicting all `Retained` achieves 85% accuracy).

* **Primary Metrics:**
  - **ROC-AUC (Receiver Operating Characteristic Area Under Curve):** Evaluates class separation quality across all probability thresholds.
  - **PR-AUC (Precision-Recall Area Under Curve):** Focuses specifically on performance in the minority positive class (`Churned`).
  - **F1-Score:** Harmonic mean of Precision and Recall.
* **Secondary Metrics:** Precision, Recall, Confusion Matrix.

---

## 3. Production Pipeline Architecture

```text
├── data/
│   └── synthetic_data_204.csv   # Raw input dataset
├── models/                       # Persisted model pipeline artifacts (.joblib)
├── src/
│   ├── config.yaml               # Pipeline parameters, schema definitions, paths
│   ├── data_preparation.py       # Modular DataPreparation class (cleaning & scaling)
│   └── model_training.py         # Modular ModelTraining class (fitting & tuning)
├── tests/                        # Automated unit tests (pytest)
│   ├── test_data_preparation.py
│   └── test_model_training.py
├── eda.ipynb                     # Hypothesis-driven EDA and prototyping
├── main.py                       # Pipeline execution orchestrator
├── experiments.json              # Structured experiment tracking logs
├── requirements.txt              # Pinned dependencies
└── README.md                     # Comprehensive project documentation