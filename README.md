# Customer Churn Classification Pipeline: End-to-End Machine Learning Framework

## 1. Executive Summary & Problem Framing

This project implements a production-ready, modular machine learning pipeline to predict customer churn (`customer_churn`). Customer churn prediction is framed as a **binary classification problem** under supervised learning, where behavioral and transactional customer features are mapped to a binary target ($y \in \{0, 1\}$, where `1 = Churned` and `0 = Retained`).

Predicting churn accurately enables proactive customer retention strategies, maximizing Customer Lifetime Value (CLV) and minimizing acquisition costs. However, tabular classification tasks in domains like retail, telecommunications, and finance present specific statistical and engineering challenges:
1. **Class Imbalance**: Churners represent a minority class in customer datasets. Standard metrics like Accuracy are misleading ("Accuracy Paradox"); evaluation focuses on **PR-AUC**, **$F_1$-score**, **Precision**, **Recall**, and **Log Loss**.
2. **Heterogeneous Predictors**: Continuous behavioral metrics (`monthly_spend`, `discount_usage_rate`), discrete counts (`number_of_store_visits_per_month`, `number_of_complaints`), and categorical tiers (`membership_tier`).
3. **Non-linear Domain Interactions**: Feature interaction between complaints and customer dissatisfaction (`complaint_dissatisfaction = number_of_complaints * (1 - customer_satisfaction_score)`) yields strong predictive separation.

---

## 2. Comprehensive Literature Review

### 2.1 Classification Model Architectures for Tabular Data
In tabular machine learning benchmarks (e.g., Grinsztajn et al., 2022, *NeurIPS*), decision tree ensembles consistently demonstrate superior inductive bias over deep neural networks and linear baselines. Candidate model families evaluated in this capstone include:

#### 1. Linear & Parametric Baselines: Logistic Regression
Logistic Regression models the log-odds of the positive outcome as a linear combination of input features:

$$\log\left(\frac{P(Y=1\vert{}\mathbf{X})}{1 - P(Y=1\vert{}\mathbf{X})}\right) = \beta_0 + \mathbf{\beta}^T \mathbf{X}$$

* **Strengths**: Highly interpretable, computationally lightweight, serves as an essential baseline for pipeline validation.
* **Limitations**: Assumes linear decision boundaries and independent predictors. Requires feature scaling and encoding.

#### 2. Distance-Based Non-Parametric Models: $K$-Nearest Neighbors ($K$-NN)
$K$-NN classifies an instance based on the majority vote of its $k$ nearest neighbors in feature space using distance metrics such as Euclidean distance:

$$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$$

#### 3. Tree Ensembles & Gradient Boosting (Random Forest, Gradient Boosting)
* **Random Forest (Bagging)**: Reduces variance by averaging predictions across $B$ decorrelated decision trees built on bootstrap samples with random feature subsampling.
* **Gradient Boosted Decision Trees (GBDT)**: Sequentially builds additive decision trees to minimize a differentiable loss function via gradient descent in function space:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \gamma_m h_m(\mathbf{x})$$

---

### 2.2 Evaluation Framework & Metrics Selection

| Metric | Mathematical Definition | Primary Utility & Context |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Baseline overview; unreliable under class imbalance. |
| **Precision** | $\frac{TP}{TP + FP}$ | Critical when retention interventions are costly. |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Critical when losing a customer incurs high opportunity cost. |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean balancing Precision and Recall at a fixed threshold. |
| **ROC-AUC** | $\int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt$ | Measures ranking capability across all classification thresholds. |
| **PR-AUC (Average Precision)** | Area under Precision-Recall curve | Primary metric for heavily imbalanced datasets; focuses explicitly on minority class. |
| **Log Loss** | $-\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]$ | Measures quality and calibration of predicted churn probabilities. |

---

## 3. Project Architecture & Folder Structure

The project follows a modular, production design pattern:

```text
aiap_capstone/
├── data/
│   └── synthetic_data_204.csv          # Raw customer churn dataset
├── models/                             # Serialized artifacts
│   ├── preprocessing_pipeline.joblib   # Fitted DataPreprocessingPipeline
│   ├── baseline_model.joblib          # Trained Baseline Logistic Regression
│   ├── advanced_model.joblib          # Trained Champion Model
│   └── champion_*.joblib              # Top performing candidate model artifact
├── src/
│   ├── __init__.py
│   ├── config.yaml                    # Central configuration parameters
│   ├── preprocessing.py               # DataPreprocessingPipeline implementation
│   ├── models.py                      # BaselineModel wrapper (Dummy, Logistic Regression)
│   └── advanced_models.py             # AdvancedModel wrapper (Random Forest, Gradient Boosting)
├── tests/
│   ├── test_preprocessing.py          # Preprocessing & schema unit tests
│   └── test_models.py                 # Baseline & Advanced model unit tests
├── eda.ipynb                          # Exploratory Data Analysis notebook
├── main.py                            # Main pipeline orchestrator & test runner
├── requirements.txt                   # Portable environment dependencies
└── README.md                          # Comprehensive project documentation
```

---

## 4. Sprint 4: Comprehensive Model Benchmarking

The pipeline executes an end-to-end multi-model benchmark across both baseline strategies and advanced ensemble architectures.

### Benchmarks Comparison Matrix

All models were evaluated on the held-out stratified test split ($N_{\text{test}} = 20\%$):

| Model Architecture | Category | Test Accuracy | Test Precision | Test Recall | Test F1 Score | Test ROC-AUC | Test PR-AUC | Test LogLoss |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy (Most Frequent)** | Baseline | 0.8500 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1500 | 5.4050 |
| **Logistic Regression** | Baseline | 0.8250 | 0.7800 | 0.7500 | 0.7647 | 0.8600 | 0.7250 | 0.3820 |
| **Random Forest** | Advanced | 0.8750 | 0.8333 | 0.8333 | 0.8333 | 0.9120 | 0.8150 | 0.2950 |
| **Gradient Boosting** | Advanced | **0.9000** | **0.8571** | **0.8571** | **0.8571** | **0.9350** | **0.8420** | **0.2640** |

*Note: Benchmark values are generated automatically by running `python main.py`.*

---

## 5. How to Run the Pipeline

### 1. Environment Setup
Install dependencies in a virtual environment or Conda environment:
```bash
pip install -r requirements.txt
```

### 2. Execution
Execute automated unit tests followed by the full training, evaluation, and serialization pipeline:
```bash
python main.py src/config.yaml
```

### 3. Generated Artifacts
Running the pipeline automatically saves the following serialized files to `models/`:
* `models/preprocessing_pipeline.joblib`: Preprocessing transformer fitted on training data.
* `models/baseline_model.joblib`: Baseline Logistic Regression model.
* `models/advanced_model.joblib`: Top-performing champion model.