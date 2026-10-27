# Production Deployment Guide

## 1. System Requirements & Environment Setup
- **Python Version**: Python 3.10+
- **Environment Creation**:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```

---

## 2. Directory & Artifact Structure
Ensure the following artifacts are serialized before running inference:
- `models/preprocessing_pipeline.joblib`: Preprocessing transformer state.
- `models/champion_model.joblib`: Serialized GBDT champion estimator and hyperparameter configuration.

---

## 3. Batch Inference Execution
Execute batch predictions using `main.py` or import pipeline artifacts:

```python
import joblib
import pandas as pd

# Load pipeline artifacts
pipeline = joblib.load("models/preprocessing_pipeline.joblib")
champion_data = joblib.load("models/champion_model.joblib")
model = champion_data["best_estimator"]

# Process new records
df_raw = pd.read_csv("data/new_customers.csv")
X_processed = pipeline.transform(df_raw)
churn_probabilities = model.predict_proba(X_processed)[:, 1]
predictions = (churn_probabilities >= 0.5).astype(int)
```

---

## 4. Logging & Monitoring
- **Logging**: Configured via Python's standard `logging` library with explicit colored log levels.
- **Monitoring Requirements**:
  - Track input feature distributions to detect covariate shift (`monthly_spend`, `customer_satisfaction_score`).
  - Monitor prediction drift and precision/recall performance against monthly churn outcome labels.
