# API & Pipeline Module Documentation

This document provides technical documentation for the primary modules in the `src/` directory.

---

## 1. Data Preprocessing Module (`src/preprocessing.py`)

### `DataPreprocessingPipeline`
```python
class DataPreprocessingPipeline:
    def __init__(self, config: Dict[str, Any]): ...
```
- **Description**: Manages schema validation, missingness auditing, outlier capping, feature engineering (`spend_per_visit`, `complaint_dissatisfaction`), scaling (`RobustScaler`), and encoding (`OrdinalEncoder`).
- **Methods**:
  - `audit_raw_missingness(df: pd.DataFrame) -> None`: Computes and logs pre-imputation missingness.
  - `validate_schema(df: pd.DataFrame) -> None`: Validates input columns against configuration schema.
  - `process_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]`: Encodes binary target (`customer_churn`).
  - `fit_transform(X: pd.DataFrame) -> pd.DataFrame`: Fits preprocessing transformers on training data.
  - `transform(X: pd.DataFrame) -> pd.DataFrame`: Transforms unseen evaluation/test data without data leakage.

---

## 2. Model Optimization Module (`src/model_optimization.py`)

### `ModelOptimizer`
```python
class ModelOptimizer:
    def __init__(self, config: Dict[str, Any]): ...
```
- **Description**: Manages Stratified $K$-Fold cross-validation, hyperparameter grid search (`GridSearchCV`/`RandomizedSearchCV`), explicit generalization diagnostics, and artifact saving.
- **Methods**:
  - `evaluate_cv_baseline(estimator, X, y) -> Dict[str, Any]`: Computes mean/std validation metrics and explicit $F_1$ generalization gap ($	ext{Train } F_1 - 	ext{Val } F_1$).
  - `generate_learning_curve(estimator, X, y) -> Dict[str, np.ndarray]`: Generates learning curve arrays across training fractions ($20\% - 100\%$).
  - `tune_model(model_name, X_train, y_train) -> Tuple[Any, Dict[str, Any]]`: Performs grid search over model-specific parameter grids.
  - `optimize_all_models(X_train, y_train, X_test, y_test) -> pd.DataFrame`: Orchestrates candidate tuning across Logistic Regression, Random Forest, and Gradient Boosting.
  - `save_champion_model(filepath: str) -> None`: Serializes top estimator state and hyperparameter dictionary to disk.

---

## 3. Orchestration Script (`main.py`)

- **Description**: Top-level driver executing schema audits, pipeline transformation, cross-validation tuning, diagnostics, test set confirmative scoring, and artifact serialization.
- **Usage**:
  ```bash
  python main.py src/config.yaml
  ```
