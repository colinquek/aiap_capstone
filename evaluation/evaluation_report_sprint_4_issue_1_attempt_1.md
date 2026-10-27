# Project Feedback Report

Hi there,

Thank you for your submission. This report provides a detailed breakdown of your project, highlighting what you did well and where you can strengthen your work further. Your project tackles an end-to-end machine learning task in customer churn prediction, and you have already put several important building blocks in place for a solid baseline pipeline.

## Table of Contents

- [1. Overall Summary](#1-overall-summary)
- [2. Machine Learning Pipeline (MLP) Feedback](#2-machine-learning-pipeline-mlp-feedback)
  - [2.1. Analysis & Interpretation](#21-analysis--interpretation)
  - [2.2. Improvement Plan](#22-improvement-plan)
- [3. Conclusion](#3-conclusion)

## 1. Overall Summary

You built a strong foundation for a binary classification pipeline focused on predicting customer churn from behavioural and transactional features. Overall, your work shows good awareness of the key stages of a machine learning workflow: target preparation, preprocessing, training, evaluation, and artifact saving.

A few of your strongest areas were:

- **Clear baseline model design**, with a wrapper that supports both a naive approach and a trainable logistic regression model.
- **Well-structured preprocessing**, including missing-value handling, feature grouping, scaling, and feature engineering in a reusable pipeline.
- **Good experimental discipline**, especially in your use of a stratified split, fixed randomness, and leakage-aware fit/transform order.

The main areas to focus on next are:

- **Making the benchmark comparison more complete**, so the naive baseline is actually evaluated side by side with the trainable model.
- **Improving the evaluation story for an imbalanced classification problem**, especially by adding metrics such as PR-AUC and possibly a probability-based error metric.
- **Tightening reproducibility**, particularly through config alignment and a portable dependency file.

Overall, this is a thoughtful submission with a lot of the right structure already in place.

## 2. Machine Learning Pipeline (MLP) Feedback

This section evaluates the structure and quality of your end-to-end machine learning pipeline.

### 2.1. Analysis & Interpretation

#### What went well

**1) You defined the baseline modeling approach clearly**

One of the strongest parts of your project is that you created a model wrapper with a clear interface and an explicit choice between a naive and trainable model:

```python
if self.model_type == "dummy":
    return DummyClassifier(
        strategy="most_frequent", random_state=self.random_state
    )
elif self.model_type == "logistic_regression":
    return LogisticRegression(random_state=self.random_state, max_iter=1000)
```

This is a good design decision because it keeps the interface consistent while allowing you to experiment with different baseline strategies. It also makes your code easier for another person to read and extend.

You also defined the target clearly as a binary churn label:

```python
if y_raw.dtype == object or isinstance(y_raw.iloc[0], str):
    y = (y_raw == "Churned").astype(int)
```

That is important because it removes ambiguity around what the model is predicting and ensures the rest of the pipeline is aligned to the problem formulation.

---

**2) Your preprocessing pipeline is reusable and thoughtfully structured**

You handled preprocessing in a way that is much stronger than one-off cleaning logic. The numerical transformations are grouped in a proper pipeline:

```python
num_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy=num_impute_strategy)),
        ("capper", OutlierCapper(iqr_multiplier=iqr_multiplier, enabled=outlier_enabled)),
        ("scaler", RobustScaler()),
    ]
)
```

And your ordinal feature processing is also clearly separated:

```python
ord_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy=ord_impute_strategy)),
        ("ordinal", OrdinalEncoder(
            categories=ord_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ]
)
```

This is a strong implementation because it shows that you are thinking in terms of reusable transformations rather than ad hoc preprocessing. It is also a good sign that you routed features through explicit groups and built this into a pipeline that can be reused during inference.

I also want to highlight your feature engineering choices. Creating features such as `spend_per_visit` and `complaint_dissatisfaction` suggests that you were trying to capture more meaningful behavioural patterns, not just passing raw variables directly into the model.

---

**3) You followed a good leakage-aware training and validation flow**

Your validation workflow is ordered correctly and reproducibly:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y
)
```

Then you fit preprocessing on the training data first and only transform the test data afterward:

```python
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)
```

This is exactly the right pattern for avoiding train-test leakage. It is especially encouraging that your pipeline also guards against accidental misuse before fitting:

```python
if not self.is_fitted:
    raise RuntimeError(
        "Pipeline instance is not fitted yet. Call 'fit_transform' on training data first."
    )
```

That kind of safeguard shows good engineering awareness and helps make your pipeline more reliable.

---

**4) Your prediction interface is clean and practical**

Your model class exposes the standard methods a downstream user would expect:

```python
def predict(self, X: np.ndarray) -> np.ndarray:
    """Predict binary target labels."""
    return self.model.predict(X)

def predict_proba(self, X: np.ndarray) -> np.ndarray:
    """Predict class probabilities."""
    if hasattr(self.model, "predict_proba"):
        return self.model.predict_proba(X)
    raise NotImplementedError(f"{self.model_type} does not support predict_proba.")
```

This is a solid choice because it keeps the model easy to integrate into the rest of the workflow and mirrors common machine learning library conventions.

You also save both the preprocessing pipeline and the trained model, which is another strength:

```python
preprocessor_path = "models/preprocessing_pipeline.joblib"
joblib.dump(pipeline, preprocessor_path)

model_path = "models/baseline_model.joblib"
baseline_model.save_model(model_path)
```

That gives your project a more realistic end-to-end structure and shows you were thinking about reuse beyond a single training run.

#### What was missing

**1) The naive benchmark is defined, but not actually compared in the main workflow**

Although you built support for a dummy baseline inside your model wrapper, the main training flow currently runs only the logistic regression baseline:

```python
model_type = config.get("baseline_model_type", "logistic_regression")
baseline_model = BaselineModel(model_type=model_type, random_state=seed)

baseline_model.fit(X_train_processed, y_train)

train_metrics = baseline_model.evaluate(X_train_processed, y_train)
test_metrics = baseline_model.evaluate(X_test_processed, y_test)
```

This means the project has the pieces needed for a proper benchmark comparison, but the comparison is not yet visible in the actual pipeline output. As a result, a reader cannot immediately tell how much better your trainable model is than a simple majority-class strategy.

For a baseline machine learning project, that side-by-side comparison is important. Without it, the reported results are harder to interpret because there is no direct reference point.

---

**2) Your metrics are a good start, but they do not yet fully support an imbalanced churn problem**

Right now, your evaluation function includes accuracy, precision, recall, F1-score, and ROC-AUC:

```python
metrics = {
    "accuracy": accuracy_score(y, y_pred),
    "precision": precision_score(y, y_pred, zero_division=0),
    "recall": recall_score(y, y_pred, zero_division=0),
    "f1_score": f1_score(y, y_pred, zero_division=0),
}

if hasattr(self.model, "predict_proba"):
    y_proba = self.predict_proba(X)[:, 1]
    metrics["roc_auc"] = roc_auc_score(y, y_proba)
```

These are useful metrics, but for churn prediction with class imbalance, it would be stronger to include at least one additional probability- or ranking-oriented metric such as **PR-AUC / Average Precision**. A metric like **log loss** could also help you assess the quality of the predicted probabilities, not just the final hard class labels.

Another small refinement is that your threshold-dependent metrics rely on the model's default threshold behaviour, but that threshold is not explicitly surfaced in your evaluation reporting. Making this clearer would improve interpretability.

---

**3) Your configuration is not fully aligned with the main workflow**

There is a mismatch between the keys you read in the main pipeline and the keys you actually define in your configuration:

```python
seed = config.get("random_seed", 42)
model_type = config.get("baseline_model_type", "logistic_regression")
```

Because the configuration uses `random_state` and does not define `baseline_model_type`, the workflow falls back to hardcoded defaults. In practice, that means your config file is not fully controlling the experiment behaviour.

This is not a conceptual modeling issue, but it does matter for reproducibility and maintainability. A central config file is most useful when it truly acts as the single source of truth.

---

**4) Dependency management is not yet portable**

Your project structure is clear and your entry-point design is sensible, but the dependency file still needs improvement for reproducibility across machines. The listed dependencies appear to come directly from a local environment export rather than a portable requirements list.

That makes it difficult for another user to recreate the environment successfully. Even if your code is well-written, reproducibility depends not only on the code itself, but also on making setup straightforward for anyone else who wants to run the project.

### 2.2. Improvement Plan

- **Add a visible benchmark comparison.** Since your model wrapper already supports both a naive classifier and logistic regression, the next valuable improvement is to evaluate both under the same train/test setup and present their results together. This will make your baseline story much more convincing.

- **Expand the evaluation metrics to better reflect class imbalance.** For churn prediction, include a metric such as PR-AUC / Average Precision so you can assess how well the model identifies the minority class. You should also consider a probability-focused error metric, such as log loss, to strengthen your interpretation of churn risk scores.

- **Make your configuration file fully effective.** Review the parameter names used in your main workflow and align them with the keys defined in your config. That way, when you change a seed or model type, the pipeline actually responds to it.

- **Improve portability of your environment setup.** Replace machine-specific dependency entries with a clean, standard dependency list that another developer could install in a fresh environment. This will make your project easier to run, review, and share.

- **Keep building the evaluation narrative around decision usefulness.** Beyond reporting numbers, aim to explain what those metrics mean for the churn problem. For example, think about whether the chosen metrics help you judge how well the model identifies at-risk customers, not just whether it performs well overall.

## 3. Conclusion

You have built a solid baseline machine learning pipeline with several strong foundations already in place: clear target handling, reusable preprocessing, leakage-aware validation, and a practical prediction interface. Those are important strengths, and they show good instincts for structuring machine learning work in a professional way.

The main next steps are to strengthen the benchmark comparison, report metrics that better match the imbalanced nature of the problem, and improve reproducibility through cleaner configuration and dependency management. These are very achievable refinements, and addressing them would significantly increase the quality and credibility of your project.

Overall, this is a promising submission. You are already doing many of the right things, and with a few targeted improvements, your project will become much more robust and polished.