# Project Feedback Report

Hi there,

Thank you for your submission. This report gives you a student-focused review of your project, highlighting what you did well and where you can take your machine learning work further.

## Table of Contents

- [1. Overall Summary](#1-overall-summary)
- [2. Machine Learning Pipeline (MLP) Feedback](#2-machine-learning-pipeline-mlp-feedback)
  - [2.1. Analysis & Interpretation](#21-analysis--interpretation)
  - [2.2. Improvement Plan](#22-improvement-plan)
- [3. Conclusion](#3-conclusion)

## 1. Overall Summary

Your project tackles a meaningful end-to-end supervised learning task: predicting customer churn from behavioral, transactional, and membership-related features. Overall, you built a solid binary classification pipeline that is well structured, reproducible, and clearly designed for comparing multiple models.

Some of your strongest areas were:

- **A consistent preprocessing workflow** that handled numeric and categorical data in a careful, leak-aware way.
- **A modular modeling design** that made it easy to compare baseline and advanced models through a shared interface.
- **Thoughtful evaluation coverage**, including both threshold-based metrics and probability-based metrics such as ROC-AUC, PR-AUC, and log loss.
- **Good reproducibility habits**, especially your use of a shared random seed and saved model artifacts.

The main areas to strengthen next are:

- **Tuning your advanced models more systematically**, rather than relying on default settings.
- **Deepening your model comparison**, especially around overfitting, interpretability, and practical trade-offs.
- **Making your evaluation outputs more complete and auditable**, for example with confusion matrices and saved benchmark results.

Taken together, this is a strong foundation. Your project already shows that you can build a functioning ML pipeline; the next stage is making your experimentation and analysis more rigorous.

## 2. Machine Learning Pipeline (MLP) Feedback

This section focuses on the structure, quality, and analytical depth of your machine learning pipeline.

### 2.1. Analysis & Interpretation

#### What went well

**1. You implemented genuinely different advanced models in a clean, reusable way**

One of the clearest strengths in your project is that you did not just add extra models—you wrapped them in a consistent interface that supports training, prediction, and evaluation.

```python
class AdvancedModel:
    """Advanced Model wrapper supporting ensemble and tree-based architectures."""

    def __init__(
        self,
        model_type: str = "random_forest",
        random_state: int = 42,
        **kwargs: Any,
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.kwargs = kwargs
        self.model: BaseEstimator = self._init_model()

    def _init_model(self) -> BaseEstimator:
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.kwargs.get("n_estimators", 100),
                max_depth=self.kwargs.get("max_depth", None),
                min_samples_split=self.kwargs.get("min_samples_split", 2),
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.kwargs.get("n_estimators", 100),
                learning_rate=self.kwargs.get("learning_rate", 0.1),
                max_depth=self.kwargs.get("max_depth", 3),
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unsupported advanced model_type: {self.model_type}")
```

This is good engineering and good ML practice. You chose two advanced models that are meaningfully different from one another and from your baselines, which makes the comparison more informative. You also exposed key hyperparameters and included an explicit guard for unsupported model types, which improves clarity and maintainability.

**2. Your preprocessing pipeline was consistent and fair across models**

You built a unified preprocessing workflow instead of handling each model separately. That is exactly the right direction for fair benchmarking.

```python
num_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy=num_impute_strategy)),
        (
            "capper",
            OutlierCapper(
                iqr_multiplier=iqr_multiplier, enabled=outlier_enabled
            ),
        ),
        ("scaler", RobustScaler()),
    ]
)

ord_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy=ord_impute_strategy)),
        (
            "ordinal",
            OrdinalEncoder(
                categories=ord_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ]
)

return ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("ord", ord_transformer, ord_cols),
    ],
    remainder="passthrough",
    n_jobs=-1,
)
```

You also handled train/test leakage correctly:

```python
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)
```

This shows strong awareness of experimental integrity. Fitting preprocessing on training data only, then transforming the test data afterward, is essential for trustworthy results. Your use of imputation, outlier handling, scaling, and categorical encoding also shows that you thought carefully about feature types rather than applying a one-size-fits-all approach.

**3. Your evaluation suite was broad and appropriate for churn prediction**

For churn classification, relying on accuracy alone would have been too shallow. You avoided that and used a richer set of metrics.

```python
metrics = {
    "accuracy": round(float(accuracy_score(y, y_pred)), 4),
    "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
    "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
    "f1_score": round(float(f1_score(y, y_pred, zero_division=0)), 4),
}

if hasattr(self.model, "predict_proba"):
    try:
        y_proba = self.predict_proba(X)
        pos_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]

        metrics["roc_auc"] = round(float(roc_auc_score(y, pos_proba)), 4)
        metrics["pr_auc"] = round(
            float(average_precision_score(y, pos_proba)), 4
        )
        metrics["log_loss"] = round(float(log_loss(y, y_proba)), 4)
    except Exception:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["log_loss"] = None
```

This is a strong choice because churn prediction often involves class imbalance and business trade-offs. By including ROC-AUC, PR-AUC, and log loss alongside precision, recall, and F1, you created a more complete view of model performance.

**4. Your project structure supported repeatable execution**

You set up the experiment so that it could be run consistently and not just as a one-off exploration. In particular, you handled randomness explicitly:

```python
seed = config.get("random_seed", config.get("random_state", 42))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y
)
```

And you reused that same seed across models:

```python
AdvancedModel(model_type="random_forest", random_state=seed),
AdvancedModel(model_type="gradient_boosting", random_state=seed),
```

You also saved artifacts after training, which is a good habit for reproducibility and deployment readiness. Altogether, this makes your project feel like a proper pipeline rather than a disconnected experiment.

---

#### What was missing

**1. The training protocol was fair, but it was not yet fully optimized**

You used the same split, the same processed features, and the same random seed for all candidate models, which makes the benchmark fair. However, the advanced models were still being compared mostly at their default settings.

```python
model_candidates = [
    ("dummy", "baseline", BaselineModel(model_type="dummy", random_state=seed)),
    (
        "logistic_regression",
        "baseline",
        BaselineModel(model_type="logistic_regression", random_state=seed),
    ),
    (
        "random_forest",
        "advanced",
        AdvancedModel(model_type="random_forest", random_state=seed),
    ),
    (
        "gradient_boosting",
        "advanced",
        AdvancedModel(model_type="gradient_boosting", random_state=seed),
    ),
]
```

This means your comparison is consistent, but it does not yet show the full potential of the stronger models. Since your advanced model wrapper already accepts hyperparameters, you are very close to being able to run more rigorous tuning. Right now, the project demonstrates a working benchmark, but not yet a fully explored one.

**2. Your comparison identified a champion, but the analysis could go deeper**

You correctly trained each model and evaluated it on train and test data:

```python
for name, m_category, model_obj in model_candidates:
    model_obj.fit(X_train_processed, y_train)
    train_metrics = model_obj.evaluate(X_train_processed, y_train)
    test_metrics = model_obj.evaluate(X_test_processed, y_test)
```

However, your final benchmark summary only surfaced the test-side metrics:

```python
benchmark_results.append(
    {
        "Model": name,
        "Category": m_category,
        "Test Acc": test_metrics.get("accuracy"),
        "Test Prec": test_metrics.get("precision"),
        "Test Rec": test_metrics.get("recall"),
        "Test F1": test_metrics.get("f1_score"),
        "Test ROC-AUC": test_metrics.get("roc_auc"),
        "Test PR-AUC": test_metrics.get("pr_auc"),
        "Test LogLoss": test_metrics.get("log_loss"),
    }
)
```

And your champion selection focused on the top test F1 score:

```python
print(
    f"\n[CHAMPION] Top Performing Model: {best_model_name.upper()} (F1: {best_f1})"
)
```

That is a reasonable starting rule, but it leaves out a more mature discussion of trade-offs. For example:

- Is the winning model also the most stable?
- Is there evidence of overfitting when train and test performance are compared directly?
- How interpretable is the chosen model?
- Does the best-scoring model justify any extra complexity or computation?

So, while your pipeline does choose a best model successfully, the analytical story around *why* that model is best in practice is not yet as developed as it could be.

### 2.2. Improvement Plan

Here are the most useful next steps to strengthen your project further:

- **Tune at least one of your advanced models systematically.** Since Gradient Boosting appears to be your strongest model so far, it would be a great candidate for a structured search over settings such as the number of estimators, learning rate, and tree depth. Random Forest would also benefit from exploring parameters such as depth and split rules. A split-aware approach such as cross-validation would make your conclusions much stronger.

- **Bring train and test metrics into the same comparison table.** You are already computing both, which is excellent. The next step is to display them side by side so you can clearly see whether a model is generalizing well or simply fitting the training data too closely.

- **Add interpretability to your comparison, not just ranking.** For tree-based models, feature importance can help you explain what is driving churn predictions. That would make your analysis more useful from both a technical and business perspective, especially if the most influential features align with the project’s domain logic.

- **Make your evaluation outputs easier to audit across runs.** At the moment, much of the comparison is printed during execution. Saving your benchmark results to a dedicated output file would make it much easier to track changes, compare experiments, and justify your final model choice later.

- **Expand your evaluation with confusion-matrix thinking.** In a churn setting, false negatives and false positives do not have the same business cost. A confusion matrix would help you understand exactly where each model is making mistakes and whether your selected model matches the real-world objective of the project.

- **State your prediction threshold more explicitly.** Even though the default threshold is commonly understood, documenting it clearly would improve the transparency of your evaluation setup. This matters even more in churn problems, where shifting the threshold can significantly affect recall and precision.

- **Connect your winning model to a practical recommendation.** Rather than only saying which model scored highest, try to explain why you would actually deploy it. For example, you might weigh predictive performance against explainability, risk of overfitting, or ease of communicating results to stakeholders.

## 3. Conclusion

You have built a strong and credible machine learning pipeline with many of the right foundations already in place: modular model design, clean preprocessing, fair benchmarking, and reproducibility controls. That is a very good place to be.

The biggest opportunity now is not simply to add more code, but to make your experimentation more deliberate: tune your advanced models, compare them more deeply, and turn your results into a clearer analytical story. Feedback like this is meant to help you sharpen an already solid project, and you are well positioned to do exactly that.