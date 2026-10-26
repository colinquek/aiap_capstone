import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


class BaselineModel:
    """Baseline Model wrapper for churn classification tasks."""

    def __init__(self, model_type: str = "logistic_regression", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model: BaseEstimator = self._init_model()

    def _init_model(self) -> BaseEstimator:
        if self.model_type == "dummy":
            return DummyClassifier(
                strategy="most_frequent", random_state=self.random_state
            )
        elif self.model_type == "logistic_regression":
            return LogisticRegression(random_state=self.random_state, max_iter=1000)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def fit(self, X: np.ndarray, y: pd.Series) -> "BaselineModel":
        """Fit the baseline model on preprocessed training features."""
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary target labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise NotImplementedError(f"{self.model_type} does not support predict_proba.")

    def evaluate(self, X: np.ndarray, y: pd.Series) -> Dict[str, float]:
        """Compute standard classification metrics."""
        y_pred = self.predict(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1_score": f1_score(y, y_pred, zero_division=0),
        }

        if hasattr(self.model, "predict_proba"):
            y_proba = self.predict_proba(X)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y, y_proba)

        return metrics

    def save_model(self, filepath: str) -> None:
        """Serialize the fitted model to disk."""
        joblib.dump(self.model, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "BaselineModel":
        """Load a serialized model from disk."""
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance
