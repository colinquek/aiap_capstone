"""src/models.py - Baseline Model wrappers and evaluation metrics."""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
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
        """Compute comprehensive classification metrics including imbalanced churn metrics."""
        y_pred = self.predict(X)

        metrics = {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y, y_pred, zero_division=0), 4),
        }

        if hasattr(self.model, "predict_proba"):
            try:
                y_proba = self.predict_proba(X)
                # Select positive class probabilities
                if y_proba.shape[1] > 1:
                    pos_proba = y_proba[:, 1]
                else:
                    pos_proba = y_proba[:, 0]

                metrics["roc_auc"] = round(roc_auc_score(y, pos_proba), 4)
                metrics["pr_auc"] = round(average_precision_score(y, pos_proba), 4)
                metrics["log_loss"] = round(log_loss(y, y_proba), 4)
            except Exception:
                metrics["roc_auc"] = None
                metrics["pr_auc"] = None
                metrics["log_loss"] = None

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
