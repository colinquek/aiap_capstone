"""src/advanced_models.py - Advanced Model implementations for Churn Prediction."""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)


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

    def fit(self, X: np.ndarray, y: Union[pd.Series, np.ndarray]) -> "AdvancedModel":
        """Fit the advanced model on preprocessed training features."""
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

    def evaluate(
        self, X: np.ndarray, y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Compute evaluation metrics including PR-AUC and Log Loss for imbalanced churn data."""
        y_pred = self.predict(X)

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

        return metrics

    def get_feature_importances(self) -> np.ndarray:
        """Retrieve feature importances if supported by estimator."""
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        raise AttributeError(
            f"{self.model_type} does not support feature_importances_."
        )

    def save_model(self, filepath: str) -> None:
        """Serialize the fitted model to disk."""
        joblib.dump(self.model, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "AdvancedModel":
        """Load a serialized model from disk."""
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance
