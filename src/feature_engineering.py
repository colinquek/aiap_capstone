"""src/feature_engineering.py - Feature engineering & selection module."""

from functools import partial

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif


class SpendPerVisitTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer calculating spend per store visit feature."""

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if (
            "monthly_spend" in X_out.columns
            and "number_of_store_visits_per_month" in X_out.columns
        ):
            X_out["spend_per_visit"] = X_out["monthly_spend"] / (
                X_out["number_of_store_visits_per_month"] + self.eps
            )
        return X_out

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags


class ComplaintDissatisfactionTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer calculating complaint x dissatisfaction feature."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if (
            "number_of_complaints" in X_out.columns
            and "customer_satisfaction_score" in X_out.columns
        ):
            X_out["complaint_dissatisfaction"] = X_out["number_of_complaints"] * (
                1.0 - X_out["customer_satisfaction_score"]
            )
        return X_out

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Target-aware feature selection transformer using SelectKBest."""

    def __init__(self, k: int = 10, enabled: bool = True, method: str = "mutual_info_classif", random_state: int = 42):
        self.k = k
        self.enabled = enabled
        self.method = method
        self.random_state = random_state
        score_func = self._resolve_score_func()
        self.selector = SelectKBest(score_func=score_func, k=self.k) if enabled else None

    def _resolve_score_func(self):
        if self.method == "mutual_info_classif":
            return partial(mutual_info_classif, random_state=self.random_state)
        if self.method == "f_classif":
            return f_classif
        if self.method == "chi2":
            return chi2
        raise ValueError(f"Unsupported feature selection method: {self.method}")

    def fit(self, X, y=None):
        if self.enabled and self.selector is not None:
            if y is None:
                raise ValueError("Target 'y' is required to fit FeatureSelector.")
            self.selector.fit(X, y)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        kept = X.columns[self.selector.get_support()]
        print(f"[INFO] Selected features: {list(kept)}")
        return self

    def transform(self, X):
        if self.enabled and self.selector is not None:
            return self.selector.transform(X)
        return X
