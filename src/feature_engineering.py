"""src/feature_engineering.py - Feature engineering & selection module."""

from functools import partial

import numpy as np
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


class ComplaintRiskBandTransformer(BaseEstimator, TransformerMixin):
    """Flag below-threshold complaint volume as a churn risk band.

    Observed in the data: customers filing fewer than 3 complaints churn at
    ~0.30 vs ~0.04 at >= 6 complaints -- low engagement (few complaints)
    is itself a disengagement signal, not just complaint count.
    """

    def __init__(self, low_complaint_threshold: float = 3.0):
        self.low_complaint_threshold = low_complaint_threshold

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if "number_of_complaints" in X_out.columns:
            X_out["complaint_risk_band"] = (
                X_out["number_of_complaints"] < self.low_complaint_threshold
            ).astype(float)
        return X_out

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags


class VisitsRiskBandTransformer(BaseEstimator, TransformerMixin):
    """Flag high-visit customers (>= threshold) as a churn risk band.

    Churn rises monotonically with visits (0.10 < 7 visits, 0.17 >= 15):
    heavy users show elevated churn risk beyond raw visit frequency.
    """

    def __init__(self, high_visits_threshold: float = 15.0):
        self.high_visits_threshold = high_visits_threshold

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if "number_of_store_visits_per_month" in X_out.columns:
            X_out["visits_risk_band"] = (
                X_out["number_of_store_visits_per_month"]
                >= self.high_visits_threshold
            ).astype(float)
        return X_out

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags


class DiscountDependencyTransformer(BaseEstimator, TransformerMixin):
    """Spend captured through discounts (rate x spend).

    Encodes discount dependency: customers whose spend is dominated by
    discounted purchases. Churn rises with the band (0.15 < 0.2, 0.18 >= 0.4).
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if (
            "discount_usage_rate" in X_out.columns
            and "monthly_spend" in X_out.columns
        ):
            X_out["discount_dependency"] = (
                X_out["discount_usage_rate"] * X_out["monthly_spend"]
            )
        return X_out

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags


class HighDiscountFlagTransformer(BaseEstimator, TransformerMixin):
    """Binary flag for customers above the high-discount-usage threshold."""

    def __init__(self, high_discount_threshold: float = 0.4):
        self.high_discount_threshold = high_discount_threshold

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if "discount_usage_rate" in X_out.columns:
            X_out["high_discount_flag"] = (
                X_out["discount_usage_rate"] >= self.high_discount_threshold
            ).astype(float)
        return X_out

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags


class ValueTierMismatchTransformer(BaseEstimator, TransformerMixin):
    """Flag low-spend customers on premium tiers (value / tier mismatch).

    Observed pattern: Premium-tier churn (0.13) sits between Basic (0.19) and
    Pro (0.05) -- low-spend premium-tier customers are the most at-risk
    mismatch group, so the band flags them explicitly.
    """

    def __init__(self, below_median_spend_threshold: float = 54.65):
        self.below_median_spend_threshold = below_median_spend_threshold

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if (
            "monthly_spend" in X_out.columns
            and "membership_tier" in X_out.columns
        ):
            tier_rank = X_out["membership_tier"].map(
                {"Basic": 1, "Pro": 2, "Premium": 3}
            )
            X_out["value_tier_mismatch"] = (
                (X_out["monthly_spend"] < self.below_median_spend_threshold)
                & (tier_rank >= 3)
            ).astype(float)
        return X_out

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop near-collinear features before target-aware selection.

    Keeps the first column in order and drops any later column whose
    absolute Pearson correlation with a kept column is >= threshold.
    Ensures SelectKBest ranks features that are not redundant copies.
    """

    def __init__(self, threshold: float = 0.90, enabled: bool = True):
        self.threshold = threshold
        self.enabled = enabled

    def fit(self, X, y=None):
        if self.enabled:
            X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = X_df.corr().abs()
            kept, dropped = [], []
            for col in X_df.columns:
                if kept and corr.loc[kept, col].max() >= self.threshold:
                    dropped.append(col)
                else:
                    kept.append(col)
            self.kept_features_ = kept
            self.dropped_features_ = dropped
            print(
                f"[INFO] Redundancy filter dropped "
                f"{len(dropped)} near-collinear feature(s): {dropped}"
            )
        return self

    def transform(self, X):
        if self.enabled and hasattr(self, "kept_features_"):
            X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
            available = [c for c in self.kept_features_ if c in X_df.columns]
            return X_df[available]
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Target-aware feature selection transformer using SelectKBest."""

    def __init__(self, k: int = 10, enabled: bool = True, method: str = "mutual_info_classif", random_state: int = 42,
                 redundancy_threshold: float = 0.90, redundancy_filter: bool = True):
        self.k = k
        self.enabled = enabled
        self.method = method
        self.random_state = random_state
        self.redundancy_threshold = redundancy_threshold
        self.redundancy_filter = redundancy_filter
        score_func = self._resolve_score_func()
        self.selector = SelectKBest(score_func=score_func, k=self.k) if enabled else None
        self.filter = CorrelationFilter(threshold=redundancy_threshold, enabled=redundancy_filter)

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
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            X_ranked = self.filter.fit_transform(X, y)
            k = min(self.k, X_ranked.shape[1])
            self.selector = SelectKBest(score_func=self._resolve_score_func(), k=k)
            self.selector.fit(X_ranked, y)
            kept = X_ranked.columns[self.selector.get_support()]
            print(f"[INFO] Selected features: {list(kept)}")
        return self

    def transform(self, X):
        if self.enabled and self.selector is not None:
            X = self.filter.transform(X)
            return self.selector.transform(X)
        return X
