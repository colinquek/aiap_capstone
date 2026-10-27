"""src/preprocessing.py - Data Preprocessing Pipeline Module."""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

# Terminal ANSI Color Constants
RED_BOLD = "\033[1;31m"
GREEN_BOLD = "\033[1;32m"
CLEAR = "\033[0m"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("preprocessing")


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


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Custom transformer to cap outliers based on Interquartile Range (IQR) bounds learned during fit."""

    def __init__(self, iqr_multiplier: float = 1.5, enabled: bool = True):
        self.iqr_multiplier = iqr_multiplier
        self.enabled = enabled
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        if not self.enabled:
            return self

        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            q25 = X_df[col].quantile(0.25)
            q75 = X_df[col].quantile(0.75)
            iqr = q75 - q25
            self.lower_bounds_[col] = q25 - (self.iqr_multiplier * iqr)
            self.upper_bounds_[col] = q75 + (self.iqr_multiplier * iqr)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled or not self.lower_bounds_:
            return X

        X_out = pd.DataFrame(X).copy()
        for col in X_out.columns:
            if col in self.lower_bounds_:
                X_out[col] = np.clip(
                    X_out[col], self.lower_bounds_[col], self.upper_bounds_[col]
                )
        return X_out


class DataPreprocessingPipeline:
    """Preprocessing pipeline encapsulating validation, target encoding, feature engineering, scaling, and encoding."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_engineering_pipeline = Pipeline(
            steps=[
                ("spend_per_visit", SpendPerVisitTransformer()),
                (
                    "complaint_dissatisfaction",
                    ComplaintDissatisfactionTransformer(),
                ),
            ]
        )
        self.column_transformer = self._create_column_transformer()
        self.is_fitted = False

    def audit_raw_missingness(self, df: pd.DataFrame) -> pd.Series:
        """Audits raw missingness before imputation or scaling."""
        logger.info(
            f"{GREEN_BOLD}[INFO] Executing Raw Missingness Audit"
            f" (Pre-Cleaning)...{CLEAR}"
        )
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        if total_missing > 0:
            logger.info(
                f"{RED_BOLD}[FINDING] Missing Values Detected Prior to"
                f" Cleaning:\n{missing_counts[missing_counts > 0]}{CLEAR}"
            )
        else:
            logger.info(
                f"{GREEN_BOLD}[INFO] Zero missing values detected in raw"
                f" dataset.{CLEAR}"
            )
        return missing_counts

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validates input column schema and verifies feature presence."""
        required_features = self.config.get("raw_features", [])
        missing_cols = [col for col in required_features if col not in df.columns]
        if missing_cols:
            logger.error(
                f"{RED_BOLD}[ERROR] Schema Validation Failed! Missing columns:"
                f" {missing_cols}{CLEAR}"
            )
            raise ValueError(f"Missing required columns: {missing_cols}")
        logger.info(
            f"{GREEN_BOLD}[INFO] Schema validation passed ({len(required_features)}"
            f" raw features verified).{CLEAR}"
        )
        return True

    def process_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Encodes string target labels to binary integers (1=Churned, 0=Retained)."""
        target_col = self.config.get("target_column", "customer_churn")
        if target_col not in df.columns:
            logger.error(
                f"{RED_BOLD}[ERROR] Target column '{target_col}' missing!{CLEAR}"
            )
            raise KeyError(f"Target column '{target_col}' not found.")

        X = df.drop(columns=[target_col])
        y_raw = df[target_col]

        if y_raw.dtype == object or isinstance(y_raw.iloc[0], str):
            y = (y_raw == "Churned").astype(int)
            logger.info(
                f"{GREEN_BOLD}[INFO] Encoded target '{target_col}' (Churned -> 1,"
                f" Retained -> 0).{CLEAR}"
            )
        else:
            y = y_raw.astype(int)

        counts = y.value_counts()
        imbalance_ratio = counts.get(0, 0) / max(counts.get(1, 1), 1)
        logger.info(f"{RED_BOLD}[FINDING] Target Class Counts:\n{counts}{CLEAR}")
        logger.info(
            f"{RED_BOLD}[FINDING] Imbalance Ratio (Majority / Minority):"
            f" {imbalance_ratio:.2f}:1{CLEAR}"
        )
        return X, y

    def _create_column_transformer(self) -> ColumnTransformer:
        """Creates scikit-learn ColumnTransformer for numerical scaling, imputation, outlier capping, and ordinal encoding."""
        num_cols = self.config.get("numerical_features", [])
        ord_cols = self.config.get("ordinal_features") or self.config.get(
            "categorical_features", []
        )
        ord_categories = self.config.get(
            "ordinal_categories", [["Basic", "Pro", "Premium"]]
        )

        impute_config = self.config.get("imputation", {})
        num_impute_strategy = impute_config.get("numerical_strategy", "median")
        ord_impute_strategy = impute_config.get("ordinal_strategy", "most_frequent")

        outlier_config = self.config.get("outlier_handling", {})
        outlier_enabled = outlier_config.get("enabled", True)
        iqr_multiplier = outlier_config.get("iqr_multiplier", 1.5)

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

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fits feature engineering and ColumnTransformer pipelines on training data."""
        logger.info(
            f"{GREEN_BOLD}[INFO] Executing Feature Engineering Pipeline...{CLEAR}"
        )
        X_fe = self.feature_engineering_pipeline.fit_transform(X)
        logger.info(
            f"{GREEN_BOLD}[INFO] Fitting ColumnTransformer on training matrix"
            f" (Shape: {X_fe.shape})...{CLEAR}"
        )
        X_trans = self.column_transformer.fit_transform(X_fe)
        self.is_fitted = True
        logger.info(
            f"{GREEN_BOLD}[INFO] Transformation complete. Shape:"
            f" {X_trans.shape}{CLEAR}"
        )
        return X_trans

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms validation or test sets using pre-fitted transformers without leakage."""
        if not self.is_fitted:
            logger.error(
                f"{RED_BOLD}[ERROR] Pipeline is not fitted! Call fit_transform on"
                f" training data first.{CLEAR}"
            )
            raise RuntimeError(
                "Pipeline instance is not fitted yet. Call 'fit_transform' on training"
                " data first."
            )
        X_fe = self.feature_engineering_pipeline.transform(X)
        return self.column_transformer.transform(X_fe)
