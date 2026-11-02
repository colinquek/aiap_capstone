"""src/preprocessing.py - Data Preprocessing Pipeline Module."""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
)

# Import refactored feature engineering logic
from src.feature_engineering import (
    SpendPerVisitTransformer,
    ComplaintDissatisfactionTransformer,
    ComplaintRiskBandTransformer,
    VisitsRiskBandTransformer,
    DiscountDependencyTransformer,
    HighDiscountFlagTransformer,
    ValueTierMismatchTransformer,
    FeatureSelector,
)

RED_BOLD = "\033[1;31m"
GREEN_BOLD = "\033[1;32m"
CLEAR = "\033[0m"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("preprocessing")


class _SourceColumnNamer:
    """Pickle-safe callable: renames source-cleaning output columns to a fixed set.

    Used as ``FunctionTransformer(feature_names_out=...)`` so the imputed/capped
    matrix (an ndarray with no column names) is re-labelled with the raw numeric
    feature names. Accepts any positional argument to stay robust across sklearn
    ``feature_names_out`` call conventions.
    """

    def __init__(self, names):
        self.names = list(names)

    def __call__(self, *args, **kwargs):
        return self.names


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
    """Preprocessing pipeline encapsulating validation, target encoding, feature engineering, scaling, and feature selection."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Raw numeric source columns: configured numerical features minus the
        # derived ones produced by feature engineering. These are cleaned first
        # so derived features are computed from imputed values, not NaNs.
        derived = {
            "spend_per_visit",
            "complaint_dissatisfaction",
            "complaint_risk_band",
            "visits_risk_band",
            "discount_dependency",
            "high_discount_flag",
            "value_tier_mismatch",
        }
        self.derived_features = derived
        self.source_numeric_cols = [
            col
            for col in self.config.get("numerical_features", [])
            if col not in derived
        ]
        self.source_cleaner = self._create_source_cleaner()
        self.feature_engineering_pipeline = Pipeline(
            steps=[
                ("spend_per_visit", SpendPerVisitTransformer()),
                (
                    "complaint_dissatisfaction",
                    ComplaintDissatisfactionTransformer(),
                ),
                ("complaint_risk_band", ComplaintRiskBandTransformer()),
                ("visits_risk_band", VisitsRiskBandTransformer()),
                ("discount_dependency", DiscountDependencyTransformer()),
                ("high_discount_flag", HighDiscountFlagTransformer()),
                ("value_tier_mismatch", ValueTierMismatchTransformer()),
            ]
        )
        self.column_transformer = self._create_column_transformer()

        fs_config = self.config.get("feature_selection", {})
        self.feature_selector = FeatureSelector(
            k=fs_config.get("k_best", self.config.get("top_k_features", 10)),
            enabled=fs_config.get("enabled", True),
            method=fs_config.get("method", "mutual_info_classif"),
            random_state=self.config.get("random_seed", 42),
            redundancy_threshold=fs_config.get("redundancy_threshold", 0.90),
            redundancy_filter=fs_config.get("redundancy_filter", True),
        )
        self.is_fitted = False

    def _create_source_cleaner(self) -> Pipeline:
        """Builds the source-cleaning stage (impute + cap) fitted on raw numeric columns BEFORE feature engineering."""
        impute_config = self.config.get("imputation", {})
        num_impute_strategy = impute_config.get("numerical_strategy", "median")
        outlier_config = self.config.get("outlier_handling", {})
        outlier_enabled = outlier_config.get("enabled", True)
        iqr_multiplier = outlier_config.get("iqr_multiplier", 1.5)
        name_source_cols = FunctionTransformer(
            feature_names_out=_SourceColumnNamer(self.source_numeric_cols)
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=num_impute_strategy)),
                (
                    "capper",
                    OutlierCapper(
                        iqr_multiplier=iqr_multiplier, enabled=outlier_enabled
                    ),
                ),
                ("name_columns", name_source_cols),
            ]
        )

    def _clean_source_columns(
        self, X: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Imputes and caps raw numeric source columns before feature engineering."""
        source_cols = [c for c in self.source_numeric_cols if c in X.columns]
        if not source_cols:
            return X
        if fit:
            cleaned = self.source_cleaner.fit_transform(X[source_cols])
        else:
            cleaned = self.source_cleaner.transform(X[source_cols])
        X_clean = X.copy()
        X_clean[source_cols] = cleaned
        return X_clean

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
        """Creates scikit-learn ColumnTransformer for numerical scaling, imputation, outlier capping, and one-hot encoding of tiers."""
        num_cols = self.config.get("numerical_features", [])
        ord_cols = self.config.get("ordinal_features") or self.config.get(
            "categorical_features", []
        )
        impute_config = self.config.get("imputation", {})
        num_impute_strategy = impute_config.get("numerical_strategy", "median")
        ord_impute_strategy = impute_config.get("ordinal_strategy", "most_frequent")

        outlier_config = self.config.get("outlier_handling", {})
        outlier_enabled = outlier_config.get("enabled", True)
        iqr_multiplier = outlier_config.get("iqr_multiplier", 1.5)

        num_steps = [
            ("imputer", SimpleImputer(strategy=num_impute_strategy)),
            (
                "capper",
                OutlierCapper(
                    iqr_multiplier=iqr_multiplier, enabled=outlier_enabled
                ),
            ),
        ]
        if self.config.get("transformation", {}).get("skew", "none") == "power":
            num_steps.append(
                ("skew", PowerTransformer(standardize=False))
            )
        num_steps.append(("scaler", RobustScaler()))
        num_transformer = Pipeline(steps=num_steps)

        # One-hot instead of ordinal: tier churn is non-monotonic
        # (Basic 0.194 > Premium 0.131 > Pro 0.047), so an ordinal
        # encoding would impose a false ordering.
        ord_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=ord_impute_strategy)),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
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

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """Fits source cleaning, feature engineering, ColumnTransformer, and FeatureSelector pipelines on training data."""
        logger.info(
            f"{GREEN_BOLD}[INFO] Cleaning raw numeric source columns"
            f" (impute + cap) before feature engineering...{CLEAR}"
        )
        X_clean = self._clean_source_columns(X, fit=True)
        logger.info(
            f"{GREEN_BOLD}[INFO] Executing Feature Engineering Pipeline...{CLEAR}"
        )
        X_fe = self.feature_engineering_pipeline.fit_transform(X_clean, y)
        logger.info(
            f"{GREEN_BOLD}[INFO] Fitting ColumnTransformer on training matrix"
            f" (Shape: {X_fe.shape})...{CLEAR}"
        )
        X_trans = self.column_transformer.fit_transform(X_fe, y)

        if self.feature_selector.enabled:
            logger.info(f"{GREEN_BOLD}[INFO] Executing Feature Selection...{CLEAR}")
            X_trans = self.feature_selector.fit_transform(X_trans, y)

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
        X_clean = self._clean_source_columns(X, fit=False)
        X_fe = self.feature_engineering_pipeline.transform(X_clean)
        X_trans = self.column_transformer.transform(X_fe)

        if self.feature_selector.enabled:
            X_trans = self.feature_selector.transform(X_trans)

        return X_trans
