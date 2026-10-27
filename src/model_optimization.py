"""model_optimization.py - Model Cross-Validation and Hyperparameter Tuning Module."""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
    learning_curve,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

RED_BOLD = "\033[1;31m"
GREEN_BOLD = "\033[1;32m"
CLEAR = "\033[0m"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("model_optimization")


class ModelOptimizer:
    """Class to manage cross-validation, hyperparameter tuning, and model selection."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.random_seed = config.get("random_seed", config.get("random_state", 42))
        self.cv_folds = config.get("cv_folds", 5)
        self.scoring_metric = config.get("scoring_metric", "f1")
        self.search_type = config.get("search_type", "grid")  # 'grid' or 'random'
        self.n_iter_search = config.get("n_iter_search", 10)
        self.overfitting_threshold = config.get("overfitting_threshold", 0.10)

        self.cv_strategy = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
        )
        self.search_results = {}
        self.best_estimator_ = None
        self.best_model_name_ = ""
        self.best_score_ = -1.0

    def get_default_param_grids(self) -> Dict[str, Dict[str, List[Any]]]:
        """Returns candidate hyperparameter search spaces for each model strategy."""
        return {
            "logistic_regression": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
                "class_weight": ["balanced", None],
                "max_iter": [1000],
            },
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 8, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": ["balanced", "balanced_subsample", None],
            },
            "gradient_boosting": {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0],
                "min_samples_split": [2, 5],
            },
        }

    def instantiate_base_model(
        self, model_name: str
    ) -> Union[LogisticRegression, RandomForestClassifier, GradientBoostingClassifier]:
        """Factory method to instantiate un-tuned base estimators."""
        if model_name == "logistic_regression":
            return LogisticRegression(random_state=self.random_seed)
        elif model_name == "random_forest":
            return RandomForestClassifier(random_state=self.random_seed)
        elif model_name == "gradient_boosting":
            return GradientBoostingClassifier(random_state=self.random_seed)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

    def evaluate_cv_baseline(
        self, estimator: Any, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        """Runs Stratified K-Fold Cross-Validation with explicit generalization gap analysis."""
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        }
        cv_results = cross_validate(
            estimator,
            X,
            y,
            cv=self.cv_strategy,
            scoring=scoring,
            return_train_score=True,
        )

        train_f1 = float(np.mean(cv_results["train_f1"]))
        val_f1 = float(np.mean(cv_results["test_f1"]))
        generalization_gap = train_f1 - val_f1
        is_overfitting = generalization_gap > self.overfitting_threshold

        if is_overfitting:
            logger.warning(
                f"{RED_BOLD}[DIAGNOSTIC WARNING] High generalization gap ({generalization_gap:.4f}) "
                f"exceeds threshold ({self.overfitting_threshold}). Model exhibits potential overfitting.{CLEAR}"
            )

        metrics = {
            "val_f1_mean": val_f1,
            "val_f1_std": float(np.std(cv_results["test_f1"])),
            "val_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
            "val_precision_mean": float(np.mean(cv_results["test_precision"])),
            "val_recall_mean": float(np.mean(cv_results["test_recall"])),
            "train_f1_mean": train_f1,
            "generalization_gap": generalization_gap,
            "overfitting_flag": is_overfitting,
        }
        return metrics

    def generate_learning_curve(
        self,
        estimator: Any,
        X: pd.DataFrame,
        y: pd.Series,
        train_sizes: Optional[List[float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Computes learning curve metrics to diagnose bias versus variance dynamics."""
        if train_sizes is None:
            train_sizes = self.config.get(
                "learning_curve_train_sizes", [0.2, 0.4, 0.6, 0.8, 1.0]
            )

        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator,
            X,
            y,
            train_sizes=train_sizes,
            cv=self.cv_strategy,
            scoring=self.scoring_metric,
            n_jobs=-1,
            random_state=self.random_seed,
        )
        return {
            "train_sizes": train_sizes_abs,
            "train_scores_mean": np.mean(train_scores, axis=1),
            "val_scores_mean": np.mean(test_scores, axis=1),
        }

    def tune_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Executes hyperparameter tuning via GridSearchCV or RandomizedSearchCV."""
        base_estimator = self.instantiate_base_model(model_name)
        if param_grid is None:
            param_grid = self.config.get("param_grid", {}).get(model_name)
            if not param_grid:
                param_grid = self.get_default_param_grids().get(model_name, {})

        logger.info(
            f"{GREEN_BOLD}[INFO] Initiating {self.search_type.upper()} search for: {model_name.upper()}{CLEAR}"
        )

        if self.search_type == "random":
            search = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=param_grid,
                n_iter=self.n_iter_search,
                scoring=self.scoring_metric,
                cv=self.cv_strategy,
                random_state=self.random_seed,
                n_jobs=-1,
                refit=True,
            )
        else:
            search = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                scoring=self.scoring_metric,
                cv=self.cv_strategy,
                n_jobs=-1,
                refit=True,
            )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = search.best_score_

        logger.info(
            f"{GREEN_BOLD}[RESULT] Best {self.scoring_metric.upper()} for {model_name}: {best_cv_score:.4f}{CLEAR}"
        )
        logger.info(f"[PARAMS] {best_params}")

        # Compute full evaluation metrics & generalization diagnostics on best model via Stratified CV
        cv_eval = self.evaluate_cv_baseline(best_model, X_train, y_train)
        learning_curve_data = self.generate_learning_curve(best_model, X_train, y_train)

        self.search_results[model_name] = {
            "best_estimator": best_model,
            "best_params": best_params,
            "best_cv_score": best_cv_score,
            "cv_eval": cv_eval,
            "learning_curve": learning_curve_data,
            "search_object": search,
        }

        if best_cv_score > self.best_score_:
            self.best_score_ = best_cv_score
            self.best_estimator_ = best_model
            self.best_model_name_ = model_name

        return best_model, best_params

    def evaluate_test_set(
        self, estimator: Any, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluates model performance on the held-out test set."""
        y_pred = estimator.predict(X_test)
        y_proba = (
            estimator.predict_proba(X_test)[:, 1]
            if hasattr(estimator, "predict_proba")
            else None
        )

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": (
                float(roc_auc_score(y_test, y_proba)) if y_proba is not None else 0.0
            ),
            "pr_auc": (
                float(average_precision_score(y_test, y_proba))
                if y_proba is not None
                else 0.0
            ),
            "log_loss": (
                float(log_loss(y_test, y_proba)) if y_proba is not None else 0.0
            ),
        }
        return metrics

    def optimize_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Runs hyperparameter tuning and cross-validation for all model candidates."""
        model_types = ["logistic_regression", "random_forest", "gradient_boosting"]
        summary_rows = []

        for model_name in model_types:
            best_estimator, best_params = self.tune_model(model_name, X_train, y_train)
            test_metrics = self.evaluate_test_set(best_estimator, X_test, y_test)
            cv_eval = self.search_results[model_name]["cv_eval"]

            summary_rows.append(
                {
                    "Model": model_name,
                    "Best CV F1": self.search_results[model_name]["best_cv_score"],
                    "CV F1 (Mean)": cv_eval["val_f1_mean"],
                    "CV F1 (Std)": cv_eval["val_f1_std"],
                    "Train F1 (Mean)": cv_eval["train_f1_mean"],
                    "Gen Gap": cv_eval["generalization_gap"],
                    "Overfit Alert": cv_eval["overfitting_flag"],
                    "Test Acc": test_metrics["accuracy"],
                    "Test Prec": test_metrics["precision"],
                    "Test Rec": test_metrics["recall"],
                    "Test F1": test_metrics["f1_score"],
                    "Test ROC-AUC": test_metrics["roc_auc"],
                    "Test PR-AUC": test_metrics["pr_auc"],
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        return summary_df

    def save_champion_model(
        self, filepath: str = "models/champion_model.joblib"
    ) -> None:
        """Saves the top-performing champion model artifact to disk."""
        if self.best_estimator_ is None:
            raise ValueError("No champion model found. Run model optimization first.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(
            {
                "model_name": self.best_model_name_,
                "best_estimator": self.best_estimator_,
                "best_cv_score": self.best_score_,
                "best_params": self.search_results[self.best_model_name_][
                    "best_params"
                ],
            },
            filepath,
        )
        logger.info(
            f"{GREEN_BOLD}[INFO] Champion model ({self.best_model_name_}) saved to: {filepath}{CLEAR}"
        )
