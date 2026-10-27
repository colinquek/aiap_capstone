"""test_model_optimization.py - Unit test suite for ModelOptimizer."""

import unittest
import numpy as np
import pandas as pd
from src.model_optimization import ModelOptimizer


class TestModelOptimizer(unittest.TestCase):
    def setUp(self):
        self.config = {
            "random_seed": 42,
            "cv_folds": 3,
            "scoring_metric": "f1",
            "search_type": "grid",
            "n_iter_search": 5,
            "overfitting_threshold": 0.10,
            "learning_curve_train_sizes": [0.5, 1.0],
        }
        self.optimizer = ModelOptimizer(self.config)

        # Synthetic dataset for testing
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        X_mat = np.random.randn(n_samples, n_features)
        y_arr = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

        self.X_train = pd.DataFrame(X_mat, columns=[f"feat_{i}" for i in range(n_features)])
        self.y_train = pd.Series(y_arr, name="target")

        self.X_test = self.X_train.copy()
        self.y_test = self.y_train.copy()

    def test_instantiate_models(self):
        lr = self.optimizer.instantiate_base_model("logistic_regression")
        rf = self.optimizer.instantiate_base_model("random_forest")
        gb = self.optimizer.instantiate_base_model("gradient_boosting")

        self.assertIsNotNone(lr)
        self.assertIsNotNone(rf)
        self.assertIsNotNone(gb)

    def test_tune_logistic_regression(self):
        param_grid = {"C": [0.1, 1.0], "max_iter": [500]}
        best_model, best_params = self.optimizer.tune_model(
            "logistic_regression", self.X_train, self.y_train, param_grid=param_grid
        )
        self.assertIsNotNone(best_model)
        self.assertIn("C", best_params)

    def test_generalization_diagnostics(self):
        param_grid = {"C": [1.0], "max_iter": [500]}
        best_model, _ = self.optimizer.tune_model(
            "logistic_regression", self.X_train, self.y_train, param_grid=param_grid
        )
        eval_dict = self.optimizer.evaluate_cv_baseline(best_model, self.X_train, self.y_train)
        self.assertIn("generalization_gap", eval_dict)
        self.assertIn("overfitting_flag", eval_dict)

    def test_learning_curve(self):
        param_grid = {"C": [1.0], "max_iter": [500]}
        best_model, _ = self.optimizer.tune_model(
            "logistic_regression", self.X_train, self.y_train, param_grid=param_grid
        )
        lc_data = self.optimizer.generate_learning_curve(best_model, self.X_train, self.y_train)
        self.assertIn("train_sizes", lc_data)
        self.assertIn("train_scores_mean", lc_data)
        self.assertIn("val_scores_mean", lc_data)

    def test_evaluate_test_set(self):
        param_grid = {"C": [1.0], "max_iter": [500]}
        best_model, _ = self.optimizer.tune_model(
            "logistic_regression", self.X_train, self.y_train, param_grid=param_grid
        )
        metrics = self.optimizer.evaluate_test_set(best_model, self.X_test, self.y_test)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("roc_auc", metrics)


if __name__ == "__main__":
    unittest.main()
