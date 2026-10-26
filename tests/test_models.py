"""tests/test_models.py - Unit tests for BaselineModel and AdvancedModel architectures."""

import unittest
import numpy as np
import pandas as pd
from src.models import BaselineModel
from src.advanced_models import AdvancedModel


class TestModelArchitectures(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.X_train = np.random.randn(100, 10)
        self.y_train = pd.Series(np.random.choice([0, 1], size=100))
        self.X_test = np.random.randn(20, 10)
        self.y_test = pd.Series(np.random.choice([0, 1], size=20))

    def test_baseline_dummy_fit_predict(self):
        model = BaselineModel(model_type="dummy", random_state=42)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), 20)

    def test_baseline_logistic_regression_eval(self):
        model = BaselineModel(model_type="logistic_regression", random_state=42)
        model.fit(self.X_train, self.y_train)
        metrics = model.evaluate(self.X_test, self.y_test)
        self.assertIn("roc_auc", metrics)
        self.assertIn("pr_auc", metrics)

    def test_advanced_random_forest(self):
        model = AdvancedModel(model_type="random_forest", random_state=42)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        metrics = model.evaluate(self.X_test, self.y_test)
        self.assertEqual(len(preds), 20)
        self.assertIsNotNone(metrics.get("f1_score"))

    def test_advanced_gradient_boosting(self):
        model = AdvancedModel(model_type="gradient_boosting", random_state=42)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        metrics = model.evaluate(self.X_test, self.y_test)
        self.assertEqual(len(preds), 20)
        self.assertIsNotNone(metrics.get("pr_auc"))


if __name__ == "__main__":
    unittest.main()
