"""tests/test_preprocessing.py - Unit tests for DataPreprocessingPipeline components."""

import unittest
import numpy as np
import pandas as pd
from src.preprocessing import DataPreprocessingPipeline, OutlierCapper


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.config = {
            "raw_features": [
                "monthly_spend",
                "number_of_store_visits_per_month",
                "customer_satisfaction_score",
                "number_of_complaints",
                "membership_tier",
                "discount_usage_rate",
            ],
            "numerical_features": [
                "monthly_spend",
                "number_of_store_visits_per_month",
                "customer_satisfaction_score",
                "number_of_complaints",
                "discount_usage_rate",
                "spend_per_visit",
                "complaint_dissatisfaction",
            ],
            "ordinal_features": ["membership_tier"],
            "ordinal_categories": [["Basic", "Pro", "Premium"]],
            "imputation": {
                "numerical_strategy": "median",
                "ordinal_strategy": "most_frequent",
            },
            "outlier_handling": {"enabled": True, "iqr_multiplier": 1.5},
            "target_column": "customer_churn",
        }
        self.raw_data = pd.DataFrame(
            {
                "monthly_spend": [100.0, 200.0, 150.0, np.nan, 1000.0],
                "number_of_store_visits_per_month": [5, 10, 2, 4, 1],
                "customer_satisfaction_score": [0.8, 0.6, 0.9, 0.7, np.nan],
                "number_of_complaints": [1, 0, 3, 2, 8],
                "membership_tier": ["Basic", "Pro", "Premium", np.nan, "Basic"],
                "discount_usage_rate": [0.1, 0.2, 0.05, 0.15, 0.8],
                "customer_churn": [
                    "Retained",
                    "Retained",
                    "Churned",
                    "Retained",
                    "Churned",
                ],
            }
        )

    def test_schema_validation(self):
        pipeline = DataPreprocessingPipeline(self.config)
        self.assertTrue(pipeline.validate_schema(self.raw_data))

    def test_target_processing(self):
        pipeline = DataPreprocessingPipeline(self.config)
        X, y = pipeline.process_target(self.raw_data)
        self.assertNotIn("customer_churn", X.columns)
        self.assertEqual(list(y), [0, 0, 1, 0, 1])

    def test_missing_value_imputation_and_capping(self):
        pipeline = DataPreprocessingPipeline(self.config)
        X, _ = pipeline.process_target(self.raw_data)
        X_trans = pipeline.fit_transform(X)
        self.assertFalse(np.isnan(X_trans).any())
        self.assertEqual(X_trans.shape[0], 5)


if __name__ == "__main__":
    unittest.main()
