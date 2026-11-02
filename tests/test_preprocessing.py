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
                "complaint_risk_band",
                "visits_risk_band",
                "discount_dependency",
                "high_discount_flag",
                "value_tier_mismatch",
            ],
            "ordinal_features": ["membership_tier"],
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
        X, y = pipeline.process_target(self.raw_data)
        X_trans = pipeline.fit_transform(X, y)
        self.assertFalse(np.isnan(X_trans).any())
        self.assertEqual(X_trans.shape[0], 5)

    def test_engineered_features_use_imputed_sources(self):
        """Focused check (item 2): derived features are computed from imputed
        (and capped) source values, not NaNs — assert the engineered feature
        values themselves rather than only the final transformed matrix."""
        pipeline = DataPreprocessingPipeline(self.config)
        X, y = pipeline.process_target(self.raw_data)
        pipeline.fit_transform(X, y)

        src = pipeline.source_numeric_cols
        cleaned = pipeline.source_cleaner.transform(X[src])
        X_clean = X.copy()
        X_clean[src] = cleaned
        X_fe = pipeline.feature_engineering_pipeline.transform(X_clean)

        # No NaN may leak into any derived feature.
        derived = [
            "spend_per_visit",
            "complaint_dissatisfaction",
            "complaint_risk_band",
            "visits_risk_band",
            "discount_dependency",
            "high_discount_flag",
            "value_tier_mismatch",
        ]
        self.assertFalse(X_fe[derived].isnull().any().any())

        # Row 3 had NaN spend -> imputed to median [100,150,200,1000]=175, no cap.
        self.assertAlmostEqual(float(cleaned["monthly_spend"].iloc[3]), 175.0)
        # spend_per_visit[3] = 175 / (4 + eps) — would be NaN before the reorder.
        self.assertAlmostEqual(
            float(X_fe["spend_per_visit"].iloc[3]), 175.0 / (4.0 + 1e-5)
        )
        # complaint_dissatisfaction[3] = 2 * (1 - 0.7)
        self.assertAlmostEqual(float(X_fe["complaint_dissatisfaction"].iloc[3]), 0.6)

        # Row 4 had NaN satisfaction -> imputed to median [0.8,0.6,0.9,0.7]=0.75.
        # complaints 8 is capped to 6 and discount 0.8 to 0.35 by the source capper.
        self.assertAlmostEqual(
            float(cleaned["customer_satisfaction_score"].iloc[4]), 0.75
        )
        self.assertAlmostEqual(float(cleaned["number_of_complaints"].iloc[4]), 6.0)
        self.assertAlmostEqual(float(cleaned["discount_usage_rate"].iloc[4]), 0.35)
        # complaint_dissatisfaction[4] = 6 * (1 - 0.75) = 1.5 — not NaN.
        self.assertAlmostEqual(float(X_fe["complaint_dissatisfaction"].iloc[4]), 1.5)
        # complaints 6 is not < 3 -> risk band 0; visits 1 is not >= 15 -> 0.
        self.assertEqual(float(X_fe["complaint_risk_band"].iloc[4]), 0.0)
        self.assertEqual(float(X_fe["visits_risk_band"].iloc[4]), 0.0)
        # discount_dependency[4] = 0.35 * 275 (spend 1000 capped to 275)
        self.assertAlmostEqual(float(X_fe["discount_dependency"].iloc[4]), 96.25)
        self.assertEqual(float(X_fe["high_discount_flag"].iloc[4]), 0.0)


if __name__ == "__main__":
    unittest.main()
