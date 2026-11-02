"""main.py - Main Pipeline Orchestrator for Sprint 3, Sprint 4 & Sprint 5."""

import sys
import os
import yaml
import unittest
import joblib
import pandas as pd
from copy import deepcopy
from src.preprocessing import DataPreprocessingPipeline, GREEN_BOLD, RED_BOLD, CLEAR
from src.models import BaselineModel
from src.advanced_models import AdvancedModel
from src.model_optimization import ModelOptimizer
from sklearn.model_selection import train_test_split


def run_pipeline():
    config_path = (
        sys.argv[1]
        if len(sys.argv) > 1 and sys.argv[1].endswith((".yaml", ".yml"))
        else "./src/config.yaml"
    )

    if not os.path.exists(config_path):
        print(
            f"{RED_BOLD}[ERROR] Configuration file not found at: {config_path}{CLEAR}"
        )
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(
        f"{GREEN_BOLD}[INFO] Successfully loaded configuration from {config_path}{CLEAR}"
    )

    # Ingest Dataset
    data_path = config.get("raw_data_path", "./data/synthetic_data_204.csv")
    if not os.path.exists(data_path):
        data_path = "synthetic_data_204.csv"

    df_raw = pd.read_csv(data_path)
    print(f"{GREEN_BOLD}[INFO] Ingested raw dataset shape: {df_raw.shape}{CLEAR}")

    # Instantiate Preprocessing Pipeline
    pipeline = DataPreprocessingPipeline(config)

    # Execute Schema Audits & Validation
    pipeline.audit_raw_missingness(df_raw)
    pipeline.validate_schema(df_raw)

    # Process Target
    X, y = pipeline.process_target(df_raw)

    # Stratified Data Split
    seed = config.get("random_seed", config.get("random_state", 42))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    print(
        f"{GREEN_BOLD}[INFO] Stratified Split Completed - Train Shape: {X_train.shape}, Test Shape: {X_test.shape}{CLEAR}"
    )

    # Pass y_train into fit_transform for target-aware feature selection
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    X_test_processed = pipeline.transform(X_test)

    print(f"{GREEN_BOLD}[INFO] Preprocessing Pipeline executed successfully!{CLEAR}")

    # Save Preprocessing Pipeline Artifact
    os.makedirs("models", exist_ok=True)
    preprocessor_path = "models/preprocessing_pipeline.joblib"
    joblib.dump(pipeline, preprocessor_path)
    print(
        f"{GREEN_BOLD}[INFO] Saved fitted preprocessor to: {preprocessor_path}{CLEAR}"
    )

    # ---------------------------------------------------------
    # Sprint 5: Model Optimization & Hyperparameter Tuning
    # Variants (all on the SAME 80/20 split) to isolate the value of each
    # pipeline stage:
    #   A: full pipeline (source cleaning + engineered features + selection)
    #   B: no engineered features (raw numerics only, identity feature step)
    #   C: no feature selection (full feature matrix, SelectKBest disabled)
    # ---------------------------------------------------------
    from sklearn.pipeline import Pipeline as _SkPipeline
    from sklearn.preprocessing import FunctionTransformer as _IdentityTransformer

    print(
        f"\n{GREEN_BOLD}=== EXECUTING SPRINT 5 MODEL HYPERPARAMETER TUNING & CROSS-VALIDATION ==={CLEAR}"
    )

    def _build_variant(variant_id):
        """Build and fit the preprocessing pipeline for one comparison variant."""
        variant_config = deepcopy(config)
        if variant_id == "B":
            # No engineered features: only raw numerics, and the feature
            # engineering step becomes a no-op passthrough (so the derived
            # columns do not leak in as raw passthrough columns).
            derived = pipeline.derived_features
            variant_config["numerical_features"] = [
                col for col in variant_config.get("numerical_features", [])
                if col not in derived
            ]
        elif variant_id == "C":
            # No feature selection: keep the full feature matrix.
            variant_config["feature_selection"]["enabled"] = False

        variant_pipe = DataPreprocessingPipeline(variant_config)
        if variant_id == "B":
            variant_pipe.feature_engineering_pipeline = _SkPipeline(
                [("identity", _IdentityTransformer())]
            )
        Xv, yv = variant_pipe.process_target(df_raw)
        Xv_train, Xv_test, yv_train, yv_test = train_test_split(
            Xv, yv, test_size=0.2, random_state=seed, stratify=yv
        )
        Xv_train = variant_pipe.fit_transform(Xv_train, yv_train)
        Xv_test = variant_pipe.transform(Xv_test)
        print(
            f"{GREEN_BOLD}[INFO] Variant {variant_id}: "
            f"train shape {Xv_train.shape}, test shape {Xv_test.shape}{CLEAR}"
        )
        return Xv_train, Xv_test, yv_train, yv_test

    variant_rows = []
    for variant_id, variant_desc in [
        ("A", "full pipeline: cleaning + engineered features + selection"),
        ("B", "no engineered features (raw numerics only)"),
        ("C", "no feature selection (all features kept)"),
    ]:
        print(
            f"\n{GREEN_BOLD}{'=' * 70}\n"
            f"=== VARIANT {variant_id}: {variant_desc.upper()} ===\n"
            f"{'=' * 70}{CLEAR}"
        )
        Xv_train, Xv_test, yv_train, yv_test = _build_variant(variant_id)
        optimizer = ModelOptimizer(config)
        summary_df = optimizer.optimize_all_models(
            Xv_train, yv_train, Xv_test, yv_test
        )
        print(f"\n=== VARIANT {variant_id} BENCHMARK SUMMARY ===\n")
        print(summary_df.to_string(index=False))
        for _, row in summary_df.iterrows():
            variant_rows.append(
                {
                    "Variant": variant_id,
                    "Model": row["Model"],
                    "Best CV F1": row["Best CV F1"],
                    "Test F1": row["Test F1"],
                }
            )
        if variant_id == "A":
            champion_optimizer = optimizer
            summary_a_df = summary_df

    # Displaying the comprehensive benchmark summary (variant A: full pipeline)
    print(f"\n{GREEN_BOLD}=== SPRINT 5 COMPREHENSIVE BENCHMARK SUMMARY (VARIANT A) ==={CLEAR}")
    print(summary_a_df.to_string(index=False))

    # Combined variant comparison (test F1 on the same 80/20 split)
    comparison_df = pd.DataFrame(variant_rows).pivot(
        index="Model", columns="Variant", values="Test F1"
    )
    comparison_df.columns = [f"Variant {c}" for c in comparison_df.columns]
    comparison_df = comparison_df.reindex(
        ["logistic_regression", "random_forest", "gradient_boosting"]
    )
    print(f"\n{GREEN_BOLD}=== VARIANT COMPARISON (TEST F1, SAME 80/20 SPLIT) ==={CLEAR}\n")
    print(comparison_df.to_string())

    # Save Champion Model Artifact (variant A: full pipeline)
    champion_optimizer.save_champion_model("models/champion_model.joblib")


def run_tests():
    print(f"{GREEN_BOLD}[INFO] Executing Unit Test Suite (tests/)...{CLEAR}")
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        print(
            f"{RED_BOLD}[ERROR] Unit tests failed! Aborting pipeline execution.{CLEAR}"
        )
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
    run_pipeline()
