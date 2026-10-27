"""main.py - Main Pipeline Orchestrator for Sprint 3, Sprint 4 & Sprint 5."""

import sys
import os
import yaml
import unittest
import joblib
import pandas as pd
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

    # FIT on Training set first, then TRANSFORM evaluation set
    X_train_processed = pipeline.fit_transform(X_train)
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
    # ---------------------------------------------------------
    print(
        f"\n{GREEN_BOLD}=== EXECUTING SPRINT 5 MODEL HYPERPARAMETER TUNING & CROSS-VALIDATION ==={CLEAR}"
    )
    optimizer = ModelOptimizer(config)
    summary_df = optimizer.optimize_all_models(
        X_train_processed, y_train, X_test_processed, y_test
    )

    print(f"\n{GREEN_BOLD}=== SPRINT 5 COMPREHENSIVE BENCHMARK SUMMARY ==={CLEAR}")
    print(summary_df.to_string(index=False))

    # Save Champion Model Artifact
    optimizer.save_champion_model("models/champion_model.joblib")


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
