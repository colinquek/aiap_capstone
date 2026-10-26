"""main.py - Main Pipeline Orchestrator for Sprint 3 & Sprint 4."""

import sys
import os
import yaml
import unittest
import joblib
import pandas as pd
from src.preprocessing import DataPreprocessingPipeline, GREEN_BOLD, RED_BOLD, CLEAR
from src.models import BaselineModel
from src.advanced_models import AdvancedModel
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
        f"{GREEN_BOLD}[INFO] Successfully loaded configuration from"
        f" {config_path}{CLEAR}"
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
        f"{GREEN_BOLD}[INFO] Stratified Split Completed - Train Shape:"
        f" {X_train.shape}, Test Shape: {X_test.shape}{CLEAR}"
    )

    # FIT on Training set first, then TRANSFORM evaluation set
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    print(f"{GREEN_BOLD}[INFO] Preprocessing Pipeline executed successfully!{CLEAR}")

    # ---------------------------------------------------------
    # Save Preprocessing Pipeline Artifact
    # ---------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    preprocessor_path = "models/preprocessing_pipeline.joblib"
    joblib.dump(pipeline, preprocessor_path)
    print(
        f"{GREEN_BOLD}[INFO] Saved fitted preprocessor to: {preprocessor_path}{CLEAR}"
    )

    # ---------------------------------------------------------
    # Sprint 4: Comprehensive Multi-Model Benchmark
    # ---------------------------------------------------------
    model_candidates = [
        ("dummy", "baseline", BaselineModel(model_type="dummy", random_state=seed)),
        (
            "logistic_regression",
            "baseline",
            BaselineModel(model_type="logistic_regression", random_state=seed),
        ),
        (
            "random_forest",
            "advanced",
            AdvancedModel(model_type="random_forest", random_state=seed),
        ),
        (
            "gradient_boosting",
            "advanced",
            AdvancedModel(model_type="gradient_boosting", random_state=seed),
        ),
    ]

    benchmark_results = []
    best_model = None
    best_f1 = -1.0
    best_model_name = ""

    print(
        f"\n{GREEN_BOLD}=== EXECUTING SPRINT 4 COMPREHENSIVE MODEL BENCHMARK ==={CLEAR}"
    )

    for name, m_category, model_obj in model_candidates:
        print(
            f"{GREEN_BOLD}[INFO] Fitting & Evaluating Strategy: {name.upper()} ({m_category.upper()}){CLEAR}"
        )
        model_obj.fit(X_train_processed, y_train)

        train_metrics = model_obj.evaluate(X_train_processed, y_train)
        test_metrics = model_obj.evaluate(X_test_processed, y_test)

        test_f1 = test_metrics.get("f1_score", 0.0) or 0.0
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model = model_obj
            best_model_name = name

        benchmark_results.append(
            {
                "Model": name,
                "Category": m_category,
                "Test Acc": test_metrics.get("accuracy"),
                "Test Prec": test_metrics.get("precision"),
                "Test Rec": test_metrics.get("recall"),
                "Test F1": test_metrics.get("f1_score"),
                "Test ROC-AUC": test_metrics.get("roc_auc"),
                "Test PR-AUC": test_metrics.get("pr_auc"),
                "Test LogLoss": test_metrics.get("log_loss"),
            }
        )

        if name == "logistic_regression":
            model_obj.save_model("models/baseline_model.joblib")

    # Display Side-by-Side Benchmark Summary
    print(f"\n{GREEN_BOLD}=== SPRINT 4 COMPREHENSIVE BENCHMARK COMPARISON ==={CLEAR}")
    df_benchmark = pd.DataFrame(benchmark_results)
    print(df_benchmark.to_string(index=False))

    # Save Champion Model Artifact
    if best_model is not None:
        champion_path = f"models/champion_{best_model_name}.joblib"
        best_model.save_model(champion_path)
        best_model.save_model("models/advanced_model.joblib")
        print(
            f"\n{GREEN_BOLD}[CHAMPION] Top Performing Model: {best_model_name.upper()} (F1: {best_f1}){CLEAR}"
        )
        print(
            f"{GREEN_BOLD}[INFO] Saved champion model artifact to: {champion_path}{CLEAR}"
        )


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
