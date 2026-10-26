"""main.py - Main Pipeline Orchestrator for Sprint 3."""

import sys
import os
import yaml
import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import DataPreprocessingPipeline, GREEN_BOLD, RED_BOLD, CLEAR


def run_pipeline():
  # Parse CLI argument or fallback to default path
  config_path = (
      sys.argv[1]
      if len(sys.argv) > 1 and sys.argv[1].endswith(('.yaml', '.yml'))
      else './src/config.yaml'
  )

  if not os.path.exists(config_path):
    print(f'{RED_BOLD}[ERROR] Configuration file not found at: {config_path}{CLEAR}')
    sys.exit(1)

  with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

  print(
      f'{GREEN_BOLD}[INFO] Successfully loaded configuration from'
      f' {config_path}{CLEAR}'
  )

  # Ingest Dataset
  data_path = config.get('raw_data_path', './data/synthetic_data_204.csv')
  if not os.path.exists(data_path):
    # Fallback check for root dataset
    data_path = 'synthetic_data_204.csv'

  df_raw = pd.read_csv(data_path)
  print(f'{GREEN_BOLD}[INFO] Ingested raw dataset shape: {df_raw.shape}{CLEAR}')

  # Instantiate Preprocessing Pipeline
  pipeline = DataPreprocessingPipeline(config)

  # Execute Schema Audits & Validation
  pipeline.audit_raw_missingness(df_raw)
  pipeline.validate_schema(df_raw)

  # Process Target
  X, y = pipeline.process_target(df_raw)

  # Stratified Data Split (Preventing Data Leakage)
  seed = config.get('random_seed', 42)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=seed, stratify=y
  )
  print(
      f'{GREEN_BOLD}[INFO] Stratified Split Completed - Train Shape:'
      f' {X_train.shape}, Test Shape: {X_test.shape}{CLEAR}'
  )

  # FIT on Training set first, then TRANSFORM evaluation set
  X_train_processed = pipeline.fit_transform(X_train)
  X_test_processed = pipeline.transform(X_test)

  print(
      f'{GREEN_BOLD}[INFO] Sprint 3 Preprocessing Pipeline executed'
      f' successfully!{CLEAR}'
  )
  print(
      f'{RED_BOLD}[FINDING] Processed Training Matrix Shape:'
      f' {X_train_processed.shape}{CLEAR}'
  )
  print(
      f'{RED_BOLD}[FINDING] Processed Test Matrix Shape:'
      f' {X_test_processed.shape}{CLEAR}'
  )

def run_tests():
    print(f"{GREEN_BOLD}[INFO] Executing Unit Test Suite (tests/)...{CLEAR}")
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='tests')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        print(f"{RED_BOLD}[ERROR] Unit tests failed! Aborting pipeline execution.{CLEAR}")
        sys.exit(1)

if __name__ == '__main__':
  run_tests()  # Run tests first
  run_pipeline() # Run main pipeline logic