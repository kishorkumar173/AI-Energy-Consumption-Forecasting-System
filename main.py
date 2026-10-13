"""
Master Orchestrator - AI-Powered Energy Consumption Forecasting System
Executes the end-to-end industrial lifecycle:
1. Data Generation / Loading
2. Exploratory Data Analysis & Visualization
3. Advanced Feature Engineering (Cyclical, Lags, Rolling Stats)
4. Model Training & Multi-Model Benchmarking
5. Metric Logging & Visualization
6. 72-Hour Multi-Step Grid Dispatch Simulation & Business Cost Optimization
"""

import sys
import os

# Ensure local modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset_generator import generate_energy_dataset
from src.preprocess import load_raw_data, engineer_features, split_train_test
from src.train import train_and_evaluate_models
from src.visualize import (
    plot_exploratory_data,
    plot_actual_vs_predicted,
    plot_model_benchmarks,
    plot_feature_importance,
)
from src.forecaster import simulate_future_dispatch


def main():
    print("""
======================================================================
     AI-POWERED ENERGY CONSUMPTION FORECASTING SYSTEM
           End-to-End Industry-Grade ML Pipeline
======================================================================
    """)
    
    # ---------------------------------------------------------
    # Step 1: Dataset Generation / Ingestion
    # ---------------------------------------------------------
    raw_data_path = "data/energy_consumption.csv"
    if not os.path.exists(raw_data_path):
        print("[Step 1/6] Synthesizing realistic hourly energy consumption dataset...")
        df_raw = generate_energy_dataset(output_path=raw_data_path)
    else:
        print(f"[Step 1/6] Loading existing telemetry from {raw_data_path}...")
        df_raw = load_raw_data(raw_data_path)
        
    print(f"Loaded {len(df_raw):,} records from {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}.\n")
    
    # ---------------------------------------------------------
    # Step 2: Exploratory Data Visualization
    # ---------------------------------------------------------
    print("[Step 2/6] Generating exploratory data visualization...")
    plot_exploratory_data(df_raw)
    
    # ---------------------------------------------------------
    # Step 3: Feature Engineering
    # ---------------------------------------------------------
    print("\n[Step 3/6] Running feature engineering pipeline...")
    df_featured = engineer_features(df_raw)
    train_df, test_df = split_train_test(df_featured, test_ratio=0.20)
    
    # ---------------------------------------------------------
    # Step 4: Model Training, Evaluation & Benchmarking
    # ---------------------------------------------------------
    print("\n[Step 4/6] Training & benchmarking machine learning architectures...")
    benchmark_results, best_model_name, preds_df = train_and_evaluate_models(train_df, test_df)
    
    # ---------------------------------------------------------
    # Step 5: Post-Training Diagnostic Visualization
    # ---------------------------------------------------------
    print("\n[Step 5/6] Generating analytical performance charts...")
    plot_actual_vs_predicted(preds_df, best_model_name)
    plot_model_benchmarks()
    plot_feature_importance()
    
    # ---------------------------------------------------------
    # Step 6: 72-Hour Future Operational Simulation
    # ---------------------------------------------------------
    print("\n[Step 6/6] Launching 72-Hour Grid Dispatch & Cost Simulation...")
    simulate_future_dispatch(df_raw, forecast_hours=72)
    
    print("""
======================================================================
[SUCCESS] PIPELINE EXECUTION COMPLETED SUCCESSFULLY!
    All models saved in:   ./models/
    All charts saved in:   ./images/
    All logs/CSVs in:      ./outputs/
======================================================================
    """)


if __name__ == "__main__":
    main()
