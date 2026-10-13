"""
Model Training & Benchmarking Module
Trains, compares, and evaluates multiple regression/time-series architectures:
1. Baseline Linear Regression
2. Ridge Regularized Regression
3. Random Forest Regressor (Non-linear ensemble)

Calculates standard industrial KPIs:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R2 Score (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_COLUMNS = [
    "temperature_celsius",
    "humidity_percent",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_std_24h"
]

TARGET_COLUMN = "energy_consumption_mwh"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes standard evaluation KPIs."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    
    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2_Score": round(r2, 4),
        "MAPE(%)": round(mape, 2)
    }


def train_and_evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models_dir: str = "models",
    outputs_dir: str = "outputs"
) -> tuple[dict, str, pd.DataFrame]:
    """
    Trains models, compares performance on the hold-out test set,
    and serializes the best performing model artifact.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]
    
    candidate_models = {
        "Linear_Regression": LinearRegression(),
        "Ridge_Regression": Ridge(alpha=1.0),
        "Random_Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
    }
    
    benchmark_results = {}
    predictions_dict = {"timestamp": test_df["timestamp"].values, "Actual_MWh": y_test.values}
    
    print("\n" + "=" * 60)
    print("           MODEL TRAINING & BENCHMARKING REPORT")
    print("=" * 60)
    
    best_r2 = -float("inf")
    best_model_name = ""
    best_model_obj = None
    
    for name, model in candidate_models.items():
        print(f"[*] Training {name}...")
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        predictions_dict[f"Pred_{name}"] = preds
        
        metrics = calculate_metrics(y_test, preds)
        benchmark_results[name] = metrics
        
        print(f"    - MAE:      {metrics['MAE']} MWh")
        print(f"    - RMSE:     {metrics['RMSE']} MWh")
        print(f"    - R2 Score: {metrics['R2_Score']}")
        print(f"    - MAPE:     {metrics['MAPE(%)']}%")
        
        if metrics["R2_Score"] > best_r2:
            best_r2 = metrics["R2_Score"]
            best_model_name = name
            best_model_obj = model
            
    print("=" * 60)
    print(f"[SUCCESS] Best Model: {best_model_name} (R^2 = {best_r2})")
    print("=" * 60)
    
    # Save best model to disk
    best_model_path = os.path.join(models_dir, "best_energy_model.pkl")
    joblib.dump(best_model_obj, best_model_path)
    print(f"[+] Serialized champion model to: {best_model_path}")
    
    # Save benchmark table to CSV
    metrics_summary_df = pd.DataFrame(benchmark_results).T
    metrics_summary_path = os.path.join(outputs_dir, "model_benchmark_results.csv")
    metrics_summary_df.to_csv(metrics_summary_path)
    print(f"[+] Exported benchmark KPI matrix to: {metrics_summary_path}")
    
    # Save predictions
    preds_df = pd.DataFrame(predictions_dict)
    preds_csv_path = os.path.join(outputs_dir, "test_predictions.csv")
    preds_df.to_csv(preds_csv_path, index=False)
    print(f"[+] Exported prediction logs to: {preds_csv_path}")
    
    return benchmark_results, best_model_name, preds_df
