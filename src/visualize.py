"""
Visualization and Analytics Reporting Module
Generates high-resolution production charts:
1. Hourly Energy Time Series Overview (EDA)
2. Actual vs Predicted Comparison Plot (Test Horizon)
3. Model Benchmark Bar Chart (MAE, RMSE, R2)
4. Feature Importance Ranking Plot (Explainable AI)
5. 72-Hour Future Operational Energy Dispatch Forecast
"""

import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.train import FEATURE_COLUMNS


# Configure Matplotlib styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 150


def plot_exploratory_data(df: pd.DataFrame, save_path: str = "images/eda_energy_overview.png"):
    """Visualizes 30-day snapshot of consumption along with ambient temperature."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sample_df = df.iloc[:24 * 30].copy()  # First 30 days
    
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()
    
    ax1.plot(sample_df["timestamp"], sample_df["energy_consumption_mwh"], color="#1f77b4", label="Energy Demand (MWh)", linewidth=1.5)
    ax2.plot(sample_df["timestamp"], sample_df["temperature_celsius"], color="#ff7f0e", linestyle="--", label="Temperature (°C)", alpha=0.7)
    
    ax1.set_xlabel("Date & Time", fontweight="bold")
    ax1.set_ylabel("Energy Consumption (MWh)", color="#1f77b4", fontweight="bold")
    ax2.set_ylabel("Ambient Temperature (°C)", color="#ff7f0e", fontweight="bold")
    
    plt.title("Industrial Energy Telemetry & Ambient Weather Interaction (30-Day Sample)", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved EDA plot -> {save_path}")


def plot_actual_vs_predicted(preds_df: pd.DataFrame, best_model_name: str, save_path: str = "images/actual_vs_predicted.png"):
    """Plots actual test energy profile vs predicted profile over a 14-day window."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sample_window = preds_df.iloc[:24 * 14].copy()  # 14 days
    
    plt.figure(figsize=(14, 5.5))
    plt.plot(sample_window["timestamp"], sample_window["Actual_MWh"], label="Actual Grid Demand (MWh)", color="#2ca02c", linewidth=2.2, alpha=0.9)
    plt.plot(sample_window["timestamp"], sample_window[f"Pred_{best_model_name}"], label=f"AI Forecast ({best_model_name})", color="#d62728", linestyle="--", linewidth=2)
    
    plt.title(f"Model Forecasting Precision: Actual vs Predicted Demand (14-Day Test Horizon)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Timeline", fontweight="bold")
    plt.ylabel("Energy Load (MWh)", fontweight="bold")
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Actual vs Predicted plot -> {save_path}")


def plot_model_benchmarks(metrics_path: str = "outputs/model_benchmark_results.csv", save_path: str = "images/model_benchmark_comparison.png"):
    """Plots comparative evaluation metrics across all candidate models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.read_csv(metrics_path, index_col=0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    models = df.index
    
    # 1. MAE
    axes[0].bar(models, df["MAE"], color=["#4e79a7", "#f28e2b", "#59a14f"], edgecolor="black", alpha=0.85)
    axes[0].set_title("Mean Absolute Error (MAE - Lower is Better)", fontweight="bold")
    axes[0].set_ylabel("MWh")
    for i, v in enumerate(df["MAE"]):
        axes[0].text(i, v + 0.3, str(v), ha="center", fontweight="bold")
        
    # 2. RMSE
    axes[1].bar(models, df["RMSE"], color=["#4e79a7", "#f28e2b", "#59a14f"], edgecolor="black", alpha=0.85)
    axes[1].set_title("Root Mean Squared Error (RMSE - Lower is Better)", fontweight="bold")
    axes[1].set_ylabel("MWh")
    for i, v in enumerate(df["RMSE"]):
        axes[1].text(i, v + 0.3, str(v), ha="center", fontweight="bold")
        
    # 3. R2 Score
    axes[2].bar(models, df["R2_Score"], color=["#4e79a7", "#f28e2b", "#59a14f"], edgecolor="black", alpha=0.85)
    axes[2].set_title("R² Goodness of Fit (Higher is Better)", fontweight="bold")
    axes[2].set_ylabel("Score (0 to 1.0)")
    axes[2].set_ylim(0, 1.1)
    for i, v in enumerate(df["R2_Score"]):
        axes[2].text(i, v + 0.03, str(v), ha="center", fontweight="bold")
        
    plt.suptitle("Comparative Evaluation Across Machine Learning Architectures", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Benchmark comparison plot -> {save_path}")


def plot_feature_importance(model_path: str = "models/best_energy_model.pkl", save_path: str = "images/feature_importance.png"):
    """Extracts and plots feature importances from the champion model."""
    if not os.path.exists(model_path):
        return
    
    model = joblib.load(model_path)
    if not hasattr(model, "feature_importances_"):
        return
    
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": importances}).sort_values("Importance", ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feat_df["Feature"], feat_df["Importance"], color="#337ab7", edgecolor="black", alpha=0.85)
    plt.title("Explainable AI: Random Forest Feature Importances", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Normalized Importance", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Feature Importance plot -> {save_path}")
