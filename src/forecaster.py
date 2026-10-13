"""
Multi-Step Future Energy Forecaster & Grid Dispatch Simulator
Simulates real-world operational decision making:
1. Loads the champion model.
2. Forecasts the next 72 hours (3 days ahead) into the future.
3. Translates forecasted megawatts into business analytics:
   - Peak Demand Alert Detection
   - Grid Stress & Overload Risk Index
   - Financial Cost Estimation ($/MWh dynamic tariffs)
   - Battery Storage Peak Shaving Recommendation
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.train import FEATURE_COLUMNS


def simulate_future_dispatch(
    history_df: pd.DataFrame,
    model_path: str = "models/best_energy_model.pkl",
    forecast_hours: int = 72,
    output_dir: str = "outputs",
    images_dir: str = "images"
) -> pd.DataFrame:
    """
    Executes a 72-hour recursive multi-step future simulation.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
        
    model = joblib.load(model_path)
    
    # Last known timestamp in dataset
    last_timestamp = history_df["timestamp"].max()
    future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=forecast_hours, freq="h")
    
    # Synthetic weather forecast for the upcoming 72 hours
    np.random.seed(99)
    future_hour = future_timestamps.hour.values
    future_dayofyear = future_timestamps.dayofyear.values
    base_temp = 22.0 + 14.0 * np.sin(2 * np.pi * (future_dayofyear - 100) / 365)
    future_temp = base_temp + 5.0 * np.sin(2 * np.pi * (future_hour - 9) / 24) + np.random.normal(0, 1.2, forecast_hours)
    future_humidity = np.clip(60.0 - 0.7 * (future_temp - 20) + np.random.normal(0, 5.0, forecast_hours), 30.0, 95.0)
    
    # Multi-step autoregressive buffer
    # Keep historical target series in memory to dynamically feed lags
    historical_series = list(history_df["energy_consumption_mwh"].values)
    
    forecasted_energy = []
    
    for i in range(forecast_hours):
        ts = future_timestamps[i]
        curr_hour = ts.hour
        curr_dow = ts.dayofweek
        curr_dom = ts.day
        curr_month = ts.month
        curr_quarter = ts.quarter
        curr_is_weekend = int(curr_dow >= 5)
        
        hour_sin = np.sin(2 * np.pi * curr_hour / 24.0)
        hour_cos = np.cos(2 * np.pi * curr_hour / 24.0)
        month_sin = np.sin(2 * np.pi * curr_month / 12.0)
        month_cos = np.cos(2 * np.pi * curr_month / 12.0)
        
        # Autoregressive lags from current buffer
        lag_1h = historical_series[-1]
        lag_24h = historical_series[-24]
        lag_168h = historical_series[-168] if len(historical_series) >= 168 else historical_series[-24]
        
        # Rolling 24h stats
        recent_24 = historical_series[-24:]
        rolling_mean_24h = np.mean(recent_24)
        rolling_std_24h = np.std(recent_24)
        
        # Build feature vector
        feat_vector = pd.DataFrame([{
            "temperature_celsius": future_temp[i],
            "humidity_percent": future_humidity[i],
            "hour": curr_hour,
            "day_of_week": curr_dow,
            "day_of_month": curr_dom,
            "month": curr_month,
            "quarter": curr_quarter,
            "is_weekend": curr_is_weekend,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "lag_1h": lag_1h,
            "lag_24h": lag_24h,
            "lag_168h": lag_168h,
            "rolling_mean_24h": rolling_mean_24h,
            "rolling_std_24h": rolling_std_24h
        }])[FEATURE_COLUMNS]
        
        pred_val = float(model.predict(feat_vector)[0])
        forecasted_energy.append(round(pred_val, 2))
        historical_series.append(pred_val) # Append for next autoregressive step
        
    # Build Business Simulation DataFrame
    forecast_df = pd.DataFrame({
        "timestamp": future_timestamps,
        "temperature_celsius": np.round(future_temp, 2),
        "forecasted_demand_mwh": forecasted_energy
    })
    
    # Business Logic & Financial Cost Modeling
    # Peak Hours: 14:00 to 20:00 -> On-Peak Tariff ($120 / MWh), Off-Peak ($65 / MWh)
    is_peak_hour = forecast_df["timestamp"].dt.hour.between(14, 20)
    forecast_df["tariff_rate_usd_per_mwh"] = np.where(is_peak_hour, 120.0, 65.0)
    forecast_df["estimated_procurement_cost_usd"] = np.round(
        forecast_df["forecasted_demand_mwh"] * forecast_df["tariff_rate_usd_per_mwh"], 2
    )
    
    # Peak Overload Warning Threshold: 340 MWh
    grid_capacity_threshold = 340.0
    forecast_df["grid_alert_level"] = np.where(
        forecast_df["forecasted_demand_mwh"] > grid_capacity_threshold,
        "HIGH_PEAK_RISK",
        "NORMAL_OPERATING_LOAD"
    )
    
    # Export simulation results
    sim_path = os.path.join(output_dir, "future_72h_energy_forecast_simulation.csv")
    forecast_df.to_csv(sim_path, index=False)
    print(f"[+] Exported 72-Hour Operational Forecast Simulation -> {sim_path}")
    
    # Plot Future Forecast Horizon
    plt.figure(figsize=(14, 5.5))
    plt.plot(forecast_df["timestamp"], forecast_df["forecasted_demand_mwh"], color="#1f77b4", marker="o", markersize=3, linewidth=2, label="Forecasted Demand (MWh)")
    plt.axhline(grid_capacity_threshold, color="red", linestyle="--", linewidth=1.5, label=f"Grid Strain Alert Threshold ({grid_capacity_threshold} MWh)")
    
    plt.fill_between(
        forecast_df["timestamp"],
        grid_capacity_threshold,
        forecast_df["forecasted_demand_mwh"],
        where=(forecast_df["forecasted_demand_mwh"] > grid_capacity_threshold),
        color="red",
        alpha=0.25,
        label="Peak Shaving Trigger Zone"
    )
    
    plt.title("72-Hour Operational Energy Demand Dispatch Simulation", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Timeline (Upcoming 72 Hours)", fontweight="bold")
    plt.ylabel("Forecasted Demand (MWh)", fontweight="bold")
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    chart_path = os.path.join(images_dir, "future_72h_forecast_simulation.png")
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved 72h Forecast Chart -> {chart_path}")
    
    # Summary Insights
    total_mwh = forecast_df["forecasted_demand_mwh"].sum()
    total_cost = forecast_df["estimated_procurement_cost_usd"].sum()
    high_risk_hours = (forecast_df["grid_alert_level"] == "HIGH_PEAK_RISK").sum()
    
    print("\n" + "=" * 60)
    print("      VIRTUAL INDUSTRIAL DISPATCH SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Total Forecasted Energy (72h):  {total_mwh:,.2f} MWh")
    print(f"Estimated Energy Cost:          ${total_cost:,.2f} USD")
    print(f"High-Strain Grid Alert Hours:   {high_risk_hours} hours")
    print("Actionable Strategy:           Dispatch battery energy storage during alert hours.")
    print("=" * 60)
    
    return forecast_df
