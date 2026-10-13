"""
Data Preprocessing & Feature Engineering Module
Transforms raw time-series energy consumption records into structured ML-ready feature matrices.
Features created:
1. Calendar / Temporal features (Hour, Day, Month, DayOfWeek, Quarter, IsWeekend)
2. Cyclical trigonometric transformations (Hour sin/cos, Month sin/cos)
3. Lag features (Lag 1h, Lag 24h, Lag 168h / 1 week)
4. Rolling window statistical aggregations (Rolling 24h Mean, Rolling 24h Std)
5. Weather interaction features
"""

import os
import pandas as pd
import numpy as np


def load_raw_data(filepath: str = "data/energy_consumption.csv") -> pd.DataFrame:
    """Loads raw energy consumption data and parses timestamp."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please run dataset_generator.py first.")
    
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts temporal, cyclical, lag, and rolling statistics.
    Returns feature-engineered dataframe ready for ML modeling.
    """
    data = df.copy()
    
    # 1. Date & Time Temporal Features
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["day_of_month"] = data["timestamp"].dt.day
    data["month"] = data["timestamp"].dt.month
    data["quarter"] = data["timestamp"].dt.quarter
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    
    # 2. Cyclical Features (sin/cos encoding for periodic continuity)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)
    
    # 3. Time Series Lag Features (Autoregressive memory)
    # Target is 'energy_consumption_mwh'
    data["lag_1h"] = data["energy_consumption_mwh"].shift(1)       # Previous hour
    data["lag_24h"] = data["energy_consumption_mwh"].shift(24)     # Same hour yesterday
    data["lag_168h"] = data["energy_consumption_mwh"].shift(168)   # Same hour last week
    
    # 4. Rolling Window Moving Aggregations
    # 24-hour moving average and standard deviation (shifted by 1 to prevent data leakage)
    rolling_24 = data["energy_consumption_mwh"].shift(1).rolling(window=24)
    data["rolling_mean_24h"] = rolling_24.mean()
    data["rolling_std_24h"] = rolling_24.std()
    
    # 5. Drop NaN rows created by lag/rolling shifts
    initial_len = len(data)
    data = data.dropna().reset_index(drop=True)
    dropped_count = initial_len - len(data)
    
    print(f"[+] Feature Engineering completed: {data.shape[1]} total columns.")
    print(f"[+] Dropped {dropped_count} initial warm-up rows containing NaNs from lag/rolling windows.")
    return data


def split_train_test(
    data: pd.DataFrame,
    test_ratio: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits time-series data chronologically (WITHOUT random shuffling)
    to strictly prevent future lookahead leakage.
    """
    split_idx = int(len(data) * (1.0 - test_ratio))
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()
    
    print(f"[+] Chronological Train/Test Split:")
    print(f"    - Training set:   {len(train_df):,} samples ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"    - Testing set:    {len(test_df):,} samples ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    return train_df, test_df
