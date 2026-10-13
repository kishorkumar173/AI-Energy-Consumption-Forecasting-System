"""
Dataset Generator & Simulation Engine
Simulates realistic hourly energy consumption data for an industrial smart facility/grid.
Incorporates:
- Base diurnal cyclical load (peak hours during day/evening)
- Temperature effects (HVAC cooling load in hot weather, heating load in cold)
- Day of week variations (weekday industrial operations vs weekend baseline)
- Seasonal variations throughout the year
- Random noise and industrial spikes
"""

import os
import numpy as np
import pandas as pd


def generate_energy_dataset(
    start_date: str = "2023-01-01 00:00:00",
    end_date: str = "2024-12-31 23:00:00",
    output_path: str = "data/energy_consumption.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic multi-year hourly energy dataset and saves it to CSV.
    """
    np.random.seed(seed)
    
    # Generate continuous hourly timestamp index
    date_range = pd.date_range(start=start_date, end=end_date, freq="h")
    n = len(date_range)
    
    # 1. Base load
    base_load = 250.0  # Base megawatt-hours (MWh)
    
    # 2. Daily diurnal cycle (24-hour periodicity): peak around 14:00 - 18:00
    hour = date_range.hour.values
    daily_cycle = 85.0 * np.sin(2 * np.pi * (hour - 6) / 24)
    daily_cycle = np.maximum(daily_cycle, -40.0) # avoid dropping too low at night
    
    # 3. Weekly cycle: Industrial weekday activity vs weekend idle
    dayofweek = date_range.dayofweek.values  # 0=Monday, 6=Sunday
    is_weekend = (dayofweek >= 5).astype(float)
    weekday_factor = 1.0 - (0.28 * is_weekend)  # 28% load drop on weekends
    
    # 4. Yearly seasonal cycle (365 days): peak in summer (cooling) and winter (heating)
    dayofyear = date_range.dayofyear.values
    yearly_temp_cycle = 22.0 + 14.0 * np.sin(2 * np.pi * (dayofyear - 100) / 365)
    
    # Temperature hourly variation: warmest at 15:00, coldest at 05:00
    temp_noise = np.random.normal(0, 1.8, n)
    temperature = yearly_temp_cycle + 5.0 * np.sin(2 * np.pi * (hour - 9) / 24) + temp_noise
    
    # HVAC Cooling load (when temp > 24 C) and Heating load (when temp < 15 C)
    cooling_demand = np.maximum(0, temperature - 24) * 5.2
    heating_demand = np.maximum(0, 15 - temperature) * 3.8
    
    # Humidity (%) between 30% and 95%
    humidity = np.clip(60.0 - 0.7 * (temperature - 20) + np.random.normal(0, 7.5, n), 25.0, 98.0)
    
    # 5. Stochastic industrial shock / stochastic variability
    stochastic_noise = np.random.normal(0, 8.5, n)
    
    # Total Energy Consumption (MW)
    energy_consumption = (base_load + daily_cycle + cooling_demand + heating_demand + stochastic_noise) * weekday_factor
    energy_consumption = np.maximum(energy_consumption, 50.0)  # Safety floor
    
    df = pd.DataFrame({
        "timestamp": date_range,
        "temperature_celsius": np.round(temperature, 2),
        "humidity_percent": np.round(humidity, 2),
        "energy_consumption_mwh": np.round(energy_consumption, 2)
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Successfully generated {len(df):,} hourly energy records -> {output_path}")
    return df


if __name__ == "__main__":
    generate_energy_dataset()
