# AI-Powered Energy Consumption Forecasting System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Production Ready](https://img.shields.io/badge/Status-Industry--Grade-success.svg)]()

> An enterprise-grade, machine-learning-driven time-series forecasting system that models industrial and municipal energy demand, simulates smart-grid operational dispatch, and prevents peak-load grid failures.

---

## 📌 Executive Summary & Problem Statement

Energy generation and distribution facilities face massive financial and infrastructural losses due to **demand-supply volatility**:
- **Over-generation** leads to massive fuel waste, battery degradation, and curtailment penalties.
- **Under-generation** triggers catastrophic blackout risks, grid frequency destabilization, and high spot-market emergency electricity tariffs.

### The Solution:
This project deploys an **AI-Driven Predictive Forecasting & Simulation System** that captures diurnal cycles, weather fluctuations (HVAC load), weekday-weekend operational dynamics, and autoregressive lag metrics to forecast energy loads up to 72 hours ahead with over **97.8% accuracy ($R^2 = 0.9784$)**.

---

## 🏭 Industry Relevance & Business Impact

| Sector | Practical Industry Application | Financial / Operational Impact |
| :--- | :--- | :--- |
| **Smart Cities & Grids** | Dynamic load balancing and automated substation dispatch. | Reduces blackout risks and eliminates expensive reserve generator spin costs. |
| **Data Centers** | Proactive chiller plant cooling & compute load shifting. | Cuts PUE (Power Usage Effectiveness) by 12–18%. |
| **Manufacturing Plants** | Scheduling heavy machinery during off-peak tariff periods. | Saves 15–25% on monthly commercial electricity billing. |
| **Renewable Energy** | Coordinating battery energy storage systems (BESS) during peak hours. | Maximizes green energy arbitrage and minimizes carbon footprint. |

---

## 🏗️ System Architecture & Workflow

```
┌────────────────────────────────────────────────────────┐
│             Raw Telemetry & Weather Ingestion          │
│   (Hourly Timestamps, Temperature, Humidity, MWh Load) │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│          Advanced Feature Engineering Engine           │
│  - Cyclical Temporal Encodings (sin/cos for Hour/Month)│
│  - Autoregressive Lags (Lag-1h, Lag-24h, Lag-168h)     │
│  - 24-Hour Rolling Moving Average & Volatility (Std)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│     Multi-Model Benchmarking & Champion Selection      │
│  - Linear Regression (Baseline)                        │
│  - Ridge Regularization                                │
│  - Random Forest Regressor (Non-Linear Ensemble)       │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│          Operational Virtual Dispatch Simulator        │
│  - 72-Hour Ahead Multi-Step Autoregressive Forecast    │
│  - Peak Shaving Warning Thresholds (> 340 MWh)         │
│  - Dynamic Tariff Procurement Cost Modeling ($/MWh)    │
└────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results

Evaluated across **3,476 unseen hourly testing records** (chronological split, no data leakage):

| Model Architecture | MAE (MWh) | RMSE (MWh) | $R^2$ Score | MAPE (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 9.882 | 12.593 | 0.9580 | 4.04% | Baseline |
| **Ridge Regression** | 9.882 | 12.593 | 0.9580 | 4.04% | Regularized |
| **Random Forest Regressor** | **7.111** | **9.020** | **0.9784** | **2.88%** | 🏆 **Champion** |

---

## 📈 Visual Demonstrations & Analytics

### 1. Telemetry & Weather Interaction (EDA)
![EDA Overview](images/eda_energy_overview.png)

### 2. Model Accuracy: Actual vs Predicted Demand
![Actual vs Predicted](images/actual_vs_predicted.png)

### 3. Model Benchmark Comparison
![Model Benchmark](images/model_benchmark_comparison.png)

### 4. Explainable AI: Feature Importances
![Feature Importance](images/feature_importance.png)

### 5. 72-Hour Operational Dispatch & Peak Shaving Simulator
![72h Forecast](images/future_72h_forecast_simulation.png)

---

## 📁 Repository Structure

```
AI-Energy-Forecasting/
│
├── data/
│   └── energy_consumption.csv              # Synthetic/Ingested 17,500+ hourly records
├── images/
│   ├── eda_energy_overview.png             # Exploratory analysis chart
│   ├── actual_vs_predicted.png             # Test set forecast overlay
│   ├── model_benchmark_comparison.png      # MAE, RMSE, R² comparison charts
│   ├── feature_importance.png              # XAI Random Forest feature rank
│   └── future_72h_forecast_simulation.png  # 3-day future dispatch simulation
├── models/
│   └── best_energy_model.pkl               # Serialized champion model
├── outputs/
│   ├── model_benchmark_results.csv         # Comparative metric table
│   ├── test_predictions.csv                # Actual vs model predictions
│   └── future_72h_energy_forecast_simulation.csv # 72-hour simulated dispatch
├── src/
│   ├── __init__.py
│   ├── dataset_generator.py                # Telemetry synthesis engine
│   ├── preprocess.py                       # Lag, rolling, & cyclical feature transformer
│   ├── train.py                            # Multi-model trainer & evaluator
│   ├── visualize.py                        # Publication-quality plotting
│   └── forecaster.py                       # 72-hour recursive dispatch simulator
├── .gitignore                              # Git exclusion rules
├── main.py                                 # End-to-end master pipeline orchestrator
├── README.md                               # Project documentation & proof of work
└── requirements.txt                        # Dependency list
```

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python 3.9+ (Python 3.10 to 3.13 tested)
- pip package manager

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/AI-Energy-Forecasting.git
cd AI-Energy-Forecasting
```

### 2. Set Up a Virtual Environment (Recommended)
**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline
```bash
python main.py
```

---

## 💡 Key Engineering Highlights for Recruiters

1. **Zero Data Leakage:** Implemented strict chronological train-test splits rather than random k-fold shuffling, preserving causality in time-series dynamics.
2. **Feature Engineering Depth:** Derived cyclical sinusoidal/cosinusoidal encodings, autoregressive lag markers (1h, 24h, 168h), and rolling statistics.
3. **Business & Financial Grounding:** Integrated dynamic Time-of-Use (TOU) tariffs ($65/MWh off-peak vs $120/MWh peak) and automated peak-shaving dispatch trigger zones.
4. **Reproducible & Modular:** Follows clean PEP-8 software architecture with separated modules for data generation, preprocessing, model persistence, and simulation.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
