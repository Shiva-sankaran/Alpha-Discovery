# Alpha Discovery Platform

A modular quantitative research platform for designing, evaluating, and integrating alpha-generating signals. Combines **Rust-powered feature engineering**, **Python backtesting**, and a **FastAPI SDK** to enable scalable experimentation on financial time series.

---

## 🚀 Features

- 🦀 **Rust-powered technical indicators** with blazing fast computation
- 🧠 **Custom signal registration** via Python or FastAPI
- 📈 **Backtesting engine** with volatility targeting, cost modeling, and Sharpe penalization
- 🐘 **PostgreSQL-backed OHLCV storage**
- 💾 **Feather-based feature persistence** for reproducible research
- 🌐 **FastAPI SDK** for plug-and-play strategy integration
- 📊 **Factor analysis** with linear regression, IC, hit rate, and turnover diagnostics

---

## 📁 Directory Structure

```
Alpha_Discovery/
├── api/                        # FastAPI app (signal registration, backtest endpoint)
│   ├── app.py
│   └── models.py
├── sdk/                        # Signal evaluation and backtesting logic
│   ├── backtester.py
│   └── factor_loader.py
├── strategy_cpp/              # C++ strategy module
│   ├── strategy.cpp           # C++ function for computing returns or strategies
│   ├── bindings.cpp           # pybind11 bindings for exposing C++ to Python
│   ├── CMakeLists.txt         # Build configuration for CMake
│   └── build/                 # Compiled artifacts after build
├── rust_factors/              # Rust library for fast technical indicators
│   └── src/
│       └── indicators/*.rs
├── bindings/                  # Python bindings (PyO3/maturin)
│   └── rust_factors.py
├── database/                  # SQL schema for OHLCV ingestion
│   └── init.sql
├── features/                  # Precomputed feather files per symbol
│   └── AAPL.feather, etc.
├── scripts/                   # Data ingestion and feature generation scripts
│   ├── data_ingest.py
│   └── generate_features.py
├── register_signal.py         # Custom user-defined signal functions
├── test_client.py             # Example: register and run backtest
├── main.py                    # Runs alpha analysis on a fixed set of features and thier pair-combinations
├── requirements.txt
├── pyproject.toml             # Rust-Python build config (for maturin)
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Install Requirements

```bash
conda create -n alpha python=3.9
conda activate alpha
pip install -r requirements.txt
```

### 2. Setup PostgreSQL
```bash
psql -U postgres
CREATE DATABASE alpha_db;
CREATE TABLE ohlcv (
    symbol TEXT,
    date DATE,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    PRIMARY KEY (symbol, date)
);
```

### 3. Compile Rust Features

Install Rust if needed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

```bash
cd rust_factors
maturin develop
```

### 3. Compile Rust Features

The C++ module is built using CMake and exposed to Python using pybind11.

Install dependencies:

```bash
sudo apt install cmake build-essential
pip install pybind11

```

```bash
cd strategy_cpp
mkdir build && cd build
cmake ..
make
```

### 4A. Populate PostgreSQL Database with Market 

```bash
python scripts/data_ingest.py
```

This script:

* Connects to alpha_db

* Fetches ~4000 days of data for predefined tickers (e.g., AAPL, MSFT, etc.)

* Populates the ohlcv table


### 4B. Compute Technical Features

The generate_features.py script pulls raw OHLCV from PostgreSQL, applies Rust-accelerated feature computation, and saves to .feather.

Run it with:
```bash
python scripts/generate_features.py
```
This allows the FastAPI SDK and backtester to load precomputed data efficiently.

After running the above scripts, you should see:

```bash
features/
├── AAPL.feather
├── MSFT.feather
└── GOOGL.feather
```

Each file contains:

* open, high, low, close, volume

* Rust-computed features

* Timestamps as the index

### 5. Run FastAPI SDK
```bash
uvicorn api.app:app --reload
```

Endpoints
* POST /register/: Register a signal by name 

* POST /backtest/: Run backtest using pre-registered signal

### 6. Define Your Own Signal

In register_signal.py:
```bash

@register("my_signal")
def my_signal(df):
    return df["momentum_14"] - df["rsi_14"]
```

```bash

curl -X POST http://localhost:8000/backtest/ \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "signal_name": "my_signal"}'
```


📊 Backtester Highlights
* Volatility targeting

* Transaction cost modeling

* Leverage + turnover penalty

* Time-series cross-validation

* Sharpe, IC, hit rate, MSE

🛠 Tech Stack
* Python — FastAPI, pandas, scikit-learn, psycopg2

* Rust — Technical indicators via PyO3

* PostgreSQL — Price + feature storage

* Feather — Fast binary columnar storage

* C++ — Strategy logic via pybind11

📌 TODO / Extensions

* Add Redis caching layer for repeated backtests

* Dockerize full stack

* Add support for fundamental or alternative data

* Web frontend for interactive research


