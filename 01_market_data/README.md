# 01 — Market Data

This folder focus on collecting market data, exploring it, and building a simple yield curve — essential steps before pricing, sensitivities, VaR/ES or stress testing.

---

## 📊 1. Equity & FX Data Exploration
**Notebook:** `01_equity_fx_data_exploration.ipynb`

This notebook covers:

- Downloading daily prices for major **equity indices** and **FX pairs** using `yfinance`
- Cleaning and preparing price series
- Computing:
- log-returns
- daily & annualized volatility
- correlation matrix
- Visualisations:
- price evolution
- cumulative returns
- rolling volatility
- rolling correlations
- A small **risk snapshot table** (mean return, annual vol, asset class)
- Short comments explaining results from a **Market Risk perspective**
(volatility spikes, rising correlations in crises, diversification effects)

---

## 📈 2. Yield Curve Bootstrapping
**Notebook:** `02_yield_curve_bootstrap.ipynb`

This notebook includes:

- A small set of rate instruments (deposits + swaps)
- A simple **bootstrapping** procedure to compute:
- discount factors
- zero-coupon rates
- Plots of:
- the zero-coupon curve
- discount factors
- Short explanations on how yield curves are used in:
- pricing
- DV01 / interest rate sensitivities
- scenario analysis

---

## 🔧 3. Utility Functions
**File:** `utils_market_data.py`

This file centralises helper functions used across notebooks:

- downloading market data
- computing log-returns
- computing annualized volatility

This keeps notebooks clean and focused on interpretation.

---

## 💬 Feedback

This project is a learning journey.
If you have suggestions or want to contribute improvements, feel free to reach out or open an issue.