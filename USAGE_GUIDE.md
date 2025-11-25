# Market Risk Analytics - Usage Guide

This guide provides detailed instructions on how to use the Market Risk Analytics library.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Market Data Module](#market-data-module)
4. [Yield Curve Construction](#yield-curve-construction)
5. [Pricing & Sensitivities](#pricing--sensitivities)
6. [Risk Metrics (VaR & ES)](#risk-metrics-var--es)
7. [Backtesting](#backtesting)
8. [Stress Testing](#stress-testing)
9. [Portfolio Risk Aggregation](#portfolio-risk-aggregation)
10. [Streamlit Dashboard](#streamlit-dashboard)

## Installation

### Clone and Install
```bash
git clone https://github.com/rokhaya-sonko/market-risk-analytics.git
cd market-risk-analytics
pip install -r requirements.txt
```

### Install as Package
```bash
pip install -e .
```

## Quick Start

Run the example script:
```bash
python examples/basic_usage.py
```

Launch the dashboard:
```bash
streamlit run app.py
```

## Market Data Module

### Working with Stock Data

```python
from src.market_risk.data import StockData

# Create stock data (synthetic data for demonstration)
stock = StockData("AAPL")

# Get returns
returns = stock.get_returns(method='simple')  # or 'log'

# Calculate volatility
volatility = stock.get_volatility()  # Annualized
print(f"Annual Volatility: {volatility:.2%}")
```

### Working with FX Data

```python
from src.market_risk.data import FXData

# Create FX data
fx = FXData("EUR/USD")

# Get rate history
rates = fx.data
returns = fx.get_returns()
```

### Working with Interest Rates

```python
from src.market_risk.data import RatesData

# Create rates term structure
rates = RatesData("USD")

# Get specific rate
five_year_rate = rates.get_zero_rate(5.0)
print(f"5Y Rate: {five_year_rate:.4%}")
```

## Yield Curve Construction

### Creating a Yield Curve

```python
from src.market_risk.curves import YieldCurve

# Define maturities and rates
maturities = [0.25, 0.5, 1, 2, 5, 10, 30]
rates = [0.02, 0.022, 0.025, 0.028, 0.032, 0.035, 0.038]

# Create curve
curve = YieldCurve(maturities, rates)
```

### Getting Discount Factors

```python
# Get discount factor for any maturity
df_3y = curve.get_discount_factor(3.0)
print(f"3Y Discount Factor: {df_3y:.6f}")
```

### Calculating Forward Rates

```python
# Forward rate between 1Y and 2Y
forward = curve.get_forward_rate(1.0, 2.0)
print(f"1Y-2Y Forward Rate: {forward:.4%}")
```

### Bootstrapping from Market Data

```python
# From bond prices
bond_data = [
    (1.0, 3.0, 100.5),   # (maturity, coupon, price)
    (2.0, 3.5, 101.2),
    (5.0, 4.0, 103.0)
]
curve = YieldCurve.bootstrap_from_bonds(bond_data)

# From swap rates
swap_maturities = [1, 2, 5, 10]
swap_rates = [0.025, 0.028, 0.032, 0.035]
curve = YieldCurve.from_swap_rates(swap_maturities, swap_rates)
```

## Pricing & Sensitivities

### Bond Pricing

```python
from src.market_risk.pricing import BondPricer

# Create pricer with yield curve
pricer = BondPricer(curve)

# Get all sensitivities
sensitivities = pricer.get_all_sensitivities(
    maturity=5.0,
    coupon_rate=0.04,
    face_value=100,
    frequency=2  # Semi-annual
)

print(f"Price: {sensitivities['price']:.2f}")
print(f"DV01: {sensitivities['dv01']:.4f}")
print(f"Duration: {sensitivities['duration']:.2f}")
print(f"Convexity: {sensitivities['convexity']:.4f}")
```

### Individual Calculations

```python
# Just price
price = pricer.price_bond(5.0, 0.04, 100)

# Just DV01
dv01 = pricer.calculate_dv01(5.0, 0.04, 100)

# Just Duration
duration = pricer.calculate_duration(5.0, 0.04, 100)
```

### Option Pricing (Black-Scholes)

```python
from src.market_risk.pricing import OptionPricer

# Create option pricer
option = OptionPricer(
    spot=100,
    strike=105,
    maturity=1.0,
    volatility=0.25,
    risk_free_rate=0.03,
    dividend_yield=0.02
)

# Get all Greeks for a call
call_greeks = option.get_all_greeks('call')
print(f"Call Price: {call_greeks['price']:.4f}")
print(f"Delta: {call_greeks['delta']:.4f}")
print(f"Gamma: {call_greeks['gamma']:.6f}")
print(f"Vega: {call_greeks['vega']:.4f}")
print(f"Theta: {call_greeks['theta']:.4f}")
print(f"Rho: {call_greeks['rho']:.4f}")

# Get Greeks for a put
put_greeks = option.get_all_greeks('put')
```

### Individual Greeks

```python
# Just delta
delta = option.delta('call')

# Just gamma (same for calls and puts)
gamma = option.gamma()

# Vega, Theta, Rho
vega = option.vega()
theta = option.theta('call')
rho = option.rho('put')
```

## Risk Metrics (VaR & ES)

### Calculating VaR

```python
from src.market_risk.risk import VaRCalculator
import pandas as pd

# Assume you have returns data
returns = pd.Series([...])  # Your returns

# Create calculator
var_calc = VaRCalculator(returns)

# Parametric VaR
param_var = var_calc.parametric_var(
    confidence_level=0.95,
    horizon=1
)
print(f"1-Day VaR (95%): {param_var:.4%}")

# Historical VaR
hist_var = var_calc.historical_var(
    confidence_level=0.95,
    horizon=1
)
print(f"Historical VaR: {hist_var:.4%}")
```

### Calculating Expected Shortfall

```python
# Parametric ES
param_es = var_calc.parametric_es(
    confidence_level=0.95,
    horizon=1
)

# Historical ES
hist_es = var_calc.historical_es(
    confidence_level=0.95,
    horizon=1
)
```

### All Metrics at Once

```python
metrics = var_calc.calculate_all_metrics(
    confidence_level=0.95,
    horizon=10  # 10-day horizon
)

print(f"Parametric VaR: {metrics['parametric_var']:.4%}")
print(f"Historical VaR: {metrics['historical_var']:.4%}")
print(f"Parametric ES: {metrics['parametric_es']:.4%}")
print(f"Historical ES: {metrics['historical_es']:.4%}")
```

## Backtesting

### Basic Backtesting

```python
from src.market_risk.risk import Backtester

# You need actual returns and VaR forecasts
returns = pd.Series([...])
var_forecasts = pd.Series([...])  # Historical VaR predictions

# Create backtester
backtester = Backtester(returns, var_forecasts)

# Get summary
results = backtester.get_backtest_summary(confidence_level=0.95)

print(f"Violations: {results['n_violations']}")
print(f"Violation Rate: {results['violation_rate']:.2%}")
print(f"Expected Violations: {results['expected_violations']:.1f}")
print(f"Violation Ratio: {results['violation_ratio']:.2f}")
```

### Statistical Tests

```python
# Kupiec test
kupiec_stat, kupiec_pval, kupiec_reject = backtester.kupiec_test(0.95)
print(f"Kupiec p-value: {kupiec_pval:.4f}")
if kupiec_reject:
    print("Reject null hypothesis (model is inadequate)")
else:
    print("Do not reject null hypothesis (model is adequate)")

# Christoffersen test
christ_stat, christ_pval, christ_reject = backtester.christoffersen_test()
print(f"Christoffersen p-value: {christ_pval:.4f}")
```

## Stress Testing

### Using Predefined Scenarios

```python
from src.market_risk.stress_tests import ScenarioAnalyzer

# Create analyzer
analyzer = ScenarioAnalyzer()

# Apply a scenario
result = analyzer.apply_scenario(
    portfolio_value=1000000,
    equity_exposure=500000,
    fx_exposure=100000,
    rates_exposure=-1000,  # DV01
    vol_exposure=500,      # Vega
    scenario_name='market_crash'
)

print(f"Scenario: {result['scenario']}")
print(f"Total P&L: ${result['total_pnl']:,.0f}")
print(f"P&L %: {result['pnl_percent']:.2f}%")
print(f"Stressed Value: ${result['stressed_value']:,.0f}")
```

### Available Scenarios

- `market_crash`: Severe market crash (-30% equities)
- `interest_rate_rise`: Sharp rate increase (+200 bps)
- `flight_to_quality`: Flight to quality scenario
- `inflation_spike`: Unexpected inflation
- `credit_crisis`: Credit market crisis
- `emerging_markets_crisis`: EM crisis

### Running All Scenarios

```python
results_df = analyzer.run_all_scenarios(
    portfolio_value=1000000,
    equity_exposure=500000,
    fx_exposure=100000,
    rates_exposure=-1000,
    vol_exposure=500
)

print(results_df)
```

### Custom Scenarios

```python
# Define your own scenario
analyzer.add_custom_scenario(
    name='custom_shock',
    description='Custom market shock',
    equity_shock=-0.15,
    fx_shock=0.08,
    rates_shock=0.01,
    vol_shock=0.12
)

# Apply it
result = analyzer.apply_scenario(
    portfolio_value=1000000,
    equity_exposure=500000,
    fx_exposure=100000,
    rates_exposure=-1000,
    vol_exposure=500,
    scenario_name='custom_shock'
)
```

### Reverse Stress Testing

```python
# Find scenarios that would cause a specific loss
reverse_result = analyzer.reverse_stress_test(
    portfolio_value=1000000,
    loss_threshold=100000,  # $100k loss
    equity_exposure=500000,
    fx_exposure=100000,
    rates_exposure=-1000
)

print(f"Required equity shock: {reverse_result['required_equity_shock_pct']:.2f}%")
```

## Portfolio Risk Aggregation

### Basic Portfolio Risk

```python
from src.market_risk.portfolio import PortfolioRisk
import numpy as np

# Create portfolio with returns
returns_df = pd.DataFrame({
    'Stock1': [...],
    'Stock2': [...],
    'Bond1': [...]
})

weights = np.array([0.4, 0.4, 0.2])

# Create portfolio
portfolio = PortfolioRisk(returns_df, weights)

print(f"Portfolio Volatility: {portfolio.portfolio_volatility:.4%}")
print(f"Portfolio Return: {portfolio.portfolio_return:.4%}")
```

### Risk Contributions

```python
# Get individual asset contributions
contributions = portfolio.get_asset_contributions()
print(contributions)
```

### Diversification Analysis

```python
# Diversification ratio
div_ratio = portfolio.calculate_diversification_ratio()
print(f"Diversification Ratio: {div_ratio:.2f}")

# Correlation impact
corr_impact = portfolio.calculate_correlation_impact()
print(f"Correlation Benefit: {corr_impact['correlation_benefit_pct']:.2f}%")
```

### Portfolio VaR and ES

```python
# Portfolio VaR
var = portfolio.calculate_var(
    confidence_level=0.95,
    horizon=1,
    method='parametric'
)

# Portfolio ES
es = portfolio.calculate_expected_shortfall(
    confidence_level=0.95,
    horizon=1,
    method='historical'
)
```

### Regulatory Capital

```python
# Basel-style capital calculation
capital = portfolio.calculate_regulatory_capital(
    confidence_level=0.99,
    horizon=10,
    capital_multiplier=3.0
)

print(f"10-Day VaR (99%): {capital['var_10day_99']:.4%}")
print(f"Capital Requirement: {capital['capital_requirement']:.4%}")
```

### Risk Parity Optimization

```python
# Calculate risk parity weights
rp_weights = portfolio.optimize_risk_parity()
print("Risk Parity Weights:", rp_weights)
```

### Comprehensive Risk Summary

```python
# Get everything at once
risk_summary = portfolio.get_risk_summary(
    confidence_level=0.95,
    portfolio_value=1000000
)

print(f"Annual Return: {risk_summary['portfolio_return_annual']:.2%}")
print(f"Annual Volatility: {risk_summary['portfolio_volatility_annual']:.2%}")
print(f"Sharpe Ratio: {risk_summary['sharpe_ratio']:.2f}")
print(f"1-Day VaR: ${risk_summary['var_1day_absolute']:,.0f}")
print(f"10-Day VaR: ${risk_summary['var_10day_absolute']:,.0f}")
```

## Streamlit Dashboard

### Launch Dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`

### Dashboard Features

1. **Market Data**: View and analyze stock, FX, and interest rate data
2. **Yield Curve**: Interactive yield curve construction and visualization
3. **Pricing & Greeks**: Bond and option pricing with sensitivity analysis
4. **VaR & ES**: Risk metrics calculation and backtesting
5. **Stress Testing**: Apply scenarios and analyze results
6. **Portfolio Risk**: Aggregate portfolio risk with detailed analytics

### Dashboard Navigation

Use the sidebar to switch between different modules. Each module provides:
- Interactive parameter inputs
- Real-time calculations
- Visual charts and graphs
- Detailed results tables

## Tips and Best Practices

### Performance
- Use parametric VaR for quick estimates
- Use historical VaR for non-normal distributions
- Cache yield curves when pricing multiple bonds

### Accuracy
- Use at least 252 data points for VaR (1 year of daily returns)
- Validate VaR models with backtesting
- Consider multiple stress scenarios

### Risk Management
- Monitor correlation changes over time
- Diversify across asset classes
- Set appropriate confidence levels based on risk appetite
- Regularly update risk metrics

### Regulatory Compliance
- Use 99% confidence for regulatory capital
- Use 10-day horizon for market risk capital
- Document all assumptions and methodologies

## Advanced Topics

### Custom Data Integration

```python
# Use your own data
import pandas as pd

# Load your data
custom_data = pd.read_csv('your_data.csv')

# Use with any module
from src.market_risk.risk import VaRCalculator
var_calc = VaRCalculator(custom_data['returns'])
```

### Batch Processing

```python
# Calculate VaR for multiple portfolios
portfolios = [...]
results = []

for portfolio in portfolios:
    var = portfolio.calculate_var(0.95, 1)
    results.append(var)
```

### Custom Pricing Models

Extend the base classes to implement your own models:

```python
from src.market_risk.pricing.bond_pricer import BondPricer

class CustomBondPricer(BondPricer):
    def price_callable_bond(self, ...):
        # Your custom implementation
        pass
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the correct directory and have installed all dependencies
2. **Data Issues**: Check data format (should be pandas Series or DataFrame)
3. **Convergence Issues**: Adjust parameters or use different methods

### Getting Help

- Check the examples in the `examples/` directory
- Review the test cases in `tests/`
- Consult the README for installation instructions

## References

- Black-Scholes Option Pricing Model
- Basel III Market Risk Framework
- VaR and Expected Shortfall Methodologies
- Yield Curve Bootstrapping Techniques
