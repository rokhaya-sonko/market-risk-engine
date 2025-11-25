# Market Risk Analytics

A comprehensive Python library for market risk management and analytics, featuring data manipulation, yield curve construction, pricing with sensitivities, Value at Risk (VaR) calculations, stress testing, and portfolio risk aggregation with an interactive Streamlit dashboard.

## 🚀 Features

### 1. Market Data Manipulation
- **Stocks**: Handle equity price data with return calculations and volatility metrics
- **FX**: Foreign exchange rate data with mean-reversion modeling
- **Interest Rates**: Term structure data with interpolation

### 2. Yield Curve Construction
- **Bootstrapping**: Build zero-rate curves from bond prices or swap rates
- **Discount Factors**: Calculate present values across maturities
- **Forward Rates**: Compute forward interest rates between any two points

### 3. Pricing & Sensitivities
#### Bond Pricing
- Fixed-rate bond pricing using discount curves
- **DV01**: Dollar value of 1 basis point change
- **Duration**: Modified duration calculation
- **Convexity**: Second-order rate sensitivity

#### Option Pricing (Black-Scholes)
- Call and Put option pricing
- **Greeks**: Delta, Gamma, Vega, Theta, Rho
- Sensitivity analysis and visualization

### 4. Risk Metrics
#### Value at Risk (VaR)
- **Parametric VaR**: Normal distribution assumption
- **Historical VaR**: Empirical distribution
- **Expected Shortfall (ES/CVaR)**: Tail risk measurement

#### Backtesting
- **Kupiec Test**: Proportion of failures test
- **Christoffersen Test**: Independence test
- Violation ratio analysis

### 5. Stress Testing
- **Predefined Scenarios**: Market crash, rate rise, credit crisis, etc.
- **Custom Scenarios**: Define your own stress scenarios
- **Historical Scenarios**: Apply historical market shocks
- **Reverse Stress Testing**: Find scenarios causing specific losses
- **Sensitivity Analysis**: Factor-by-factor impact assessment

### 6. Portfolio Risk Aggregation
- **Volatility Aggregation**: Portfolio-level risk calculation
- **Correlation Analysis**: Impact of correlations on risk
- **Risk Contributions**: Individual asset contributions to portfolio risk
- **Diversification Metrics**: Quantify diversification benefits
- **Regulatory Capital**: Simplified Basel-style capital calculation
- **Risk Parity**: Equal risk contribution optimization

### 7. Interactive Dashboard
- **Streamlit-based UI**: User-friendly web interface
- **Real-time Calculations**: Interactive parameter adjustment
- **Visualizations**: Charts and graphs for all metrics
- **Multi-page Navigation**: Organized by functionality

## 📦 Installation

### Clone the repository
```bash
git clone https://github.com/rokhaya-sonko/market-risk-analytics.git
cd market-risk-analytics
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### Run the Streamlit Dashboard
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Using the Library Programmatically

#### Market Data
```python
from src.market_risk.data import StockData, FXData, RatesData

# Stock data
stock = StockData("AAPL")
returns = stock.get_returns()
volatility = stock.get_volatility()

# FX data
fx = FXData("EUR/USD")

# Interest rates
rates = RatesData("USD")
zero_rate = rates.get_zero_rate(5.0)  # 5-year rate
```

#### Yield Curve
```python
from src.market_risk.curves import YieldCurve

# Create yield curve
maturities = [0.5, 1, 2, 5, 10, 30]
rates = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
curve = YieldCurve(maturities, rates)

# Get discount factor
df = curve.get_discount_factor(3.0)

# Get forward rate
forward = curve.get_forward_rate(1.0, 2.0)
```

#### Bond Pricing
```python
from src.market_risk.pricing import BondPricer

pricer = BondPricer(curve)
sensitivities = pricer.get_all_sensitivities(
    maturity=5.0,
    coupon_rate=0.04,
    face_value=100
)

print(f"Price: {sensitivities['price']:.2f}")
print(f"DV01: {sensitivities['dv01']:.4f}")
print(f"Duration: {sensitivities['duration']:.2f}")
```

#### Option Pricing
```python
from src.market_risk.pricing import OptionPricer

option = OptionPricer(
    spot=100,
    strike=100,
    maturity=1.0,
    volatility=0.2,
    risk_free_rate=0.03
)

greeks = option.get_all_greeks('call')
print(f"Call Price: {greeks['price']:.4f}")
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.6f}")
```

#### VaR Calculation
```python
from src.market_risk.risk import VaRCalculator

# Calculate VaR from returns
var_calc = VaRCalculator(returns)

metrics = var_calc.calculate_all_metrics(
    confidence_level=0.95,
    horizon=1
)

print(f"Parametric VaR: {metrics['parametric_var']:.4%}")
print(f"Historical VaR: {metrics['historical_var']:.4%}")
print(f"Expected Shortfall: {metrics['parametric_es']:.4%}")
```

#### Backtesting
```python
from src.market_risk.risk import Backtester

backtester = Backtester(returns, var_forecasts)
results = backtester.get_backtest_summary(confidence_level=0.95)

print(f"Violations: {results['n_violations']}")
print(f"Kupiec p-value: {results['kupiec_pvalue']:.4f}")
```

#### Stress Testing
```python
from src.market_risk.stress_tests import ScenarioAnalyzer

analyzer = ScenarioAnalyzer()

# Run a stress scenario
result = analyzer.apply_scenario(
    portfolio_value=1000000,
    equity_exposure=500000,
    fx_exposure=100000,
    rates_exposure=-1000,
    vol_exposure=500,
    scenario_name='market_crash'
)

print(f"Total P&L: ${result['total_pnl']:,.0f}")
print(f"P&L %: {result['pnl_percent']:.2f}%")
```

#### Portfolio Risk
```python
from src.market_risk.portfolio import PortfolioRisk

# Create portfolio with returns data
portfolio = PortfolioRisk(returns_df, weights)

# Get comprehensive risk summary
risk_summary = portfolio.get_risk_summary(
    confidence_level=0.95,
    portfolio_value=1000000
)

print(f"Portfolio Vol: {risk_summary['portfolio_volatility_annual']:.2%}")
print(f"Sharpe Ratio: {risk_summary['sharpe_ratio']:.2f}")
print(f"1-Day VaR: ${risk_summary['var_1day_absolute']:,.0f}")

# Analyze risk contributions
contributions = portfolio.get_asset_contributions()
print(contributions)
```

## 📊 Dashboard Navigation

The Streamlit dashboard is organized into six main sections:

1. **Market Data**: View and analyze stock, FX, and interest rate data
2. **Yield Curve**: Construct and visualize yield curves with discount factors
3. **Pricing & Greeks**: Price bonds and options with full sensitivity analysis
4. **VaR & ES**: Calculate risk metrics and perform backtesting
5. **Stress Testing**: Run predefined and custom stress scenarios
6. **Portfolio Risk**: Aggregate portfolio risk with correlation and capital metrics

## 🧪 Testing

The library includes comprehensive test coverage. To run tests:

```bash
pytest tests/
```

## 📁 Project Structure

```
market-risk-analytics/
├── src/
│   └── market_risk/
│       ├── data/              # Market data handling
│       ├── curves/            # Yield curve construction
│       ├── pricing/           # Bond and option pricing
│       ├── risk/              # VaR, ES, and backtesting
│       ├── stress_tests/      # Scenario analysis
│       └── portfolio/         # Portfolio risk aggregation
├── tests/                     # Test suite
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Technical Details

### Key Technologies
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **SciPy**: Statistical functions and optimization
- **Matplotlib & Seaborn**: Visualization
- **Streamlit**: Interactive dashboard

### Financial Models
- **Black-Scholes**: Option pricing model
- **Nelson-Siegel**: Yield curve parameterization
- **Geometric Brownian Motion**: Stock price simulation
- **Mean Reversion**: FX rate modeling

### Risk Models
- **Parametric VaR**: Based on normal distribution
- **Historical VaR**: Based on empirical distribution
- **Component VaR**: Risk attribution at asset level
- **Basel-style Capital**: Simplified regulatory capital

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🙏 Acknowledgments

This project implements standard financial risk management practices and methodologies used in the industry.