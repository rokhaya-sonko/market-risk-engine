"""
Basic usage examples for Market Risk Analytics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market_risk.data import StockData, RatesData
from src.market_risk.curves import YieldCurve
from src.market_risk.pricing import BondPricer, OptionPricer
from src.market_risk.risk import VaRCalculator
from src.market_risk.stress_tests import ScenarioAnalyzer
import pandas as pd
import numpy as np

print("=" * 60)
print("Market Risk Analytics - Basic Usage Examples")
print("=" * 60)

# 1. Market Data
print("\n1. MARKET DATA")
print("-" * 60)
stock = StockData("AAPL")
print(f"Stock volatility: {stock.get_volatility():.2%}")
print(f"Last price: {stock.data['Close'].iloc[-1]:.2f}")

# 2. Yield Curve
print("\n2. YIELD CURVE")
print("-" * 60)
rates_data = RatesData()
curve = YieldCurve(
    rates_data.data.index.tolist(),
    rates_data.data['Rate'].tolist()
)
print(f"5-year zero rate: {curve.get_zero_rate(5.0):.4%}")
print(f"5-year discount factor: {curve.get_discount_factor(5.0):.6f}")
print(f"Forward rate (1Y-2Y): {curve.get_forward_rate(1.0, 2.0):.4%}")

# 3. Bond Pricing
print("\n3. BOND PRICING")
print("-" * 60)
pricer = BondPricer(curve)
sensitivities = pricer.get_all_sensitivities(
    maturity=5.0,
    coupon_rate=0.04,
    face_value=100
)
print(f"Bond Price: {sensitivities['price']:.2f}")
print(f"DV01: {sensitivities['dv01']:.4f}")
print(f"Duration: {sensitivities['duration']:.2f}")
print(f"Convexity: {sensitivities['convexity']:.4f}")

# 4. Option Pricing
print("\n4. OPTION PRICING (BLACK-SCHOLES)")
print("-" * 60)
option = OptionPricer(
    spot=100,
    strike=100,
    maturity=1.0,
    volatility=0.2,
    risk_free_rate=0.03
)
call_greeks = option.get_all_greeks('call')
print(f"Call Price: {call_greeks['price']:.4f}")
print(f"Delta: {call_greeks['delta']:.4f}")
print(f"Gamma: {call_greeks['gamma']:.6f}")
print(f"Vega: {call_greeks['vega']:.4f}")

put_greeks = option.get_all_greeks('put')
print(f"Put Price: {put_greeks['price']:.4f}")
print(f"Delta: {put_greeks['delta']:.4f}")

# 5. VaR Calculation
print("\n5. VALUE AT RISK")
print("-" * 60)
returns = stock.get_returns()
var_calc = VaRCalculator(returns)
metrics = var_calc.calculate_all_metrics(confidence_level=0.95, horizon=1)
print(f"Parametric VaR (95%, 1-day): {metrics['parametric_var']:.4%}")
print(f"Historical VaR (95%, 1-day): {metrics['historical_var']:.4%}")
print(f"Parametric ES (95%, 1-day): {metrics['parametric_es']:.4%}")
print(f"Historical ES (95%, 1-day): {metrics['historical_es']:.4%}")

# 6. Stress Testing
print("\n6. STRESS TESTING")
print("-" * 60)
analyzer = ScenarioAnalyzer()
result = analyzer.apply_scenario(
    portfolio_value=1000000,
    equity_exposure=500000,
    fx_exposure=100000,
    rates_exposure=-1000,
    vol_exposure=500,
    scenario_name='market_crash'
)
print(f"Scenario: {result['scenario']}")
print(f"Description: {result['description']}")
print(f"Total P&L: ${result['total_pnl']:,.0f}")
print(f"P&L %: {result['pnl_percent']:.2f}%")
print(f"Stressed Value: ${result['stressed_value']:,.0f}")

print("\n" + "=" * 60)
print("Examples completed successfully!")
print("=" * 60)
