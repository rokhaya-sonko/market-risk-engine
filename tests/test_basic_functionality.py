"""
Basic functionality tests for Market Risk Analytics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.market_risk.data import StockData, FXData, RatesData
from src.market_risk.curves import YieldCurve
from src.market_risk.pricing import BondPricer, OptionPricer
from src.market_risk.risk import VaRCalculator, Backtester
from src.market_risk.stress_tests import ScenarioAnalyzer
from src.market_risk.portfolio import PortfolioRisk


def test_market_data():
    """Test market data functionality"""
    stock = StockData("TEST")
    assert len(stock.data) > 0
    assert stock.get_volatility() > 0
    
    fx = FXData("EUR/USD")
    assert len(fx.data) > 0
    
    rates = RatesData()
    assert len(rates.data) > 0
    print("✓ Market data tests passed")


def test_yield_curve():
    """Test yield curve construction"""
    maturities = [1, 2, 5, 10]
    rates = [0.02, 0.025, 0.03, 0.035]
    curve = YieldCurve(maturities, rates)
    
    assert curve.get_discount_factor(1) > 0
    assert curve.get_zero_rate(5) > 0
    assert curve.get_forward_rate(1, 2) > 0
    print("✓ Yield curve tests passed")


def test_bond_pricing():
    """Test bond pricing"""
    maturities = [1, 2, 5, 10]
    rates = [0.02, 0.025, 0.03, 0.035]
    curve = YieldCurve(maturities, rates)
    pricer = BondPricer(curve)
    
    sensitivities = pricer.get_all_sensitivities(5.0, 0.04, 100)
    assert sensitivities['price'] > 0
    assert sensitivities['dv01'] > 0
    assert sensitivities['duration'] > 0
    print("✓ Bond pricing tests passed")


def test_option_pricing():
    """Test option pricing"""
    option = OptionPricer(100, 100, 1.0, 0.2, 0.03)
    
    call_greeks = option.get_all_greeks('call')
    assert call_greeks['price'] > 0
    assert 0 <= call_greeks['delta'] <= 1
    
    put_greeks = option.get_all_greeks('put')
    assert put_greeks['price'] > 0
    assert -1 <= put_greeks['delta'] <= 0
    print("✓ Option pricing tests passed")


def test_var_calculation():
    """Test VaR calculation"""
    returns = pd.Series(np.random.normal(0, 0.01, 252))
    var_calc = VaRCalculator(returns)
    
    metrics = var_calc.calculate_all_metrics(0.95, 1)
    assert metrics['parametric_var'] > 0
    assert metrics['historical_var'] > 0
    assert metrics['parametric_es'] >= metrics['parametric_var']
    print("✓ VaR calculation tests passed")


def test_backtesting():
    """Test VaR backtesting"""
    returns = pd.Series(np.random.normal(0, 0.01, 252))
    var_forecasts = pd.Series([0.02] * 252)
    
    backtester = Backtester(returns, var_forecasts)
    results = backtester.get_backtest_summary(0.95)
    
    assert results['n_observations'] > 0
    assert results['violation_ratio'] >= 0
    print("✓ Backtesting tests passed")


def test_stress_testing():
    """Test stress testing"""
    analyzer = ScenarioAnalyzer()
    
    result = analyzer.apply_scenario(
        1000000, 500000, 100000, -1000, 500, 'market_crash'
    )
    assert 'total_pnl' in result
    assert 'stressed_value' in result
    print("✓ Stress testing tests passed")


def test_portfolio_risk():
    """Test portfolio risk aggregation"""
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns_df = pd.DataFrame({
        'Asset1': np.random.normal(0.0005, 0.01, 252),
        'Asset2': np.random.normal(0.0004, 0.012, 252)
    }, index=dates)
    
    weights = np.array([0.6, 0.4])
    portfolio = PortfolioRisk(returns_df, weights)
    
    assert portfolio.portfolio_volatility > 0
    assert portfolio.calculate_diversification_ratio() > 0
    
    risk_summary = portfolio.get_risk_summary(0.95, 1000000)
    assert risk_summary['var_1day_absolute'] > 0
    print("✓ Portfolio risk tests passed")


if __name__ == '__main__':
    print("Running Market Risk Analytics Tests...")
    print("=" * 60)
    
    test_market_data()
    test_yield_curve()
    test_bond_pricing()
    test_option_pricing()
    test_var_calculation()
    test_backtesting()
    test_stress_testing()
    test_portfolio_risk()
    
    print("=" * 60)
    print("All tests passed successfully! ✓")
