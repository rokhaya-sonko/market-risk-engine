"""
Streamlit Dashboard for Market Risk Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Import our modules
from src.market_risk.data.market_data import StockData, FXData, RatesData
from src.market_risk.curves.yield_curve import YieldCurve
from src.market_risk.pricing.bond_pricer import BondPricer
from src.market_risk.pricing.option_pricer import OptionPricer
from src.market_risk.risk.var_calculator import VaRCalculator
from src.market_risk.risk.backtesting import Backtester
from src.market_risk.stress_tests.scenario_analysis import ScenarioAnalyzer
from src.market_risk.portfolio.portfolio_risk import PortfolioRisk

# Page configuration
st.set_page_config(
    page_title="Market Risk Analytics",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Market Risk Analytics Dashboard")
st.markdown("---")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["Market Data", "Yield Curve", "Pricing & Greeks", "VaR & ES", 
     "Stress Testing", "Portfolio Risk"]
)

# Market Data Page
if page == "Market Data":
    st.header("📈 Market Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Stock Data")
        ticker = st.text_input("Stock Ticker", "AAPL")
        stock = StockData(ticker)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        stock.data['Close'].plot(ax=ax)
        ax.set_title(f"{ticker} Price History")
        ax.set_ylabel("Price")
        st.pyplot(fig)
        
        returns = stock.get_returns()
        st.metric("Annual Volatility", f"{stock.get_volatility():.2%}")
        st.metric("Mean Daily Return", f"{returns.mean():.4%}")
    
    with col2:
        st.subheader("FX Data")
        fx_pair = st.text_input("FX Pair", "EUR/USD")
        fx = FXData(fx_pair)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        fx.data['Rate'].plot(ax=ax)
        ax.set_title(f"{fx_pair} Rate History")
        ax.set_ylabel("Rate")
        st.pyplot(fig)
        
        fx_returns = fx.get_returns()
        st.metric("Annual Volatility", f"{fx.get_volatility():.2%}")
        st.metric("Mean Daily Return", f"{fx_returns.mean():.4%}")
    
    st.subheader("Interest Rates Term Structure")
    rates = RatesData()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rates.data.index, rates.data['Rate'] * 100, marker='o')
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Zero Rate Curve")
    ax.grid(True)
    st.pyplot(fig)
    
    st.dataframe(rates.data.style.format({'Rate': '{:.4%}'}))

# Yield Curve Page
elif page == "Yield Curve":
    st.header("📉 Yield Curve Construction")
    
    st.subheader("Bootstrap Yield Curve from Market Rates")
    
    # Generate sample curve
    rates_data = RatesData()
    maturities = rates_data.data.index.tolist()
    rates = rates_data.data['Rate'].tolist()
    
    curve = YieldCurve(maturities, rates)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Zero Rates")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(curve.maturities, curve.rates * 100, marker='o', label='Zero Rates')
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Rate (%)")
        ax.set_title("Zero Rate Curve")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Discount Factors")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(curve.maturities, curve.discount_factors, marker='o', 
               label='Discount Factors', color='green')
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Discount Factor")
        ax.set_title("Discount Factor Curve")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    
    st.subheader("Forward Rates")
    t1 = st.slider("Start Maturity (years)", 0.25, 20.0, 1.0, 0.25)
    t2 = st.slider("End Maturity (years)", 0.25, 20.0, 2.0, 0.25)
    
    if t2 > t1:
        forward_rate = curve.get_forward_rate(t1, t2)
        st.metric(f"Forward Rate ({t1}Y - {t2}Y)", f"{forward_rate:.4%}")
    
    st.dataframe(curve.to_dataframe().style.format({
        'Zero_Rate': '{:.4%}',
        'Discount_Factor': '{:.6f}'
    }))

# Pricing & Greeks Page
elif page == "Pricing & Greeks":
    st.header("💰 Pricing & Sensitivities")
    
    tab1, tab2 = st.tabs(["Bond Pricing", "Option Pricing"])
    
    with tab1:
        st.subheader("Fixed-Rate Bond Pricing")
        
        # Create yield curve
        rates_data = RatesData()
        curve = YieldCurve(
            rates_data.data.index.tolist(),
            rates_data.data['Rate'].tolist()
        )
        pricer = BondPricer(curve)
        
        col1, col2 = st.columns(2)
        
        with col1:
            maturity = st.number_input("Maturity (years)", 1.0, 30.0, 5.0, 0.5)
            coupon_rate = st.number_input("Coupon Rate (%)", 0.0, 10.0, 3.0, 0.1) / 100
            face_value = st.number_input("Face Value", 100, 10000, 100, 100)
        
        sensitivities = pricer.get_all_sensitivities(maturity, coupon_rate, face_value)
        
        with col2:
            st.metric("Bond Price", f"{sensitivities['price']:.2f}")
            st.metric("DV01", f"{sensitivities['dv01']:.4f}")
            st.metric("Duration", f"{sensitivities['duration']:.2f}")
            st.metric("Convexity", f"{sensitivities['convexity']:.4f}")
    
    with tab2:
        st.subheader("Black-Scholes Option Pricing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            spot = st.number_input("Spot Price", 50.0, 200.0, 100.0, 1.0)
            strike = st.number_input("Strike Price", 50.0, 200.0, 100.0, 1.0)
            maturity = st.number_input("Maturity (years)", 0.1, 5.0, 1.0, 0.1)
            volatility = st.number_input("Volatility (%)", 5.0, 100.0, 20.0, 1.0) / 100
            risk_free = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 3.0, 0.1) / 100
            option_type = st.radio("Option Type", ["call", "put"])
        
        pricer = OptionPricer(spot, strike, maturity, volatility, risk_free)
        greeks = pricer.get_all_greeks(option_type)
        
        with col2:
            st.metric(f"{option_type.title()} Price", f"{greeks['price']:.4f}")
            st.metric("Delta", f"{greeks['delta']:.4f}")
            st.metric("Gamma", f"{greeks['gamma']:.6f}")
            st.metric("Vega", f"{greeks['vega']:.4f}")
            st.metric("Theta", f"{greeks['theta']:.4f}")
            st.metric("Rho", f"{greeks['rho']:.4f}")
        
        # Greeks visualization
        st.subheader("Greeks Sensitivity")
        spot_range = np.linspace(spot * 0.7, spot * 1.3, 50)
        deltas = []
        gammas = []
        
        for s in spot_range:
            p = OptionPricer(s, strike, maturity, volatility, risk_free)
            deltas.append(p.delta(option_type))
            gammas.append(p.gamma())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(spot_range, deltas)
        ax1.axvline(spot, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel("Spot Price")
        ax1.set_ylabel("Delta")
        ax1.set_title("Delta vs Spot Price")
        ax1.grid(True)
        
        ax2.plot(spot_range, gammas)
        ax2.axvline(spot, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel("Spot Price")
        ax2.set_ylabel("Gamma")
        ax2.set_title("Gamma vs Spot Price")
        ax2.grid(True)
        
        st.pyplot(fig)

# VaR & ES Page
elif page == "VaR & ES":
    st.header("⚠️ Value at Risk & Expected Shortfall")
    
    # Generate sample returns
    stock = StockData("Portfolio")
    returns = stock.get_returns()
    
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_level = st.slider("Confidence Level (%)", 90, 99, 95) / 100
        horizon = st.number_input("Time Horizon (days)", 1, 30, 1, 1)
    
    var_calc = VaRCalculator(returns)
    metrics = var_calc.calculate_all_metrics(confidence_level, horizon)
    
    with col2:
        st.metric("Parametric VaR", f"{metrics['parametric_var']:.4%}")
        st.metric("Historical VaR", f"{metrics['historical_var']:.4%}")
        st.metric("Parametric ES", f"{metrics['parametric_es']:.4%}")
        st.metric("Historical ES", f"{metrics['historical_es']:.4%}")
    
    # Returns distribution
    st.subheader("Returns Distribution")
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.hist(returns, bins=50, alpha=0.7, edgecolor='black', density=True)
    
    # Add VaR lines
    ax.axvline(-metrics['parametric_var'], color='r', linestyle='--', 
              label=f"Parametric VaR ({confidence_level:.0%})")
    ax.axvline(-metrics['historical_var'], color='orange', linestyle='--',
              label=f"Historical VaR ({confidence_level:.0%})")
    
    # Fit normal distribution
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    from scipy.stats import norm
    ax.plot(x, norm.pdf(x, mu, sigma), 'k-', linewidth=2, label='Normal Fit')
    
    ax.set_xlabel("Returns")
    ax.set_ylabel("Density")
    ax.set_title("Returns Distribution with VaR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Backtesting
    st.subheader("VaR Backtesting")
    
    # Generate VaR forecasts
    var_forecasts = pd.Series(
        [metrics['parametric_var']] * len(returns),
        index=returns.index
    )
    
    backtester = Backtester(returns, var_forecasts)
    backtest_results = backtester.get_backtest_summary(confidence_level)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", backtest_results['n_observations'])
    col2.metric("Violations", backtest_results['n_violations'])
    col3.metric("Violation Rate", f"{backtest_results['violation_rate']:.2%}")
    col4.metric("Violation Ratio", f"{backtest_results['violation_ratio']:.2f}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Kupiec Test p-value", f"{backtest_results['kupiec_pvalue']:.4f}")
        if backtest_results['kupiec_reject']:
            st.error("❌ Kupiec test: Reject null hypothesis")
        else:
            st.success("✅ Kupiec test: Do not reject null hypothesis")
    
    with col2:
        st.metric("Christoffersen Test p-value", 
                 f"{backtest_results['christoffersen_pvalue']:.4f}")
        if backtest_results['christoffersen_reject']:
            st.error("❌ Christoffersen test: Reject null hypothesis")
        else:
            st.success("✅ Christoffersen test: Do not reject null hypothesis")

# Stress Testing Page
elif page == "Stress Testing":
    st.header("🔥 Stress Testing & Scenario Analysis")
    
    analyzer = ScenarioAnalyzer()
    
    st.subheader("Portfolio Inputs")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        portfolio_value = st.number_input("Portfolio Value", 100000, 10000000, 
                                         1000000, 100000)
    with col2:
        equity_exposure = st.number_input("Equity Exposure", -1000000, 1000000, 
                                         500000, 10000)
    with col3:
        fx_exposure = st.number_input("FX Exposure", -1000000, 1000000, 
                                     100000, 10000)
    with col4:
        rates_exposure = st.number_input("Rates DV01", -10000, 10000, 
                                        -1000, 100)
    
    vol_exposure = st.number_input("Vega (Vol Exposure)", -10000, 10000, 
                                   500, 100)
    
    # Run all scenarios
    st.subheader("Stress Test Results")
    
    results = analyzer.run_all_scenarios(
        portfolio_value, equity_exposure, fx_exposure,
        rates_exposure, vol_exposure
    )
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # P&L by scenario
    scenarios = results['scenario'].values
    pnl = results['total_pnl'].values
    colors = ['red' if x < 0 else 'green' for x in pnl]
    
    ax1.barh(scenarios, pnl, color=colors, alpha=0.7)
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel("Total P&L")
    ax1.set_title("Stress Test P&L by Scenario")
    ax1.grid(True, alpha=0.3)
    
    # P&L breakdown
    scenarios_short = [s[:20] for s in scenarios]
    width = 0.2
    x = np.arange(len(scenarios))
    
    ax2.bar(x - 1.5*width, results['equity_pnl'], width, label='Equity', alpha=0.8)
    ax2.bar(x - 0.5*width, results['fx_pnl'], width, label='FX', alpha=0.8)
    ax2.bar(x + 0.5*width, results['rates_pnl'], width, label='Rates', alpha=0.8)
    ax2.bar(x + 1.5*width, results['vol_pnl'], width, label='Vol', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios_short, rotation=45, ha='right')
    ax2.set_ylabel("P&L")
    ax2.set_title("P&L Breakdown by Risk Factor")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Table of results
    st.subheader("Detailed Results")
    display_results = results[['scenario', 'description', 'total_pnl', 
                               'pnl_percent', 'stressed_value']].copy()
    st.dataframe(display_results.style.format({
        'total_pnl': '{:,.0f}',
        'pnl_percent': '{:.2f}%',
        'stressed_value': '{:,.0f}'
    }))

# Portfolio Risk Page
elif page == "Portfolio Risk":
    st.header("📊 Portfolio Risk Aggregation")
    
    # Generate sample portfolio data
    st.subheader("Portfolio Configuration")
    
    n_assets = st.slider("Number of Assets", 2, 10, 4)
    
    # Generate synthetic returns for multiple assets
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
    
    returns_dict = {}
    weights = []
    
    for i in range(n_assets):
        asset_name = f"Asset_{i+1}"
        returns_dict[asset_name] = np.random.normal(0.0005, 0.015, 252)
        weights.append(1.0 / n_assets)
    
    returns_df = pd.DataFrame(returns_dict, index=dates)
    weights = np.array(weights)
    
    # Allow user to adjust weights
    st.subheader("Portfolio Weights")
    cols = st.columns(n_assets)
    for i, col in enumerate(cols):
        with col:
            weights[i] = st.number_input(f"Asset {i+1}", 0.0, 1.0, 
                                        weights[i], 0.05, key=f"weight_{i}")
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Calculate portfolio risk
    portfolio = PortfolioRisk(returns_df, weights)
    
    # Display key metrics
    st.subheader("Portfolio Risk Metrics")
    
    portfolio_value = st.number_input("Portfolio Value", 100000, 100000000, 
                                     1000000, 100000)
    
    risk_summary = portfolio.get_risk_summary(0.95, portfolio_value)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Annual Return", f"{risk_summary['portfolio_return_annual']:.2%}")
        st.metric("Annual Volatility", f"{risk_summary['portfolio_volatility_annual']:.2%}")
    
    with col2:
        st.metric("Sharpe Ratio", f"{risk_summary['sharpe_ratio']:.2f}")
        st.metric("Diversification Ratio", f"{risk_summary['diversification_ratio']:.2f}")
    
    with col3:
        st.metric("1-Day VaR (95%)", f"${risk_summary['var_1day_absolute']:,.0f}")
        st.metric("10-Day VaR (95%)", f"${risk_summary['var_10day_absolute']:,.0f}")
    
    with col4:
        st.metric("1-Day ES (95%)", f"${risk_summary['es_1day_absolute']:,.0f}")
        st.metric("Correlation Benefit", f"{risk_summary['correlation_benefit_pct']:.1f}%")
    
    # Risk contributions
    st.subheader("Asset Risk Contributions")
    
    contributions = portfolio.get_asset_contributions()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Weights vs contribution
    x = np.arange(len(contributions))
    width = 0.35
    
    ax1.bar(x - width/2, contributions['Weight'] * 100, width, 
           label='Weight', alpha=0.8)
    ax1.bar(x + width/2, contributions['Percent_Contribution'], width,
           label='Risk Contribution', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(contributions['Asset'])
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Portfolio Weight vs Risk Contribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Correlation matrix
    sns.heatmap(portfolio.corr_matrix, annot=True, fmt='.2f', 
               cmap='coolwarm', center=0, ax=ax2, cbar_kws={'label': 'Correlation'})
    ax2.set_title("Asset Correlation Matrix")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Correlation impact
    st.subheader("Correlation Impact Analysis")
    corr_impact = portfolio.calculate_correlation_impact()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Zero Correlation Vol", 
                 f"{corr_impact['zero_correlation_volatility']*np.sqrt(252):.2%}")
    with col2:
        st.metric("Actual Vol", 
                 f"{corr_impact['actual_volatility']*np.sqrt(252):.2%}")
    with col3:
        st.metric("Perfect Correlation Vol", 
                 f"{corr_impact['perfect_correlation_volatility']*np.sqrt(252):.2%}")
    
    # Regulatory capital
    st.subheader("Regulatory Capital (Simplified Basel)")
    capital = portfolio.calculate_regulatory_capital()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("10-Day VaR (99%)", 
                 f"${capital['var_10day_99'] * portfolio_value:,.0f}")
    with col2:
        st.metric("Stressed VaR", 
                 f"${capital['stressed_var'] * portfolio_value:,.0f}")
    with col3:
        st.metric("Capital Requirement", 
                 f"${capital['capital_requirement'] * portfolio_value:,.0f}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Market Risk Analytics Dashboard | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
