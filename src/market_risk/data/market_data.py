"""
Market data handling module for stocks, FX, and rates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Union


class MarketData:
    """Base class for market data"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def get_returns(self, method: str = 'simple') -> pd.Series:
        """Calculate returns"""
        if method == 'simple':
            returns = self.data.pct_change().dropna()
        elif method == 'log':
            returns = np.log(self.data / self.data.shift(1)).dropna()
        else:
            raise ValueError("Method must be 'simple' or 'log'")
        
        # Flatten to Series if DataFrame
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        return returns
    
    def get_volatility(self, window: int = 252) -> float:
        """Calculate annualized volatility"""
        returns = self.get_returns()
        return float(returns.std() * np.sqrt(window))


class StockData(MarketData):
    """Stock market data handler"""
    
    def __init__(self, ticker: str, data: Optional[pd.DataFrame] = None):
        self.ticker = ticker
        if data is None:
            # Generate synthetic data for demonstration
            data = self._generate_synthetic_data()
        super().__init__(data)
    
    def _generate_synthetic_data(self, days: int = 252) -> pd.DataFrame:
        """Generate synthetic stock price data"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Geometric Brownian Motion
        S0 = 100  # Initial price
        mu = 0.1  # Drift
        sigma = 0.2  # Volatility
        dt = 1/252
        
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), days)
        prices = S0 * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'Date': dates,
            'Close': prices
        }).set_index('Date')


class FXData(MarketData):
    """Foreign exchange data handler"""
    
    def __init__(self, pair: str, data: Optional[pd.DataFrame] = None):
        self.pair = pair
        if data is None:
            # Generate synthetic data for demonstration
            data = self._generate_synthetic_data()
        super().__init__(data)
    
    def _generate_synthetic_data(self, days: int = 252) -> pd.DataFrame:
        """Generate synthetic FX rate data"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Mean-reverting process
        S0 = 1.2  # Initial rate
        mu = 1.2  # Long-term mean
        theta = 0.1  # Mean reversion speed
        sigma = 0.1  # Volatility
        dt = 1/252
        
        rates = [S0]
        for _ in range(days - 1):
            dS = theta * (mu - rates[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
            rates.append(rates[-1] + dS)
        
        return pd.DataFrame({
            'Date': dates,
            'Rate': rates
        }).set_index('Date')


class RatesData(MarketData):
    """Interest rates data handler"""
    
    def __init__(self, currency: str = 'USD', data: Optional[pd.DataFrame] = None):
        self.currency = currency
        if data is None:
            # Generate synthetic data for demonstration
            data = self._generate_synthetic_data()
        super().__init__(data)
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic interest rate term structure"""
        maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]  # Years
        
        # Nelson-Siegel-Svensson model parameters
        beta0 = 0.03  # Long-term level
        beta1 = -0.01  # Short-term component
        beta2 = 0.02  # Medium-term component
        tau = 2.0  # Decay factor
        
        rates = []
        for m in maturities:
            # Simplified Nelson-Siegel
            rate = (beta0 + 
                   beta1 * (1 - np.exp(-m/tau)) / (m/tau) +
                   beta2 * ((1 - np.exp(-m/tau)) / (m/tau) - np.exp(-m/tau)))
            rates.append(max(rate, 0.001))  # Floor at 0.1%
        
        return pd.DataFrame({
            'Maturity': maturities,
            'Rate': rates
        }).set_index('Maturity')
    
    def get_zero_rate(self, maturity: float) -> float:
        """Get zero rate for a specific maturity via interpolation"""
        if maturity in self.data.index:
            return self.data.loc[maturity, 'Rate']
        
        # Linear interpolation
        maturities = self.data.index.values
        rates = self.data['Rate'].values
        return np.interp(maturity, maturities, rates)
