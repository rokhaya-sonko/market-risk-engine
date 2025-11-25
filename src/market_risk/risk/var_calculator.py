"""
VaR and Expected Shortfall calculations (parametric and historical)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Literal, Tuple


class VaRCalculator:
    """Value at Risk and Expected Shortfall calculator"""
    
    def __init__(self, returns: pd.Series):
        """
        Initialize VaR calculator
        
        Args:
            returns: Time series of returns
        """
        self.returns = returns.dropna()
        self.mean = self.returns.mean()
        self.std = self.returns.std()
    
    def parametric_var(self, confidence_level: float = 0.95,
                      horizon: int = 1) -> float:
        """
        Calculate parametric VaR (assumes normal distribution)
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            horizon: Time horizon in days
            
        Returns:
            VaR value (as positive number representing loss)
        """
        z_score = norm.ppf(1 - confidence_level)
        var = -(self.mean * horizon + z_score * self.std * np.sqrt(horizon))
        return var
    
    def historical_var(self, confidence_level: float = 0.95,
                      horizon: int = 1) -> float:
        """
        Calculate historical VaR (empirical distribution)
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            horizon: Time horizon in days
            
        Returns:
            VaR value (as positive number representing loss)
        """
        # Scale returns to horizon
        scaled_returns = self.returns * np.sqrt(horizon)
        
        # Calculate percentile
        var = -np.percentile(scaled_returns, (1 - confidence_level) * 100)
        return var
    
    def parametric_es(self, confidence_level: float = 0.95,
                     horizon: int = 1) -> float:
        """
        Calculate parametric Expected Shortfall (CVaR)
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            horizon: Time horizon in days
            
        Returns:
            ES value (as positive number representing expected loss beyond VaR)
        """
        z_score = norm.ppf(1 - confidence_level)
        # ES for normal distribution
        es = -(self.mean * horizon - self.std * np.sqrt(horizon) * 
               norm.pdf(z_score) / (1 - confidence_level))
        return es
    
    def historical_es(self, confidence_level: float = 0.95,
                     horizon: int = 1) -> float:
        """
        Calculate historical Expected Shortfall (CVaR)
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            horizon: Time horizon in days
            
        Returns:
            ES value (as positive number representing expected loss beyond VaR)
        """
        # Scale returns to horizon
        scaled_returns = self.returns * np.sqrt(horizon)
        
        # Calculate VaR threshold
        var_threshold = -np.percentile(scaled_returns, (1 - confidence_level) * 100)
        
        # ES is the average of losses beyond VaR
        tail_losses = scaled_returns[scaled_returns < -var_threshold]
        if len(tail_losses) > 0:
            es = -tail_losses.mean()
        else:
            es = var_threshold
        
        return es
    
    def calculate_all_metrics(self, confidence_level: float = 0.95,
                            horizon: int = 1) -> dict:
        """
        Calculate all VaR and ES metrics
        
        Args:
            confidence_level: Confidence level
            horizon: Time horizon in days
            
        Returns:
            Dictionary with all metrics
        """
        return {
            'parametric_var': self.parametric_var(confidence_level, horizon),
            'historical_var': self.historical_var(confidence_level, horizon),
            'parametric_es': self.parametric_es(confidence_level, horizon),
            'historical_es': self.historical_es(confidence_level, horizon)
        }
    
    def calculate_component_var(self, portfolio_returns: pd.DataFrame,
                               weights: np.ndarray,
                               confidence_level: float = 0.95) -> pd.Series:
        """
        Calculate component VaR for a portfolio
        
        Args:
            portfolio_returns: DataFrame with returns for each asset
            weights: Portfolio weights
            confidence_level: Confidence level
            
        Returns:
            Series with component VaR for each asset
        """
        # Portfolio return
        portfolio_ret = (portfolio_returns * weights).sum(axis=1)
        
        # Portfolio VaR
        portfolio_var_calc = VaRCalculator(portfolio_ret)
        portfolio_var = portfolio_var_calc.parametric_var(confidence_level)
        
        # Marginal VaR (sensitivity of portfolio VaR to each position)
        cov_matrix = portfolio_returns.cov()
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        marginal_var = (cov_matrix @ weights) / portfolio_vol
        
        # Component VaR
        z_score = norm.ppf(1 - confidence_level)
        component_var = weights * marginal_var * (-z_score)
        
        return pd.Series(component_var, index=portfolio_returns.columns)
