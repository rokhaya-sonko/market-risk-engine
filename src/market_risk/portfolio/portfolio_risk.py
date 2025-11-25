"""
Portfolio risk aggregation with volatility, correlation, and capital calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from scipy.stats import norm


class PortfolioRisk:
    """Portfolio risk aggregation and capital calculation"""
    
    def __init__(self, returns: pd.DataFrame, weights: Optional[np.ndarray] = None):
        """
        Initialize portfolio risk calculator
        
        Args:
            returns: DataFrame with returns for each asset (columns are assets)
            weights: Portfolio weights (if None, equal weights)
        """
        self.returns = returns
        
        if weights is None:
            n_assets = len(returns.columns)
            self.weights = np.ones(n_assets) / n_assets
        else:
            self.weights = np.array(weights)
        
        # Calculate statistics
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        self.corr_matrix = self.returns.corr()
        
        # Portfolio metrics
        self.portfolio_return = self._calculate_portfolio_return()
        self.portfolio_volatility = self._calculate_portfolio_volatility()
    
    def _calculate_portfolio_return(self) -> float:
        """Calculate expected portfolio return"""
        return np.dot(self.weights, self.mean_returns)
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility"""
        portfolio_variance = self.weights @ self.cov_matrix @ self.weights
        return np.sqrt(portfolio_variance)
    
    def get_asset_contributions(self) -> pd.DataFrame:
        """
        Calculate individual asset contributions to portfolio risk
        
        Returns:
            DataFrame with risk contributions
        """
        # Marginal contribution to risk
        marginal_contrib = (self.cov_matrix @ self.weights) / self.portfolio_volatility
        
        # Component contribution (weighted marginal)
        component_contrib = self.weights * marginal_contrib
        
        # Percentage contribution
        pct_contrib = (component_contrib / self.portfolio_volatility) * 100
        
        return pd.DataFrame({
            'Asset': self.returns.columns,
            'Weight': self.weights,
            'Volatility': self.returns.std().values,
            'Marginal_Contribution': marginal_contrib,
            'Component_Contribution': component_contrib,
            'Percent_Contribution': pct_contrib
        })
    
    def calculate_diversification_ratio(self) -> float:
        """
        Calculate diversification ratio
        
        Returns:
            Diversification ratio (weighted average vol / portfolio vol)
        """
        weighted_avg_vol = np.dot(self.weights, self.returns.std())
        return weighted_avg_vol / self.portfolio_volatility
    
    def calculate_correlation_impact(self) -> Dict[str, float]:
        """
        Calculate impact of correlations on portfolio risk
        
        Returns:
            Dictionary with correlation metrics
        """
        # Portfolio variance with actual correlations
        actual_variance = self.weights @ self.cov_matrix @ self.weights
        
        # Portfolio variance assuming zero correlations
        individual_variances = np.diag(self.cov_matrix)
        zero_corr_variance = np.sum((self.weights ** 2) * individual_variances)
        
        # Portfolio variance assuming perfect correlations
        stds = np.sqrt(individual_variances)
        perfect_corr_vol = np.dot(self.weights, stds)
        perfect_corr_variance = perfect_corr_vol ** 2
        
        return {
            'actual_volatility': np.sqrt(actual_variance),
            'zero_correlation_volatility': np.sqrt(zero_corr_variance),
            'perfect_correlation_volatility': np.sqrt(perfect_corr_variance),
            'correlation_benefit': np.sqrt(zero_corr_variance) - np.sqrt(actual_variance),
            'correlation_benefit_pct': (1 - np.sqrt(actual_variance) / np.sqrt(zero_corr_variance)) * 100
        }
    
    def calculate_var(self, confidence_level: float = 0.95,
                     horizon: int = 1, method: str = 'parametric') -> float:
        """
        Calculate portfolio VaR
        
        Args:
            confidence_level: Confidence level
            horizon: Time horizon in days
            method: 'parametric' or 'historical'
            
        Returns:
            Portfolio VaR
        """
        portfolio_returns = (self.returns * self.weights).sum(axis=1)
        
        if method == 'parametric':
            z_score = norm.ppf(1 - confidence_level)
            var = -(self.portfolio_return * horizon + 
                   z_score * self.portfolio_volatility * np.sqrt(horizon))
        else:  # historical
            scaled_returns = portfolio_returns * np.sqrt(horizon)
            var = -np.percentile(scaled_returns, (1 - confidence_level) * 100)
        
        return var
    
    def calculate_expected_shortfall(self, confidence_level: float = 0.95,
                                    horizon: int = 1, method: str = 'parametric') -> float:
        """
        Calculate portfolio Expected Shortfall
        
        Args:
            confidence_level: Confidence level
            horizon: Time horizon in days
            method: 'parametric' or 'historical'
            
        Returns:
            Portfolio ES
        """
        portfolio_returns = (self.returns * self.weights).sum(axis=1)
        
        if method == 'parametric':
            z_score = norm.ppf(1 - confidence_level)
            es = -(self.portfolio_return * horizon - 
                  self.portfolio_volatility * np.sqrt(horizon) * 
                  norm.pdf(z_score) / (1 - confidence_level))
        else:  # historical
            scaled_returns = portfolio_returns * np.sqrt(horizon)
            var_threshold = -np.percentile(scaled_returns, (1 - confidence_level) * 100)
            tail_losses = scaled_returns[scaled_returns < -var_threshold]
            es = -tail_losses.mean() if len(tail_losses) > 0 else var_threshold
        
        return es
    
    def calculate_regulatory_capital(self, confidence_level: float = 0.99,
                                    horizon: int = 10,
                                    capital_multiplier: float = 3.0) -> Dict[str, float]:
        """
        Calculate simplified regulatory capital (Basel-style)
        
        Args:
            confidence_level: Confidence level (typically 99%)
            horizon: Time horizon in days (typically 10)
            capital_multiplier: Regulatory multiplier (typically 3)
            
        Returns:
            Dictionary with capital metrics
        """
        # Market risk VaR
        var_10day = self.calculate_var(confidence_level, horizon, 'parametric')
        
        # Stressed VaR (approximation: 1.5x normal VaR)
        stressed_var = var_10day * 1.5
        
        # Capital requirement
        capital_requirement = max(var_10day, stressed_var) * capital_multiplier
        
        return {
            'var_10day_99': var_10day,
            'stressed_var': stressed_var,
            'capital_multiplier': capital_multiplier,
            'capital_requirement': capital_requirement,
            'capital_as_pct_portfolio': (capital_requirement / 
                                        (self.weights.sum() * 100)) * 100
        }
    
    def optimize_risk_parity(self) -> np.ndarray:
        """
        Calculate risk parity weights (equal risk contribution)
        
        Returns:
            Risk parity weights
        """
        n_assets = len(self.weights)
        
        # Start with equal weights
        weights = np.ones(n_assets) / n_assets
        
        # Iterative approach to risk parity
        for _ in range(100):
            portfolio_vol = np.sqrt(weights @ self.cov_matrix @ weights)
            marginal_contrib = (self.cov_matrix @ weights) / portfolio_vol
            
            # Adjust weights inversely to marginal contribution
            new_weights = 1 / marginal_contrib
            new_weights = new_weights / new_weights.sum()
            
            # Check convergence
            if np.allclose(weights, new_weights, rtol=1e-4):
                break
            
            weights = new_weights
        
        return weights
    
    def get_risk_summary(self, confidence_level: float = 0.95,
                        portfolio_value: float = 1000000) -> Dict:
        """
        Get comprehensive portfolio risk summary
        
        Args:
            confidence_level: Confidence level for VaR/ES
            portfolio_value: Portfolio value for absolute risk measures
            
        Returns:
            Dictionary with all risk metrics
        """
        var_1day = self.calculate_var(confidence_level, 1, 'parametric')
        var_10day = self.calculate_var(confidence_level, 10, 'parametric')
        es_1day = self.calculate_expected_shortfall(confidence_level, 1, 'parametric')
        
        corr_impact = self.calculate_correlation_impact()
        div_ratio = self.calculate_diversification_ratio()
        capital = self.calculate_regulatory_capital()
        
        return {
            'portfolio_return_annual': self.portfolio_return * 252,
            'portfolio_volatility_annual': self.portfolio_volatility * np.sqrt(252),
            'sharpe_ratio': (self.portfolio_return * 252) / (self.portfolio_volatility * np.sqrt(252)),
            'diversification_ratio': div_ratio,
            'var_1day_relative': var_1day,
            'var_1day_absolute': var_1day * portfolio_value,
            'var_10day_relative': var_10day,
            'var_10day_absolute': var_10day * portfolio_value,
            'es_1day_relative': es_1day,
            'es_1day_absolute': es_1day * portfolio_value,
            'correlation_benefit_pct': corr_impact['correlation_benefit_pct'],
            'regulatory_capital': capital['capital_requirement'] * portfolio_value
        }
