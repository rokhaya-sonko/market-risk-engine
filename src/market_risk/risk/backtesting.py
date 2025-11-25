"""
VaR backtesting module
"""

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2
from typing import Tuple, Dict


class Backtester:
    """Backtest VaR models"""
    
    def __init__(self, returns: pd.Series, var_forecasts: pd.Series):
        """
        Initialize backtester
        
        Args:
            returns: Actual returns
            var_forecasts: VaR forecasts (as positive numbers)
        """
        self.returns = returns
        self.var_forecasts = var_forecasts
        
        # Align data
        common_index = returns.index.intersection(var_forecasts.index)
        self.returns = returns.loc[common_index]
        self.var_forecasts = var_forecasts.loc[common_index]
        
        # Calculate violations (losses exceeding VaR)
        self.violations = (self.returns < -self.var_forecasts).astype(int)
        self.n_violations = self.violations.sum()
        self.n_observations = len(self.violations)
    
    def kupiec_test(self, confidence_level: float = 0.95) -> Tuple[float, float, bool]:
        """
        Kupiec's POF (Proportion of Failures) test
        
        Args:
            confidence_level: Confidence level used for VaR
            
        Returns:
            Tuple of (test statistic, p-value, reject null hypothesis)
        """
        expected_violations = (1 - confidence_level) * self.n_observations
        
        if self.n_violations == 0:
            lr_stat = 0
        else:
            # Likelihood ratio statistic
            p = self.n_violations / self.n_observations
            lr_stat = -2 * (
                self.n_observations * ((1 - confidence_level) * np.log(1 - confidence_level) +
                                      confidence_level * np.log(confidence_level)) -
                (self.n_violations * np.log(p) + 
                 (self.n_observations - self.n_violations) * np.log(1 - p))
            )
        
        # Chi-square test with 1 degree of freedom
        p_value = 1 - chi2.cdf(lr_stat, df=1)
        reject = p_value < 0.05
        
        return lr_stat, p_value, reject
    
    def christoffersen_test(self) -> Tuple[float, float, bool]:
        """
        Christoffersen's independence test
        
        Returns:
            Tuple of (test statistic, p-value, reject null hypothesis)
        """
        violations = self.violations.values
        
        # Count transitions
        n00 = np.sum((violations[:-1] == 0) & (violations[1:] == 0))
        n01 = np.sum((violations[:-1] == 0) & (violations[1:] == 1))
        n10 = np.sum((violations[:-1] == 1) & (violations[1:] == 0))
        n11 = np.sum((violations[:-1] == 1) & (violations[1:] == 1))
        
        # Transition probabilities
        if n00 + n01 > 0:
            p01 = n01 / (n00 + n01)
        else:
            p01 = 0
            
        if n10 + n11 > 0:
            p11 = n11 / (n10 + n11)
        else:
            p11 = 0
        
        p = (n01 + n11) / (n00 + n01 + n10 + n11)
        
        # Likelihood ratio test
        if p01 > 0 and p11 > 0 and p > 0 and p < 1:
            lr_ind = -2 * (
                (n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p) -
                n00 * np.log(1 - p01) - n01 * np.log(p01) -
                n10 * np.log(1 - p11) - n11 * np.log(p11)
            )
        else:
            lr_ind = 0
        
        # Chi-square test with 1 degree of freedom
        p_value = 1 - chi2.cdf(lr_ind, df=1)
        reject = p_value < 0.05
        
        return lr_ind, p_value, reject
    
    def calculate_violation_ratio(self, confidence_level: float = 0.95) -> float:
        """
        Calculate violation ratio
        
        Args:
            confidence_level: Confidence level used for VaR
            
        Returns:
            Violation ratio (observed / expected)
        """
        expected_rate = 1 - confidence_level
        observed_rate = self.n_violations / self.n_observations
        return observed_rate / expected_rate
    
    def get_backtest_summary(self, confidence_level: float = 0.95) -> Dict:
        """
        Get comprehensive backtest summary
        
        Args:
            confidence_level: Confidence level used for VaR
            
        Returns:
            Dictionary with backtest results
        """
        kupiec_stat, kupiec_pval, kupiec_reject = self.kupiec_test(confidence_level)
        christ_stat, christ_pval, christ_reject = self.christoffersen_test()
        violation_ratio = self.calculate_violation_ratio(confidence_level)
        
        return {
            'n_observations': self.n_observations,
            'n_violations': self.n_violations,
            'violation_rate': self.n_violations / self.n_observations,
            'expected_violations': (1 - confidence_level) * self.n_observations,
            'violation_ratio': violation_ratio,
            'kupiec_stat': kupiec_stat,
            'kupiec_pvalue': kupiec_pval,
            'kupiec_reject': kupiec_reject,
            'christoffersen_stat': christ_stat,
            'christoffersen_pvalue': christ_pval,
            'christoffersen_reject': christ_reject
        }
