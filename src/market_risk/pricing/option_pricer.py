"""
Option pricing with Greeks calculation (Black-Scholes model)
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Literal


class OptionPricer:
    """Option pricer using Black-Scholes model with Greeks"""
    
    def __init__(self, spot: float, strike: float, maturity: float,
                 volatility: float, risk_free_rate: float, dividend_yield: float = 0):
        """
        Initialize option pricer
        
        Args:
            spot: Current spot price
            strike: Strike price
            maturity: Time to maturity in years
            volatility: Implied volatility (annualized)
            risk_free_rate: Risk-free rate (continuously compounded)
            dividend_yield: Continuous dividend yield
        """
        self.spot = spot
        self.strike = strike
        self.maturity = maturity
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        
        self._calculate_d1_d2()
    
    def _calculate_d1_d2(self):
        """Calculate d1 and d2 for Black-Scholes formula"""
        if self.maturity <= 0:
            self.d1 = 0
            self.d2 = 0
            return
        
        self.d1 = ((np.log(self.spot / self.strike) + 
                   (self.risk_free_rate - self.dividend_yield + 
                    0.5 * self.volatility ** 2) * self.maturity) /
                  (self.volatility * np.sqrt(self.maturity)))
        
        self.d2 = self.d1 - self.volatility * np.sqrt(self.maturity)
    
    def price_call(self) -> float:
        """Calculate call option price"""
        if self.maturity <= 0:
            return max(self.spot - self.strike, 0)
        
        call_price = (self.spot * np.exp(-self.dividend_yield * self.maturity) * 
                     norm.cdf(self.d1) -
                     self.strike * np.exp(-self.risk_free_rate * self.maturity) * 
                     norm.cdf(self.d2))
        
        return call_price
    
    def price_put(self) -> float:
        """Calculate put option price"""
        if self.maturity <= 0:
            return max(self.strike - self.spot, 0)
        
        put_price = (self.strike * np.exp(-self.risk_free_rate * self.maturity) * 
                    norm.cdf(-self.d2) -
                    self.spot * np.exp(-self.dividend_yield * self.maturity) * 
                    norm.cdf(-self.d1))
        
        return put_price
    
    def delta(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate delta (sensitivity to spot price)
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            Delta value
        """
        if self.maturity <= 0:
            if option_type == 'call':
                return 1.0 if self.spot > self.strike else 0.0
            else:
                return -1.0 if self.spot < self.strike else 0.0
        
        if option_type == 'call':
            return np.exp(-self.dividend_yield * self.maturity) * norm.cdf(self.d1)
        else:
            return -np.exp(-self.dividend_yield * self.maturity) * norm.cdf(-self.d1)
    
    def gamma(self) -> float:
        """
        Calculate gamma (second derivative with respect to spot)
        
        Returns:
            Gamma value
        """
        if self.maturity <= 0:
            return 0.0
        
        gamma = (np.exp(-self.dividend_yield * self.maturity) * 
                norm.pdf(self.d1) / 
                (self.spot * self.volatility * np.sqrt(self.maturity)))
        
        return gamma
    
    def vega(self) -> float:
        """
        Calculate vega (sensitivity to volatility)
        
        Returns:
            Vega value (per 1% change in volatility)
        """
        if self.maturity <= 0:
            return 0.0
        
        vega = (self.spot * np.exp(-self.dividend_yield * self.maturity) * 
               norm.pdf(self.d1) * np.sqrt(self.maturity))
        
        return vega / 100  # Per 1% change
    
    def theta(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate theta (time decay)
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            Theta value (per day)
        """
        if self.maturity <= 0:
            return 0.0
        
        term1 = -(self.spot * np.exp(-self.dividend_yield * self.maturity) * 
                 norm.pdf(self.d1) * self.volatility / 
                 (2 * np.sqrt(self.maturity)))
        
        if option_type == 'call':
            term2 = (self.dividend_yield * self.spot * 
                    np.exp(-self.dividend_yield * self.maturity) * norm.cdf(self.d1))
            term3 = -(self.risk_free_rate * self.strike * 
                     np.exp(-self.risk_free_rate * self.maturity) * norm.cdf(self.d2))
        else:
            term2 = -(self.dividend_yield * self.spot * 
                     np.exp(-self.dividend_yield * self.maturity) * norm.cdf(-self.d1))
            term3 = (self.risk_free_rate * self.strike * 
                    np.exp(-self.risk_free_rate * self.maturity) * norm.cdf(-self.d2))
        
        theta = term1 + term2 + term3
        
        return theta / 365  # Per day
    
    def rho(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate rho (sensitivity to interest rate)
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            Rho value (per 1% change in rate)
        """
        if self.maturity <= 0:
            return 0.0
        
        if option_type == 'call':
            rho = (self.strike * self.maturity * 
                  np.exp(-self.risk_free_rate * self.maturity) * 
                  norm.cdf(self.d2))
        else:
            rho = -(self.strike * self.maturity * 
                   np.exp(-self.risk_free_rate * self.maturity) * 
                   norm.cdf(-self.d2))
        
        return rho / 100  # Per 1% change
    
    def get_all_greeks(self, option_type: Literal['call', 'put'] = 'call') -> Dict[str, float]:
        """
        Calculate all Greeks for an option
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with price and all Greeks
        """
        if option_type == 'call':
            price = self.price_call()
        else:
            price = self.price_put()
        
        return {
            'price': price,
            'delta': self.delta(option_type),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(option_type),
            'rho': self.rho(option_type)
        }
