"""
Yield curve construction with bootstrapping and discount factors
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy.interpolate import interp1d


class YieldCurve:
    """Yield curve with bootstrapping and discount factor calculations"""
    
    def __init__(self, maturities: List[float], rates: List[float]):
        """
        Initialize yield curve
        
        Args:
            maturities: List of maturities in years
            rates: List of zero rates (continuously compounded)
        """
        self.maturities = np.array(maturities)
        self.rates = np.array(rates)
        self.discount_factors = self._calculate_discount_factors()
        
    def _calculate_discount_factors(self) -> np.ndarray:
        """Calculate discount factors from zero rates"""
        return np.exp(-self.rates * self.maturities)
    
    def get_discount_factor(self, maturity: float) -> float:
        """
        Get discount factor for a specific maturity via interpolation
        
        Args:
            maturity: Maturity in years
            
        Returns:
            Discount factor
        """
        if maturity in self.maturities:
            idx = np.where(self.maturities == maturity)[0][0]
            return self.discount_factors[idx]
        
        # Linear interpolation on log discount factors
        log_df = np.log(self.discount_factors)
        interp_func = interp1d(self.maturities, log_df, kind='linear', 
                              fill_value='extrapolate')
        return np.exp(interp_func(maturity))
    
    def get_zero_rate(self, maturity: float) -> float:
        """
        Get zero rate for a specific maturity via interpolation
        
        Args:
            maturity: Maturity in years
            
        Returns:
            Zero rate
        """
        if maturity in self.maturities:
            idx = np.where(self.maturities == maturity)[0][0]
            return self.rates[idx]
        
        # Linear interpolation
        interp_func = interp1d(self.maturities, self.rates, kind='linear',
                              fill_value='extrapolate')
        return float(interp_func(maturity))
    
    def get_forward_rate(self, t1: float, t2: float) -> float:
        """
        Calculate forward rate between two maturities
        
        Args:
            t1: Start time in years
            t2: End time in years
            
        Returns:
            Forward rate
        """
        df1 = self.get_discount_factor(t1)
        df2 = self.get_discount_factor(t2)
        return np.log(df1 / df2) / (t2 - t1)
    
    @classmethod
    def bootstrap_from_bonds(cls, bond_data: List[Tuple[float, float, float]]) -> 'YieldCurve':
        """
        Bootstrap yield curve from bond prices
        
        Args:
            bond_data: List of tuples (maturity, coupon_rate, price)
                      where coupon_rate is annual and price is clean price
        
        Returns:
            YieldCurve object
        """
        bond_data = sorted(bond_data, key=lambda x: x[0])  # Sort by maturity
        
        maturities = []
        zero_rates = []
        discount_factors = []
        
        for maturity, coupon, price in bond_data:
            if maturity <= 1:
                # For short maturities, use simple calculation
                # Assume annual payment for simplicity
                df = price / (100 + coupon)
                zero_rate = -np.log(df) / maturity
            else:
                # Bootstrap using previously calculated discount factors
                coupon_payment = coupon
                remaining_value = price
                
                # Subtract PV of all known coupon payments
                for i, prev_mat in enumerate(maturities):
                    if prev_mat < maturity:
                        # Annual coupons
                        payment_times = np.arange(1, int(maturity) + 1)
                        for t in payment_times[payment_times <= prev_mat]:
                            if abs(t - prev_mat) < 0.01:  # Close to a discount factor we have
                                remaining_value -= coupon_payment * discount_factors[i]
                
                # Calculate remaining discount factor
                # Simplified: assume annual payments
                n_payments = int(maturity)
                pv_coupons = 0
                for i in range(1, n_payments):
                    if i <= len(discount_factors):
                        pv_coupons += coupon_payment * discount_factors[min(i-1, len(discount_factors)-1)]
                
                df = (remaining_value - pv_coupons) / (100 + coupon_payment)
                zero_rate = -np.log(df) / maturity
            
            maturities.append(maturity)
            zero_rates.append(zero_rate)
            discount_factors.append(df)
        
        return cls(maturities, zero_rates)
    
    @classmethod
    def from_swap_rates(cls, swap_maturities: List[float], 
                       swap_rates: List[float]) -> 'YieldCurve':
        """
        Bootstrap yield curve from swap rates (simplified)
        
        Args:
            swap_maturities: List of swap maturities in years
            swap_rates: List of swap rates (annual)
            
        Returns:
            YieldCurve object
        """
        # Simplified bootstrap assuming annual payments
        maturities = []
        zero_rates = []
        
        for i, (mat, swap_rate) in enumerate(zip(swap_maturities, swap_rates)):
            if i == 0:
                # First point: swap rate = zero rate for short maturity
                zero_rate = swap_rate
            else:
                # Bootstrap: solve for zero rate that makes swap value = 0
                # Simplified approach: approximate zero rate
                zero_rate = swap_rate * 1.02  # Simple approximation
            
            maturities.append(mat)
            zero_rates.append(zero_rate)
        
        return cls(maturities, zero_rates)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert yield curve to DataFrame"""
        return pd.DataFrame({
            'Maturity': self.maturities,
            'Zero_Rate': self.rates,
            'Discount_Factor': self.discount_factors
        })
