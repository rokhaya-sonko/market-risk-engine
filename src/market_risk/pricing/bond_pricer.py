"""
Bond pricing with DV01 calculation
"""

import numpy as np
from typing import Optional, Dict
from ..curves.yield_curve import YieldCurve


class BondPricer:
    """Bond pricer with sensitivity calculations"""
    
    def __init__(self, yield_curve: YieldCurve):
        self.yield_curve = yield_curve
    
    def price_bond(self, maturity: float, coupon_rate: float, 
                   face_value: float = 100, frequency: int = 1) -> float:
        """
        Price a fixed-rate bond
        
        Args:
            maturity: Maturity in years
            coupon_rate: Annual coupon rate (e.g., 0.05 for 5%)
            face_value: Face value
            frequency: Coupon payment frequency per year
            
        Returns:
            Bond price
        """
        coupon = coupon_rate * face_value / frequency
        n_payments = int(maturity * frequency)
        
        price = 0
        for i in range(1, n_payments + 1):
            t = i / frequency
            df = self.yield_curve.get_discount_factor(t)
            price += coupon * df
        
        # Add face value
        df_maturity = self.yield_curve.get_discount_factor(maturity)
        price += face_value * df_maturity
        
        return price
    
    def calculate_dv01(self, maturity: float, coupon_rate: float,
                      face_value: float = 100, frequency: int = 1,
                      shift: float = 0.0001) -> float:
        """
        Calculate DV01 (Dollar Value of 1 basis point)
        
        Args:
            maturity: Maturity in years
            coupon_rate: Annual coupon rate
            face_value: Face value
            frequency: Coupon payment frequency per year
            shift: Rate shift in decimal (default 1bp = 0.0001)
            
        Returns:
            DV01 value
        """
        # Base price
        price_base = self.price_bond(maturity, coupon_rate, face_value, frequency)
        
        # Shift curve up
        shifted_rates = self.yield_curve.rates + shift
        shifted_curve = YieldCurve(self.yield_curve.maturities.tolist(), 
                                   shifted_rates.tolist())
        pricer_shifted = BondPricer(shifted_curve)
        price_shifted = pricer_shifted.price_bond(maturity, coupon_rate, 
                                                  face_value, frequency)
        
        # DV01 is the negative of price change for 1bp increase
        dv01 = -(price_shifted - price_base)
        
        return dv01
    
    def calculate_duration(self, maturity: float, coupon_rate: float,
                          face_value: float = 100, frequency: int = 1) -> float:
        """
        Calculate modified duration
        
        Args:
            maturity: Maturity in years
            coupon_rate: Annual coupon rate
            face_value: Face value
            frequency: Coupon payment frequency per year
            
        Returns:
            Modified duration
        """
        price = self.price_bond(maturity, coupon_rate, face_value, frequency)
        dv01 = self.calculate_dv01(maturity, coupon_rate, face_value, frequency)
        
        # Modified duration = DV01 * 10000 / Price
        # (multiplying by 10000 to convert from 1bp to 100%)
        duration = dv01 * 10000 / price
        
        return duration
    
    def calculate_convexity(self, maturity: float, coupon_rate: float,
                           face_value: float = 100, frequency: int = 1,
                           shift: float = 0.0001) -> float:
        """
        Calculate convexity
        
        Args:
            maturity: Maturity in years
            coupon_rate: Annual coupon rate
            face_value: Face value
            frequency: Coupon payment frequency per year
            shift: Rate shift in decimal
            
        Returns:
            Convexity
        """
        # Base price
        price_base = self.price_bond(maturity, coupon_rate, face_value, frequency)
        
        # Price with rates shifted up
        shifted_rates_up = self.yield_curve.rates + shift
        shifted_curve_up = YieldCurve(self.yield_curve.maturities.tolist(),
                                      shifted_rates_up.tolist())
        pricer_up = BondPricer(shifted_curve_up)
        price_up = pricer_up.price_bond(maturity, coupon_rate, face_value, frequency)
        
        # Price with rates shifted down
        shifted_rates_down = self.yield_curve.rates - shift
        shifted_curve_down = YieldCurve(self.yield_curve.maturities.tolist(),
                                        shifted_rates_down.tolist())
        pricer_down = BondPricer(shifted_curve_down)
        price_down = pricer_down.price_bond(maturity, coupon_rate, face_value, frequency)
        
        # Convexity formula
        convexity = (price_up + price_down - 2 * price_base) / (price_base * shift ** 2)
        
        return convexity
    
    def get_all_sensitivities(self, maturity: float, coupon_rate: float,
                             face_value: float = 100, frequency: int = 1) -> Dict[str, float]:
        """
        Calculate all sensitivities for a bond
        
        Returns:
            Dictionary with price, DV01, duration, and convexity
        """
        price = self.price_bond(maturity, coupon_rate, face_value, frequency)
        dv01 = self.calculate_dv01(maturity, coupon_rate, face_value, frequency)
        duration = self.calculate_duration(maturity, coupon_rate, face_value, frequency)
        convexity = self.calculate_convexity(maturity, coupon_rate, face_value, frequency)
        
        return {
            'price': price,
            'dv01': dv01,
            'duration': duration,
            'convexity': convexity
        }
