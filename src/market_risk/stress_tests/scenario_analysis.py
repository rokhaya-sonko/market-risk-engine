"""
Stress testing and scenario analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional


class ScenarioAnalyzer:
    """Stress testing and scenario analysis"""
    
    def __init__(self):
        self.scenarios = {}
        self._define_standard_scenarios()
    
    def _define_standard_scenarios(self):
        """Define standard market stress scenarios"""
        self.scenarios = {
            'market_crash': {
                'description': 'Severe market crash (-30% equities, +20% vol)',
                'equity_shock': -0.30,
                'fx_shock': 0.10,
                'rates_shock': -0.005,
                'vol_shock': 0.20
            },
            'interest_rate_rise': {
                'description': 'Sharp interest rate increase (+200 bps)',
                'equity_shock': -0.10,
                'fx_shock': 0.05,
                'rates_shock': 0.02,
                'vol_shock': 0.05
            },
            'flight_to_quality': {
                'description': 'Flight to quality (widening spreads)',
                'equity_shock': -0.15,
                'fx_shock': -0.05,
                'rates_shock': -0.01,
                'vol_shock': 0.15
            },
            'inflation_spike': {
                'description': 'Unexpected inflation spike',
                'equity_shock': -0.08,
                'fx_shock': 0.08,
                'rates_shock': 0.015,
                'vol_shock': 0.10
            },
            'credit_crisis': {
                'description': 'Credit market crisis',
                'equity_shock': -0.25,
                'fx_shock': 0.12,
                'rates_shock': -0.003,
                'vol_shock': 0.25
            },
            'emerging_markets_crisis': {
                'description': 'Emerging markets crisis',
                'equity_shock': -0.20,
                'fx_shock': 0.15,
                'rates_shock': 0.001,
                'vol_shock': 0.18
            }
        }
    
    def add_custom_scenario(self, name: str, description: str,
                           equity_shock: float, fx_shock: float,
                           rates_shock: float, vol_shock: float):
        """
        Add a custom stress scenario
        
        Args:
            name: Scenario name
            description: Scenario description
            equity_shock: Equity price shock (e.g., -0.10 for -10%)
            fx_shock: FX rate shock
            rates_shock: Interest rate shock (absolute, e.g., 0.01 for +100 bps)
            vol_shock: Volatility shock (absolute change)
        """
        self.scenarios[name] = {
            'description': description,
            'equity_shock': equity_shock,
            'fx_shock': fx_shock,
            'rates_shock': rates_shock,
            'vol_shock': vol_shock
        }
    
    def apply_scenario(self, portfolio_value: float,
                      equity_exposure: float,
                      fx_exposure: float,
                      rates_exposure: float,
                      vol_exposure: float,
                      scenario_name: str) -> Dict[str, float]:
        """
        Apply a stress scenario to a portfolio
        
        Args:
            portfolio_value: Current portfolio value
            equity_exposure: Equity exposure (delta or DV01)
            fx_exposure: FX exposure
            rates_exposure: Interest rate exposure (DV01)
            vol_exposure: Volatility exposure (vega)
            scenario_name: Name of the scenario to apply
            
        Returns:
            Dictionary with scenario results
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        scenario = self.scenarios[scenario_name]
        
        # Calculate P&L impact
        equity_pnl = equity_exposure * scenario['equity_shock']
        fx_pnl = fx_exposure * scenario['fx_shock']
        rates_pnl = rates_exposure * scenario['rates_shock'] * 10000  # DV01 to full rate change
        vol_pnl = vol_exposure * scenario['vol_shock'] * 100  # Vega per 1% vol
        
        total_pnl = equity_pnl + fx_pnl + rates_pnl + vol_pnl
        stressed_value = portfolio_value + total_pnl
        pnl_percent = (total_pnl / portfolio_value) * 100
        
        return {
            'scenario': scenario_name,
            'description': scenario['description'],
            'initial_value': portfolio_value,
            'equity_pnl': equity_pnl,
            'fx_pnl': fx_pnl,
            'rates_pnl': rates_pnl,
            'vol_pnl': vol_pnl,
            'total_pnl': total_pnl,
            'stressed_value': stressed_value,
            'pnl_percent': pnl_percent
        }
    
    def run_all_scenarios(self, portfolio_value: float,
                         equity_exposure: float,
                         fx_exposure: float,
                         rates_exposure: float,
                         vol_exposure: float) -> pd.DataFrame:
        """
        Run all defined stress scenarios
        
        Args:
            portfolio_value: Current portfolio value
            equity_exposure: Equity exposure
            fx_exposure: FX exposure
            rates_exposure: Interest rate exposure
            vol_exposure: Volatility exposure
            
        Returns:
            DataFrame with all scenario results
        """
        results = []
        for scenario_name in self.scenarios.keys():
            result = self.apply_scenario(
                portfolio_value, equity_exposure, fx_exposure,
                rates_exposure, vol_exposure, scenario_name
            )
            results.append(result)
        
        return pd.DataFrame(results)
    
    def historical_scenario_analysis(self, returns: pd.DataFrame,
                                    weights: np.ndarray,
                                    shock_dates: List[str]) -> pd.DataFrame:
        """
        Apply historical scenarios to a portfolio
        
        Args:
            returns: DataFrame with asset returns
            weights: Portfolio weights
            shock_dates: List of historical shock dates
            
        Returns:
            DataFrame with historical scenario results
        """
        results = []
        
        for date in shock_dates:
            try:
                date_obj = pd.to_datetime(date)
                if date_obj in returns.index:
                    shock_returns = returns.loc[date_obj]
                    portfolio_return = (shock_returns * weights).sum()
                    
                    results.append({
                        'date': date,
                        'portfolio_return': portfolio_return,
                        'portfolio_return_pct': portfolio_return * 100
                    })
            except:
                continue
        
        return pd.DataFrame(results)
    
    def sensitivity_analysis(self, base_value: float,
                           risk_factor: str,
                           shocks: List[float],
                           valuation_func: Callable) -> pd.DataFrame:
        """
        Perform sensitivity analysis for a risk factor
        
        Args:
            base_value: Base portfolio value
            risk_factor: Name of risk factor
            shocks: List of shocks to apply
            valuation_func: Function that takes a shock and returns new value
            
        Returns:
            DataFrame with sensitivity analysis results
        """
        results = []
        
        for shock in shocks:
            new_value = valuation_func(shock)
            pnl = new_value - base_value
            pnl_pct = (pnl / base_value) * 100
            
            results.append({
                'risk_factor': risk_factor,
                'shock': shock,
                'new_value': new_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
        
        return pd.DataFrame(results)
    
    def reverse_stress_test(self, portfolio_value: float,
                           loss_threshold: float,
                           equity_exposure: float,
                           fx_exposure: float = 0,
                           rates_exposure: float = 0) -> Dict[str, float]:
        """
        Find market moves that would cause a specific loss
        
        Args:
            portfolio_value: Current portfolio value
            loss_threshold: Target loss amount (positive number)
            equity_exposure: Equity exposure
            fx_exposure: FX exposure
            rates_exposure: Interest rate exposure
            
        Returns:
            Dictionary with required shocks
        """
        # Simplified: assume losses come primarily from equity
        if equity_exposure != 0:
            required_equity_shock = -loss_threshold / equity_exposure
        else:
            required_equity_shock = 0
        
        if fx_exposure != 0:
            required_fx_shock = -loss_threshold / fx_exposure
        else:
            required_fx_shock = 0
        
        if rates_exposure != 0:
            required_rates_shock = -loss_threshold / (rates_exposure * 10000)
        else:
            required_rates_shock = 0
        
        return {
            'loss_threshold': loss_threshold,
            'loss_threshold_pct': (loss_threshold / portfolio_value) * 100,
            'required_equity_shock': required_equity_shock,
            'required_equity_shock_pct': required_equity_shock * 100,
            'required_fx_shock': required_fx_shock,
            'required_fx_shock_pct': required_fx_shock * 100,
            'required_rates_shock_bps': required_rates_shock * 10000
        }
