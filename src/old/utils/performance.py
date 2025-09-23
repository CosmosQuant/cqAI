# src/utils/performance.py
import numpy as np
import pandas as pd

def compute_sharpe(daily_returns):
    """Compute Sharpe ratio (assuming risk-free rate = 0)."""
    return daily_returns.mean() / daily_returns.std() 

def compute_monthly_rolling_sharpe(daily_series):
    """
    Compute monthly rolling Sharpe ratio based on daily series.
    Returns a series with monthly Sharpe ratio for each day.
    """
    grouped = daily_series.groupby(daily_series.index.to_period("M"))
    sharpe_by_month = grouped.apply(lambda x: (x.mean() / x.std() * np.sqrt(len(x))) if x.std() != 0 else np.nan)
    sharpe_daily = daily_series.index.to_series().apply(lambda d: sharpe_by_month[d.to_period("M")])
    return sharpe_daily
