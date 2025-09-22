# src/utils/system_tools.py
import numpy as np

def compare_pnl(pnl1, pnl2):
    """Compare average pnl of two strategies."""
    return pnl1.mean() - pnl2.mean()

def check_correlation(data1, data2):
    """Compute correlation coefficient between two data series."""
    return np.corrcoef(data1, data2)[0, 1]
