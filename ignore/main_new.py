import os, datetime, time, itertools
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.config.settings import BASE_CONFIG
from src.config.config_generator import ConfigGenerator
from src.data.data_handler import DataHandler
from src.indicators.technical import Signal  # Factory parent class
from src.simulation.simulator import Simulator
from src.strategies.simple_strategy import SimpleStrategy
from src.utils.daily_results import DailyResults
from src.utils.performance import compute_monthly_rolling_sharpe 
from src.analysis.daily import DailyResultAnalyzer

# helper for rolling zscore
def zscore_single(x):
    return (x.iloc[-1] - x.mean()) / x.std() if x.std() != 0 else 0


cfg_base = {'ticker': 'btc',  'data_folder': 'data/btc_csv',  'plot': False}
data_handler = DataHandler(cfg_base)
full_df = data_handler.load_data(saving_space=False)
# Save as parquet (much better than CSV/pickle)
# full_df.to_parquet('crypto_df.parquet', engine='pyarrow', index=True)
# full_df = pd.read_parquet('crypto_df.parquet', engine='pyarrow')

"""

signal functions - input is full_df, output is signals
make sure the full_df has the following columns: Open, High, Low, Close, Volume

signal has two types:
1. yhat - from any forcast function
2. rule-based - e.g. if MA1 > MA2, then long, hold till some cnodition; if MA1 < MA2, then short, hold till some other condition

"""


class SignalGenerator:
    def __init__(self, factor: pd.Series):
        self.factor_raw = factor
        self.factor_cleaned = None
        self.factor_normalized = None
        self.signal = None

    def clean(self, winsorize_pct=0.01, method='zscore'):
        """Step 1: Clean the factor (winsorize + smoothing)"""
        factor = self.factor_raw.copy()

        # Winsorization
        lower = factor.quantile(winsorize_pct)
        upper = factor.quantile(1 - winsorize_pct)
        factor = factor.clip(lower, upper)

        # (Optional) You can add more smoothing methods here
        self.factor_cleaned = factor
        return self

    def normalize(self, method='zscore', rolling_window=None):
        """Step 2: Normalize the factor"""
        factor = self.factor_cleaned.copy()

        if method == 'zscore':
            if rolling_window:
                self.factor_normalized = factor.rolling(rolling_window).apply(zscore_single, raw=False)
            else:
                self.factor_normalized = zscore(factor.dropna())
                self.factor_normalized = pd.Series(self.factor_normalized, index=factor.dropna().index)
        elif method == 'rank':
            self.factor_normalized = factor.rank(pct=True)
        else:
            raise ValueError("Unsupported normalization method")

        return self

    def to_signal(self, method='crossover', threshold=0, direction='long_only'):
        """Step 3: Convert normalized factor into signal"""
        factor = self.factor_normalized.copy()

        if method == 'crossover':
            if direction == 'long_only':
                # Check if factor crosses above threshold (false at n-1, true at n)
                self.signal = ((factor.shift(1) <= threshold) & (factor > threshold)).astype(int)
            elif direction == 'short_only':
                self.signal = -((factor.shift(1) <= threshold) & (factor < threshold)).astype(int)
            elif direction == 'both':
                self.signal = ((factor.shift(1) <= threshold) & (factor > threshold)).astype(int) - ((factor.shift(1) >= threshold) & (factor < threshold)).astype(int)
        elif method == 'ReLU':
            self.signal = factor.clip(lower=0)
        else:
            raise ValueError("Unsupported signal generation method")

        return self

    def get_signal(self):
        return self.signal






# cfg_signal = {"signal_type":"RSI", "window":14, "threshold":30, "longshort":"long", "hold_period":10, "num_of_share":1, "cost_ratio":0.001, "slippage_ratio":0.0}
# signal_obj = Signal.create(full_df, cfg_signal)
# signals = signal_obj.get_signal()
# signals.head()

# simulator = Simulator(full_df, signals=signals, config=cfg_signal)
# simulator.run()
# simulator.pnl_gross.sum()
# # daily_results = DailyResults.get_daily_results_df(full_df, simulator.pnl_gross, simulator.cost, simulator.slippage, simulator.pos)

