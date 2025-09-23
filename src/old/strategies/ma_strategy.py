# src/strategies/ma_strategy.py
import pandas as pd
from src.strategies.base import BaseStrategy

class MAStrategy(BaseStrategy):
    def __init__(self, df, short_window, long_window):
        super().__init__(df)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        df = self.df.copy()
        df["short_ma"] = df["Close"].rolling(window=self.short_window).mean()
        df["long_ma"] = df["Close"].rolling(window=self.long_window).mean()
        df["signal"] = 0
        df.loc[df["short_ma"] > df["long_ma"], "signal"] = 1
        df.loc[df["short_ma"] <= df["long_ma"], "signal"] = -1
        return df["signal"]
