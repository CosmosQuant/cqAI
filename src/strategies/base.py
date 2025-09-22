# src/strategies/base.py

## Base strategy class that defines a common interface for all trading strategies.

class BaseStrategy:
    def __init__(self, df):
        self.df = df
        self.position = None

    def generate_signals(self):
        raise NotImplementedError("Subclasses must implement generate_signals method.")
