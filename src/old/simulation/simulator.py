# src/simulation/simulator.py
import pandas as pd
## Simulator class that runs trading simulations using computed signal vectors and trade parameters, returning performance metrics.

class Simulator:
    def __init__(self, df, signals=None, config=None, strategy=None):
        self.df = df
        self.signals=signals        
        self.strategy = strategy 
        self.config = config
        
        self.hold_period = self.config.get("hold_period", 15)
        self.num_of_share = self.config.get("num_of_share", 1)
        self.cost_ratio = self.config.get("cost_ratio", 0.0)
        self.slippage_ratio = self.config.get("slippage_ratio", 0.0)
        
        self.pnl_gross = None
        self.pnl_net = None
        self.cost = None
        self.slippage = None
        self.pos = None

    def get_strategy(self, strategy):
        self.strategy = strategy

    def run(self):
        price = self.df["Close"]
        self.pos = self.signals.rolling(self.hold_period).sum().shift(-1) * self.num_of_share
        self.pnl_gross = price.diff() * self.pos.shift(1)
        self.cost = -abs(self.pos.diff()) * price * self.cost_ratio
        self.slippage = -abs(self.pos.diff()) * price * self.slippage_ratio
        self.pnl_net = self.pnl_gross - self.cost - self.slippage
    