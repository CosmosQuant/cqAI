# src/strategies/simple_strategy.py
import pandas as pd
from src.strategies.base import BaseStrategy

class SimpleStrategy(BaseStrategy):
    def __init__(self, hold_duration):
        self.hold_duration = hold_duration

    def generate_orders(self, df, signals):
        """
        This simple strategy takes the raw signals (assumed to be 1 for buy and 0 otherwise)
        and, when a buy signal occurs, extends that signal for the next hold_duration periods.
        """
        orders = signals.copy()
        for i in range(len(orders)):
            if orders.iloc[i] == 1:
                orders.iloc[i:i + self.hold_duration] = 1
        return orders
