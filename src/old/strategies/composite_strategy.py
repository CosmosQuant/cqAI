# src/strategies/composite_strategy.py
import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy   
# from src.strategies.simple_strategy import SimpleStrategy  # Optional: if you want to inherit common methods
## Composite strategy class that aggregates multiple indicator signals using weighted combination; inherits from BaseStrategy.

class CompositeStrategy(BaseStrategy):
    def __init__(self, signals, weights=None, threshold=0):
        self.signals = signals
        self.num_signals = len(signals)
        if weights is None:
            self.weights = [1.0 / self.num_signals] * self.num_signals
        else:
            self.weights = weights
        self.threshold = threshold

    def get_combined_signal(self):
        weighted_signals = [w * s for w, s in zip(self.weights, self.signals)]
        combined = sum(weighted_signals)
        final_signal = combined.apply(lambda x: 1 if x > self.threshold else (-1 if x < -self.threshold else 0))
        return final_signal
'''
# src/strategies/composite_strategy.py
import numpy as np
import pandas as pd
from src.strategies.base_strategy import BaseStrategy

class CompositeStrategy(BaseStrategy):
    def __init__(self, signals, weights=None, threshold=0):
        super().__init__(signals)
        self.num_signals = len(signals)
        self.weights = weights if weights is not None else [1.0 / self.num_signals] * self.num_signals
        self.threshold = threshold

    def get_signal(self):
        weighted_signals = [w * s for w, s in zip(self.weights, self.signals)]
        combined = sum(weighted_signals)
        final_signal = combined.apply(lambda x: 1 if x > self.threshold else (-1 if x < -self.threshold else 0))
        return final_signal
'''