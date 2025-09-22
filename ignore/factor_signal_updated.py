import os, datetime, time, itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numba import jit




def zscore_single(x):
    return (x.iloc[-1] - x.mean()) / x.std() if x.std() != 0 else 0


@jit(nopython=True)
def fast_SMA_numpy(data, window):
    """
    Ultra-fast moving average using Numba JIT compilation.
    Returns numpy array for internal calculations.
    """
    n = len(data)
    result = np.empty(n)
    
    # Handle edge cases
    for i in range(min(window - 1, n)):
        result[i] = np.mean(data[:i+1])
    
    # Main calculation using sliding window
    if n >= window:
        # Calculate first full window
        window_sum = 0.0
        for i in range(window):
            window_sum += data[i]
        result[window-1] = window_sum / window
        
        # Slide window for remaining elements
        for i in range(window, n):
            window_sum = window_sum - data[i-window] + data[i]
            result[i] = window_sum / window
    
    return result

def fast_SMA(data: pd.Series, window: int) -> pd.Series:
    """
    Ultra-fast moving average using Numba JIT compilation.
    Returns pandas Series with proper index.
    
    Args:
        data: pandas Series with numeric values
        window: window size for moving average
        
    Returns:
        pandas Series with moving average values
    """
    result_numpy = fast_SMA_numpy(data.values, window)
    return pd.Series(result_numpy, index=data.index)


"""

signal functions - input is full_df, output is signals
make sure the full_df has the following columns: Open, High, Low, Close, Volume

signal has two types:
1. yhat - from any forcast function
2. rule-based - e.g. if MA1 > MA2, then long, hold till some cnodition; if MA1 < MA2, then short, hold till some other condition

"""

class Factor:
    """Modular, extensible factor calculator with plugin support."""

    _FACTOR_REGISTRY = {}

    def __init__(self, full_df: pd.DataFrame, factor_type='MAX', **kwargs):
        self.full_df = full_df
        self.factor_type = factor_type
        self.factor = None

        # Parameter container (easy to extend)
        self.PARA = {
            'factor_type': factor_type,
            'method_clean': kwargs.get('method_clean', None),
            'winsorize_pct': kwargs.get('winsorize_pct', 0.01),
            'method_norm': kwargs.get('method_norm', None),
            'norm_window': kwargs.get('norm_window', 1000),
        }

        # Allow arbitrary kwargs to go into PARA for extensibility
        for k, v in kwargs.items():
            self.PARA[k] = v

    @classmethod
    def register(cls, name):
        """Decorator to register a new factor calculation function"""
        def decorator(fn):
            cls._FACTOR_REGISTRY[name] = fn
            return fn
        return decorator

    def clean_factor(self, winsorize_pct=None):
        """Winsorize the factor"""
        if self.factor is None:
            raise ValueError("Factor must be calculated before cleaning")

        if winsorize_pct is None:
            winsorize_pct = self.PARA['winsorize_pct']

        lower = self.factor.quantile(winsorize_pct)
        upper = self.factor.quantile(1 - winsorize_pct)
        self.factor = self.factor.clip(lower, upper)
        return self

    def normalize_factor(self, method=None, window=None):
        """Apply normalization: z-score or rank"""
        if self.factor is None:
            raise ValueError("Factor must be calculated before normalization")

        if method is None:
            method = self.PARA['method_norm']
        if window is None:
            window = self.PARA['norm_window']

        if method is None:
            return self

        if method == 'rank':
            self.factor = self.factor.rolling(window).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5,
                raw=False
            )
        elif method == 'zscore':
            self.factor = self.factor.rolling(window).apply(
                lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() != 0 else 0,
                raw=False
            )
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        return self

    def calculate_factor(self):
        """Main entry: calculate → clean → normalize"""
        if self.factor_type not in self._FACTOR_REGISTRY:
            raise ValueError(f"Unsupported factor type: {self.factor_type}")
        
        # Run factor logic
        self._FACTOR_REGISTRY[self.factor_type](self)

        # Optional cleaning
        if self.PARA['method_clean'] is not None:
            self.clean_factor()

        # Optional normalization
        if self.PARA['method_norm'] is not None:
            self.normalize_factor()

        return self

    def get_factor(self):
        return self.factor


class SignalGenerator:
    """Generate trading signals from preprocessed factors"""
    
    def __init__(self, factor: pd.Series, **kwargs):
        self.factor = factor
        self.signal = None

        # Define all parameters in PARA dictionary with default values
        self.PARA = {
            'operator': kwargs.get('operator', 'crossover'),
            'operator_threshold': kwargs.get('operator_threshold', 0.0),
            'direction': kwargs.get('direction', 'long')
        }
    
    def operator_crossover(self, threshold=0, direction='long'):
        """Crossover operator"""
        if direction == 'long':
            self.signal = ((self.factor.shift(1) <= threshold) & (self.factor > threshold)).astype(int)
        elif direction == 'short':
            self.signal = -((self.factor.shift(1) <= threshold) & (self.factor < threshold)).astype(int)
        elif direction == 'both':
            self.signal = ((self.factor.shift(1) <= threshold) & (self.factor > threshold)).astype(int) - \
                         ((self.factor.shift(1) >= threshold) & (self.factor < threshold)).astype(int)
        else:
            raise ValueError("Unsupported direction")
        return self
    
    def operator_ReLU(self, threshold=0, direction='long'):
        """ReLU operator"""
        if direction == 'long':
            self.signal = self.factor.clip(lower=0)  
        elif direction == 'short':
            self.signal = -self.factor.clip(upper=0)
        elif direction == 'both':
            self.signal = self.factor.clip(lower=0) - self.factor.clip(upper=0)
        else:
            raise ValueError("Unsupported direction")
        return self
    
    def to_signal(self):
        """Generate signal using specified operator"""
        if self.PARA['operator'] == 'crossover':
            self.operator_crossover(threshold=self.PARA['operator_threshold'], direction=self.PARA['direction'])
        elif self.PARA['operator'] == 'ReLU':
            self.operator_ReLU(threshold=self.PARA['operator_threshold'], direction=self.PARA['direction'])
        else:
            raise ValueError("Unsupported signal generation method")
        return self

    def get_signal(self):
        return self.signal


# === REGISTERED FACTORS ===

@Factor.register("MAX")
def factor_MAX(self):
    short = self.PARA.get("ma_short", 3)
    long = self.PARA.get("ma_long", 100)
    print(f"Calculating MAX factor with short={short} and long={long}")
    
    # Use fast moving average that returns pandas Series
    close_prices = self.full_df["Close"]
    ma_short = fast_SMA(close_prices, short)
    print("ma_short first 5 values:")
    print(ma_short.head())
    ma_long = fast_SMA(close_prices, long)
    print("ma_long first 5 values:")
    print(ma_long.head())
    
    self.factor = ma_short / ma_long - 1


@Factor.register("RSI")
def factor_RSI(self):
    window = self.PARA.get("rsi_window", 14)
    print(f"Calculating RSI factor with window={window}")
    delta = self.full_df["Close"].diff()
    
    # Use fast_SMA instead of rolling().mean() for better performance
    gain_series = delta.where(delta > 0, 0)
    loss_series = -delta.where(delta < 0, 0)
    
    gain = fast_SMA(gain_series, window)
    loss = fast_SMA(loss_series, window)
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    self.factor = (rsi - 50) / 50
    self.factor = self.factor.replace([np.inf, -np.inf], 0)


@Factor.register("ROC")
def factor_ROC(self):
    window = self.PARA.get("roc_window", 10)
    print(f"Calculating ROC factor with window={window}")
    self.factor = self.full_df["Close"].pct_change(window)


if __name__ == "__main__":
    # Create sample price data for testing
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    full_df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Load parquet file (much better than pickle)
    full_df = pd.read_parquet('crypto_df.parquet', engine='pyarrow')


    print("Testing Factor + SignalGenerator integration...")

    start_time = time.time()    # Create factor and generate signals
    factor_max = Factor(full_df, factor_type='MAX', ma_short=3, ma_long=40, norm_window=100).calculate_factor()
    signals = SignalGenerator(factor=factor_max.get_factor(), operator='crossover', operator_threshold=0, direction='both').to_signal().get_signal()
    print(f"Factor calculation time: {time.time() - start_time} seconds")
    
    factor_rsiL = Factor(full_df, factor_type='RSI', rsi_window=14, norm_window=100).calculate_factor()
    signals_rsi = SignalGenerator(factor=factor_rsiL.get_factor(), operator='crossover', operator_threshold=0, direction='both').to_signal().get_signal()
    print(f"Factor calculation time: {time.time() - start_time} seconds")
    
    
    
    print(f"   Signal range: {signals.min()} to {signals.max()}")
    print(f"   Buy signals: {(signals > 0).sum()}")
    print(f"   Sell signals: {(signals < 0).sum()}")

    # TODO - need to rename long, short, both to high, low, net 