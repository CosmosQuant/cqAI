import os, datetime, time, itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numba import njit

# Set pandas display options to show more columns
pd.set_option('display.max_columns', 10); pd.set_option('display.width', 1000); pd.set_option('display.max_colwidth', 1000)


def fast_SMA(series: pd.Series, window: int, exact=False) -> pd.Series:
    """
    Ultra-fast moving average using Numba JIT compilation.

    Allows NaNs only at the beginning of the series.

    Args:
        series: pandas Series
        window: window size
        exact: 
            - True: first (window-1) values are NaN
            - False: use partial average for first (window-1) values

    Returns:
        pandas Series with SMA
    """
    isnan = series.isna()
    first_valid_pos = isnan.idxmin()  # First non-NaN index label
    first_valid_idx = series.index.get_loc(first_valid_pos)  # Position (int)

    # Check: only initial values are NaN
    if not isnan.iloc[:first_valid_idx].all():
        raise ValueError("NaNs are only allowed at the beginning of the series.")
    if isnan.iloc[first_valid_idx:].any():
        raise ValueError("NaNs are not allowed in the middle or end of the series.")

    # Work on valid part only
    valid_series = series.iloc[first_valid_idx:]

    @njit
    def _inner(arr, window, exact):
        n = len(arr)
        result = np.empty(n)
        result[:] = np.nan

        if not exact:
            for i in range(min(window - 1, n)):
                result[i] = np.mean(arr[:i + 1])

        if n >= window:
            window_sum = 0.0
            for i in range(window):
                window_sum += arr[i]
            result[window - 1] = window_sum / window

            for i in range(window, n):
                window_sum += arr[i] - arr[i - window]
                result[i] = window_sum / window

        return result

    arr = valid_series.values.astype(np.float64)
    sma_valid = _inner(arr, window, exact)

    # Pad front NaNs
    n_pad = first_valid_idx
    padded = np.full(len(series), np.nan)
    padded[n_pad:] = sma_valid

    return pd.Series(padded, index=series.index)

    """ Test code:
        s = pd.Series([np.nan, np.nan, 1, 2, 3, 123, 3, 6])
        fast_SMA(s, window=3, exact=True)
        s.rolling(window=3).mean()
    """

def fast_std(series: pd.Series, window: int, type_exact=False, sample=True) -> pd.Series:
    sma_x = fast_SMA(series, window, exact=type_exact)
    sma_x2 = fast_SMA(series ** 2, window, exact=type_exact)
    var = sma_x2 - sma_x ** 2

    if sample:
        if type_exact:
            correction = np.sqrt(window / (window - 1))
            return np.sqrt(var) * correction
        else:
            n = np.minimum(np.arange(len(series)) + 1, window).astype(np.float64)
            correction = np.ones_like(n)
            mask = n > 1
            correction[mask] = np.sqrt(n[mask] / (n[mask] - 1))
            correction[~mask] = np.nan  # avoid divide by zero
            return pd.Series(np.sqrt(var.values) * correction, index=series.index)
    else:
        return np.sqrt(var)

def fast_zscore(series: pd.Series, window: int, exact=False, sample=True) -> pd.Series:
    """
    Rolling z-score using fast SMA and STD.

    Args:
        series: input time series
        window: rolling window size
        exact: 
            - True: output NaN for first (window-1) points (after leading NaNs)
            - False: use partial mean/std for early points
        sample: whether to use sample std (ddof=1)

    Returns:
        pandas Series of rolling z-score
    """
    mu = fast_SMA(series, window=window, exact=exact)
    sigma = fast_std(series, window=window, type_exact=exact, sample=sample)

    z = (series - mu) / sigma
    z = z.replace([np.inf, -np.inf], np.nan)

    return z

def fast_rank(series: pd.Series, window: int, exact=False) -> pd.Series:
    """
    Fast rolling rank (percentile) using Numba.

    Args:
        series: pandas Series of numeric values
        window: lookback window
        exact: 
            - True: first (window-1) values = NaN
            - False: use partial window for first values

    Returns:
        pandas Series of rolling rank in [0,1]
    """
    # Step 1: Validate NaN locations (same logic as fast_SMA)
    isnan = series.isna()
    first_valid_pos = isnan.idxmin()  # first non-NaN index label
    first_valid_idx = series.index.get_loc(first_valid_pos)  # integer position

    if not isnan.iloc[:first_valid_idx].all():
        raise ValueError("NaNs are only allowed at the beginning of the series.")
    if isnan.iloc[first_valid_idx:].any():
        raise ValueError("NaNs are not allowed in the middle or end of the series.")

    valid_series = series.iloc[first_valid_idx:]

    @njit
    def _inner(arr, window, exact):
        n = len(arr)
        result = np.empty(n)
        result[:] = np.nan

        for i in range(n):
            lookback = min(i + 1, window) if not exact else window
            if i < window - 1 and exact:
                result[i] = np.nan
            else:
                count = 0
                for j in range(i - lookback + 1, i):
                    if j >= 0 and arr[j] <= arr[i]:
                        count += 1
                result[i] = (count + 1) / lookback
        return result

    arr = valid_series.values.astype(np.float64)
    rank_valid = _inner(arr, window, exact)

    # Pad front NaNs to match original index
    n_pad = first_valid_idx
    padded = np.full(len(series), np.nan)
    padded[n_pad:] = rank_valid

    return pd.Series(padded, index=series.index)


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
            self.factor = fast_rank(self.factor, window)
        elif method == 'zscore':
            self.factor = fast_zscore(self.factor, window)
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
            'operator': kwargs.get('operator', 'original'),
            'operator_threshold': kwargs.get('operator_threshold', 0.0),
            'direction': kwargs.get('direction', 'high')
        }
    
    def operator_original(self):
        """Original operator"""
        self.signal = self.factor
        return self
    
    def operator_crossover(self, threshold=0, direction='high'):
        """Crossover operator
        direction: high, low, both
        threshold: threshold for crossover
        signal is 1 if the factor is above the threshold and the previous factor was below the threshold
        signal is -1 if the factor is below the threshold and the previous factor was above the threshold
        if direction is both, then signal is the combination of the 'high' with threshold and 'low' with -threshold
        """
        if direction == 'high':
            self.signal = ((self.factor.shift(1) <= threshold) & (self.factor > threshold)).astype(int)
        elif direction == 'low':
            self.signal = -((self.factor.shift(1) > threshold) & (self.factor <= threshold)).astype(int)
        elif direction == 'both':            
            self.signal = ((self.factor.shift(1) <= threshold) & (self.factor > threshold)).astype(int) - \
                         ((self.factor.shift(1) > -threshold) & (self.factor <= -threshold)).astype(int)
        else:
            raise ValueError("Unsupported direction")
        return self
    
    def operator_ReLU(self, threshold=0, direction='high'):
        """ReLU operator"""
        if direction.lower() == 'high':
            self.signal = self.factor.clip(lower=0) * (self.factor > threshold)
        elif direction.lower() == 'low':
            self.signal = -self.factor.clip(upper=0) * (self.factor < threshold)
        elif direction.lower() == 'both':
            self.signal = self.factor.clip(lower=0) * (self.factor > threshold) - self.factor.clip(upper=0) * (self.factor < -threshold)
        else:
            raise ValueError("Unsupported direction")
        return self
    
    def to_signal(self):
        """Generate signal using specified operator"""
        if self.PARA['operator'].lower() == 'original':
            self.operator_original()
        elif self.PARA['operator'].lower() == 'crossover':
            self.operator_crossover(threshold=self.PARA['operator_threshold'], direction=self.PARA['direction'])
        elif self.PARA['operator'].lower() == 'relu':
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
    ma_long = fast_SMA(close_prices, long)
    
    self.factor = ma_short / ma_long - 1


@Factor.register("RSI")
def factor_RSI(self):
    """
    factor value is transformed to be between -1 and +1, corresponding to rsi value between 0 and 100
    """
    
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
    # TODO - other versions: diff, logdiff(be careful of negative values)
    window = self.PARA.get("roc_window", 10)
    print(f"Calculating ROC factor with window={window}")
    self.factor = self.full_df["Close"].pct_change(window)


if __name__ == "__main__":
    # Create sample price data for testing
    np.random.seed(42)
    periods = 100000
    dates = pd.date_range('2025-01-01', periods=periods, freq='T')
    prices = 100 + np.cumsum(np.random.randn(periods) * 0.5)
    full_df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, periods)
    }, index=dates)
    
    # Load parquet file (much better than pickle)
    # full_df = pd.read_parquet('crypto_df.parquet', engine='pyarrow')    
    
    full_df = full_df.dropna(axis=0)


    print("Testing Factor + SignalGenerator integration...")

    start_time = time.time()

    method_norm = None
    
    # MAX - moving average crossover
    factor_MAX = Factor(full_df, factor_type='MAX', ma_short=3, ma_long=40, method_norm=method_norm, norm_window=7200).calculate_factor().get_factor()
    signal_MAX_original = SignalGenerator(factor_MAX, operator='original').to_signal().get_signal()
    signal_MAX_high = SignalGenerator(factor_MAX, operator='crossover', operator_threshold=0, direction='high').to_signal().get_signal()
    signal_MAX_low = SignalGenerator(factor_MAX, operator='crossover', operator_threshold=0, direction='low').to_signal().get_signal()
    signal_MAX_both = SignalGenerator(factor_MAX, operator='crossover', operator_threshold=0, direction='both').to_signal().get_signal()
    
    # RSI - relative strength index
    factor_RSI = Factor(full_df, factor_type='RSI', rsi_window=14, norm_window=7200).calculate_factor().get_factor()
    signal_RSI_original = SignalGenerator(factor_RSI, operator='original').to_signal().get_signal()
    signal_RSI_high = SignalGenerator(factor_RSI, operator='crossover', operator_threshold=0, direction='high').to_signal().get_signal()
    signal_RSI_low = SignalGenerator(factor_RSI, operator='crossover', operator_threshold=0, direction='low').to_signal().get_signal()
    signal_RSI_both = SignalGenerator(factor_RSI, operator='crossover', operator_threshold=0, direction='both').to_signal().get_signal()

    # ROC - rate of change
    factor_ROC = Factor(full_df, factor_type='ROC', roc_window=10, method_norm=method_norm, norm_window=7200).calculate_factor().get_factor()
    signal_ROC_original = SignalGenerator(factor_ROC, operator='original').to_signal().get_signal()
    
    # combine all signals into a single dataframe
    res = pd.DataFrame({
        'signal_MAX_original': signal_MAX_original,
        'signal_MAX_high': signal_MAX_high,
        'signal_MAX_low': signal_MAX_low,
        'signal_MAX_both': signal_MAX_both,
        'signal_RSI_original': signal_RSI_original,
        'signal_RSI_high': signal_RSI_high,
        'signal_RSI_low': signal_RSI_low,
        'signal_RSI_both': signal_RSI_both,
        'signal_ROC_original': signal_ROC_original,
    })
    
    
    print("Signals DataFrame:")
    print(res.tail(10))  # Show last 10 rows instead of 0    
    print("Value range of signals:")
    print(res.signal_MAX_original.min(), res.signal_MAX_original.max())
    print(res.signal_RSI_original.min(), res.signal_RSI_original.max())
    print(res.signal_ROC_original.min(), res.signal_ROC_original.max())





