import time, pandas as pd, numpy as np
from numba import njit
from .data import generate_test_data

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

    result = pd.Series(padded, index=series.index)
    
    return result

    """ Test code:
        s = pd.Series([np.nan, np.nan, 1, 2, 3, 123, 3, 6])
        fast_SMA(s, window=3, exact=True)
        s.rolling(window=3).mean()
    """

def fast_std(series: pd.Series, window: int, type_exact=False, sample=True) -> pd.Series:
    """Fast standard deviation."""
    sma_x = fast_SMA(series, window, exact=type_exact)
    sma_x2 = fast_SMA(series ** 2, window, exact=type_exact)
    var = sma_x2 - sma_x ** 2

    if sample:
        if type_exact:
            correction = np.sqrt(window / (window - 1))
            result = np.sqrt(var) * correction
        else:
            n = np.minimum(np.arange(len(series)) + 1, window).astype(np.float64)
            correction = np.ones_like(n)
            mask = n > 1
            correction[mask] = np.sqrt(n[mask] / (n[mask] - 1))
            correction[~mask] = np.nan  # avoid divide by zero
            result = pd.Series(np.sqrt(var.values) * correction, index=series.index)
    else:
        result = np.sqrt(var)
    
    return result

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
    # Use boolean indexing for better performance
    z[np.isinf(z)] = np.nan

    return z

def _calculate_sr(series: pd.Series, smooth_window: int, sr_window: int, threshold: float = 10.0) -> pd.Series:
    """
    Calculate Sharpe ratio with smoothing and thresholding.    
    """
    
    if smooth_window > 1:
        d1 = fast_SMA(series, smooth_window).diff()
    else:
        d1 = series.diff()
    
    # Calculate SR on diff of smoothed data
    mean_series = fast_SMA(d1, sr_window)
    std_series = fast_std(d1, sr_window)
    
    # Sharpe ratio = mean / std with safe division and clipping in one step
    # Use np.divide with out parameter and np.clip for better performance
    sr = np.divide(mean_series, std_series, out=np.zeros_like(mean_series), where=(std_series != 0))
    sr = sr.clip(-threshold, threshold)
    
    return sr

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


# === FEATURE DEFINITIONS ARCHITECTURE ===

class Feature:
    """Base feature class with common functionality"""
    
    _FEATURE_REGISTRY = {}
    
    def __init__(self, df: pd.DataFrame, feature_type: str, **kwargs):
        self.df = df
        self.feature_type = feature_type
        self.feature = None
        
        # Common parameters
        self.PARA = {
            'winsorize_pct': kwargs.get('winsorize_pct', None),
            'method_norm': kwargs.get('method_norm', None),
            'norm_window': kwargs.get('norm_window', 1000),
        }
        
        # Feature-specific parameters
        for k, v in kwargs.items():
            self.PARA[k] = v
    
    @classmethod
    def register(cls, name):
        """Decorator to register feature calculation functions"""
        def decorator(fn):
            cls._FEATURE_REGISTRY[name] = fn
            return fn
        return decorator
    
    def calculate_feature(self):
        """Main entry: calculate → winsorize → normalize"""
        if self.feature_type not in self._FEATURE_REGISTRY:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
        
        # Run feature logic
        self._FEATURE_REGISTRY[self.feature_type](self)
        
        # Optional winsorizing
        if self.PARA.get('winsorize_pct') is not None:
            self.winsorize_feature()
        
        # Optional normalization
        if self.PARA.get('method_norm') is not None:
            self.normalize_feature()
        
        return self
    
    def winsorize_feature(self):
        """Winsorize the feature to remove outliers"""
        if self.feature is None:
            raise ValueError("Feature must be calculated before winsorizing")
        
        winsorize_pct = self.PARA.get('winsorize_pct')
        if winsorize_pct is None:
            return self
        
        lower = self.feature.quantile(winsorize_pct)
        upper = self.feature.quantile(1 - winsorize_pct)
        self.feature = self.feature.clip(lower, upper)
        return self
    
    def normalize_feature(self, method=None, window=None):
        """Apply normalization: z-score or rank"""
        if self.feature is None:
            raise ValueError("Feature must be calculated before normalization")
        
        if method is None:
            method = self.PARA.get('method_norm', None)
        if window is None:
            window = self.PARA.get('norm_window', 1000)
        
        if method is None:
            return self
        
        if method == 'rank':
            self.feature = fast_rank(self.feature, window)
        elif method == 'zscore':
            self.feature = fast_zscore(self.feature, window)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        return self
    
    def get_feature(self):
        return self.feature
    
    def get_name(self):
        """Generate feature name based on parameters"""
        # Override in subclasses or use naming function
        return f"{self.feature_type}_{self.PARA}"

# Register feature calculation functions
@Feature.register("maratio")
def feature_maratio(self):
    short = self.PARA.get('short', 5)
    long = self.PARA.get('long', 20)
    
    ma_short = fast_SMA(self.df['close'], short)
    ma_long = fast_SMA(self.df['close'], long)
    self.feature = ma_short / ma_long - 1

@Feature.register("sr")
def feature_sr(self):
    ma_window = self.PARA.get('ma_window', 5)
    sr_window = self.PARA.get('sr_window', 20)
    
    self.feature = _calculate_sr(self.df['close'], ma_window, sr_window)

def generate_feature_objects(df: pd.DataFrame, feature_definitions: dict) -> list:
    """Generate all possible Feature objects with parameter combinations"""
    import itertools
        
    feature_objects = []
    
    for feature_name, config in feature_definitions.items():
        param_names = list(config['params'].keys())
        param_values = list(config['params'].values())
        
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            
            if config['conditions'](**param_dict):
                # Create Feature object
                feature_obj = Feature(
                    df=df,
                    feature_type=feature_name,
                    winsorize_pct=config.get('winsorize_pct', None),
                    method_norm=config.get('method_norm', None),
                    norm_window=config.get('norm_window', 1000),
                    **param_dict
                )
                # Set custom name
                feature_obj.name = config['naming'](**param_dict)
                feature_objects.append(feature_obj)
    
    return feature_objects


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
    # Use boolean indexing for better performance
    self.factor[np.isinf(self.factor)] = 0


@Factor.register("ROC")
def factor_ROC(self):
    # TODO - other versions: diff, logdiff(be careful of negative values)
    window = self.PARA.get("roc_window", 10)
    print(f"Calculating ROC factor with window={window}")
    self.factor = self.full_df["Close"].pct_change(window)


if __name__ == "__main__":
    # Test caching functionality
    test_series = generate_test_data(100000, 42)['close']
    
    # Test 1: Basic functionality
    sma_10 = fast_SMA(test_series, 10)
    print(f"SMA(10) calculated: {len(sma_10)} points")
    
    # Test 2: fast_std
    std_20 = fast_std(test_series, 20)
    print(f"STD(20) calculated: {len(std_20)} points")
    
    print("All tests completed")




    # method_norm = None    
    # factor_MAX = Factor(full_df, factor_type='MAX', ma_short=3, ma_long=40, method_norm=method_norm, norm_window=7200).calculate_factor().get_factor()    
    # factor_RSI = Factor(full_df, factor_type='RSI', rsi_window=14, norm_window=7200).calculate_factor().get_factor()
    # factor_ROC = Factor(full_df, factor_type='ROC', roc_window=10, method_norm=method_norm, norm_window=7200).calculate_factor().get_factor()    
    # # combine all factors into a single dataframe
    # res = pd.DataFrame({'factor_MAX': factor_MAX, 'factor_RSI': factor_RSI, 'factor_ROC': factor_ROC,})
    # print("Factors DataFrame:")
    # print(res.tail(10))