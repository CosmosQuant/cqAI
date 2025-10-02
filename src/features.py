import pandas as pd, numpy as np
from numba import njit
from .data import generate_test_data

# ---------- base functions ----------

def configure_pandas_display():
    """Configure pandas display options for better development experience"""
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.precision', 4)
    pd.set_option('display.expand_frame_repr', False)


# ---------- core functions ----------

    @njit
def _run_sma_inner(arr: np.ndarray, window: int, exact: bool) -> np.ndarray:
        n = len(arr)
        result = np.empty(n)
        result[:] = np.nan

        if not exact:
        cumsum = 0.0
            for i in range(min(window - 1, n)):
            cumsum += arr[i]
            result[i] = cumsum / (i + 1)

        if n >= window:
        window_sum = np.sum(arr[:window])
            result[window - 1] = window_sum / window
            for i in range(window, n):
                window_sum += arr[i] - arr[i - window]
                result[i] = window_sum / window

        return result

def run_SMA(series: pd.Series, window: int, exact: bool = True) -> pd.Series:
    """
    Ultra-fast SMA with Numba backend.
    Only allows NaN at the beginning of the series.
    """
    if window <= 0:
        raise ValueError("window must be positive")

    isnan = series.isna()
    if isnan.all():
        return pd.Series(np.nan, index=series.index)

    first_valid_pos = isnan.idxmin()
    first_valid_idx = series.index.get_loc(first_valid_pos)

    if isnan.iloc[first_valid_idx:].any():
        raise ValueError("NaNs are not allowed after the first valid value.")

    arr = series.iloc[first_valid_idx:].values.astype(np.float64)
    sma_valid = _run_sma_inner(arr, window, exact)

    padded = np.full(len(series), np.nan)
    padded[first_valid_idx:] = sma_valid

    return pd.Series(padded, index=series.index)

@njit
def _rolling_std(arr: np.ndarray, window: int, exact: bool=True, ddof: int=1) -> np.ndarray:
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan

    # Special-case window == 1
    if window == 1:
        denom = 1 - ddof
        if denom <= 0:
            return out                 # ddof=1 -> NaN
        for i in range(n):
            out[i] = 0.0               # ddof=0 -> 0
        return out

    s = 0.0     # running sum
    s2 = 0.0    # running sum of squares

    for i in range(n):
        x = arr[i]
        s += x
        s2 += x * x

        if i >= window:
            x_old = arr[i - window]
            s -= x_old
            s2 -= x_old * x_old

        if exact:
            if i >= window - 1:
                n_win = window
            else:
                out[i] = np.nan
                continue
        else:
            n_win = i + 1 if i + 1 < window else window

        numer = s2 - (s * s) / n_win
        denom = n_win - ddof
        if denom > 0.0:
            var = numer / denom
            if var < 0.0:
                var = 0.0              # clamp FP noise
            out[i] = np.sqrt(var)
    else:
            out[i] = np.nan

    return out

def run_std(series: pd.Series, window: int, exact: bool=True, sample: bool=True) -> pd.Series:
    """
    Ultra-fast rolling standard deviation (single-pass, numba-accelerated).

    Constraints:
    - NaNs are only allowed at the beginning of the series.
    - exact=True: identical to pandas rolling().std(ddof=...)
    - exact=False: partial windows for first window-1 points. - default
    """
    if window <= 0:
        raise ValueError("window must be positive")

    isnan = series.isna()
    if isnan.all():
        return pd.Series(np.nan, index=series.index)

    first_valid_pos = isnan.idxmin()
    first_valid_idx = series.index.get_loc(first_valid_pos)

    if isnan.iloc[first_valid_idx:].any():
        raise ValueError("NaNs are not allowed after the first valid value.")

    valid = series.iloc[first_valid_idx:].to_numpy(dtype=np.float64)
    ddof = 1 if sample else 0
    std_valid = _rolling_std(valid, window, exact, ddof)

    out = np.full(len(series), np.nan, dtype=np.float64)
    out[first_valid_idx:] = std_valid
    return pd.Series(out, index=series.index)

def run_zscore(series: pd.Series, window: int, exact=True, sample=True, mu=None) -> pd.Series:
    """
    Rolling z-score using fast SMA and STD.

    Args:
        exact: 
            - True: output NaN for first (window-1) points (after leading NaNs)
            - False: use partial mean/std for early points
        sample: is for std, whether to use sample std (ddof=1, divided by n-1) or population std (ddof=0, divided by n)
        mu: If None, calculate rolling mean; if provided (constant number), use this value for mean

    Returns:
        pandas Series of rolling z-score
    """
    # Input validation
    if window <= 0:
        raise ValueError("window must be positive")
    
    if len(series) == 0:
        return pd.Series(dtype=float, index=series.index)
    
    if series.isna().all():
        return pd.Series(np.nan, index=series.index)
    
    # Check for non-numeric data
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError("series must be numeric (got {})".format(series.dtype))
    
    # Calculate mean and std
    if mu is None:
        mu_series = run_SMA(series, window=window, exact=exact)
    else:
        # Use provided constant mu value
        mu_series = pd.Series(mu, index=series.index)
    
    sigma = run_std(series, window=window, exact=exact, sample=sample)
    
    # Handle zero volatility proactively (more efficient than post-hoc cleanup)
    z = pd.Series(np.nan, index=series.index)
    
    # Only calculate z-score where sigma is non-zero
    valid_mask = (sigma != 0) & ~sigma.isna()
    if valid_mask.any():
        z[valid_mask] = (series[valid_mask] - mu_series[valid_mask]) / sigma[valid_mask]

    return z

def run_SR(series: pd.Series, smooth_window: int, sr_window: int, max_abs: float = 10.0) -> pd.Series:
    """
    Calculate Sharpe ratio with smoothing and thresholding.    
    
    Args:
        series: pandas Series of numeric values
            - WARNING: make sure the input is the price not the diff
        smooth_window: lookback window for smoothing
        sr_window: lookback window for SR
        max_abs: maximum absolute value for clipping


    """
    
    if smooth_window > 1:
        d1 = run_SMA(series, smooth_window).diff()
    else:
        d1 = series.diff()
    
    # Calculate SR using run_zscore with mu=0 (Sharpe ratio is mean/std, which is z-score with zero mean)
    sr = run_zscore(d1, window=sr_window, exact=True, sample=True, mu=0)
    
    # Apply thresholding
    sr = sr.clip(-max_abs, max_abs)
    
    return sr

def run_roc(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate Rate of Change (ROC) efficiently
    TODO - other versions: diff, logdiff(be careful of negative values)
    Args:
        series: Input time series
        window: Lookback window for ROC calculation
        
    Returns:
        pandas Series with ROC values: (current - past) / past
    """
    if window <= 0:
        raise ValueError("window must be positive")
    
    if len(series) == 0:
        return pd.Series(dtype='float64', index=series.index)
    
    # More efficient than pct_change for large datasets
    # Handles division by zero proactively and ensures float64 dtype
    series_shifted = series.shift(window)
    
    # Ensure consistent float64 dtype to avoid int casting issues
    series_vals = series.values.astype(np.float64)
    series_shifted_vals = series_shifted.values.astype(np.float64)
    
    roc_result = np.divide(
        series_vals - series_shifted_vals,
        series_shifted_vals,
        out=np.full_like(series_vals, np.nan, dtype=np.float64),
        where=(series_shifted_vals != 0)
    )
    return pd.Series(roc_result, index=series.index)

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
    # Step 1: Validate NaN locations (same logic as run_SMA)
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



# ---------- Feature and Signal Generator classes ----------


class Feature:
    """Base feature class with common functionality"""
    
    _FEATURE_REGISTRY = {}
    
    def __init__(self, df: pd.DataFrame, feature_type: str, **kwargs):
        self.df = df
        self.feature_type = feature_type
        self.feature = None
        
        # Common parameters
        self.PARA = {
            'winsorize': kwargs.get('winsorize', None),  # e.g. {'pct': 0.01}
            'normalize': kwargs.get('normalize', None),  # e.g. {'method': 'rank', 'window': 100}
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

    def calculate(self):
        """Main entry: calculate -> winsorize -> normalize"""
        if self.feature_type not in self._FEATURE_REGISTRY:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
        
        # Run feature logic
        self._FEATURE_REGISTRY[self.feature_type](self)
        
        # Validate that the feature function set self.feature
        if self.feature is None:
            raise ValueError(
                f"Feature function '{self.feature_type}' failed to set self.feature. "
                f"Registered functions must assign the result to self.feature, not return it."
            )
        
        # Optional winsorizing
        if self.PARA.get('winsorize') is not None:
            self.winsorize()
        
        # Optional normalization
        if self.PARA.get('normalize') is not None:
            self.normalize()
        
        return self
    
    def winsorize(self, winsorize_config=None):
        """Winsorize the feature to remove outliers"""
        if self.feature is None:
            raise ValueError("Feature must be calculated before winsorizing")
        
        if winsorize_config is None:
            winsorize_config = self.PARA.get('winsorize', None)
        
        if winsorize_config is None:
            return self

        pct = winsorize_config.get('pct')
        self.feature = winsorize(self.feature, winsor_pct=pct)
        return self

    def normalize(self, normalize_config=None):
        """Apply normalization: z-score or rank"""
        if self.feature is None:
            raise ValueError("Feature must be calculated before normalization")
        
        if normalize_config is None:
            normalize_config = self.PARA.get('normalize', None)
        
        if normalize_config is None:
            return self
        
        method = normalize_config.get('method')
        window = normalize_config.get('window')

        if method is None:
            raise ValueError("normalize config must specify 'method'")
        if window is None:
            raise ValueError("normalize config must specify 'window'")

        if method == 'rank':
            self.feature = fast_rank(self.feature, window)
        elif method == 'zscore':
            self.feature = run_zscore(self.feature, window)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        return self

    def get_feature(self):
        return self.feature
    
    def get_name(self):
        """Generate feature name based on parameters"""
        # Start with base name (if custom name exists, use it)
        if hasattr(self, 'name'):
            base_name = self.name
        else:
            base_name = self.feature_type
            
            # Add feature-specific parameters (exclude processing parameters)
            feature_params = []
            exclude_keys = {'winsorize', 'normalize'}
            for key, value in self.PARA.items():
                if key not in exclude_keys:
                    feature_params.append(str(value))
            
            if feature_params:
                base_name += '_' + '_'.join(feature_params)
        
        # Add winsorize suffix if present
        if self.PARA.get('winsorize') is not None:
            winsorize_config = self.PARA['winsorize']
            pct = winsorize_config.get('pct')
            if pct is not None:
                base_name += f"_wzr{int(pct*100)}"
        
        # Add normalize suffix if present
        if self.PARA.get('normalize') is not None:
            normalize_config = self.PARA['normalize']
            method = normalize_config.get('method')
            window = normalize_config.get('window')
            if method is not None and window is not None:
                base_name += f"_{method}{window}"
        
        return base_name

def create_features(df: pd.DataFrame, feature_combo: dict) -> list:
    """Generate all possible Feature objects with parameter combinations"""
    import itertools
        
    feature_objects = []
    
    for feature_name, config in feature_combo.items():
        param_names = list(config['params'].keys())
        param_values = list(config['params'].values())
        
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            
            if config['conditions'](**param_dict):
                # Check for parameter conflicts
                conflicts = []
                if 'winsorize' in param_dict and 'winsorize' in config:
                    conflicts.append('winsorize')
                if 'normalize' in param_dict and 'normalize' in config:
                    conflicts.append('normalize')
                
                if conflicts:
                    raise ValueError(
                        f"Parameter conflict in feature '{feature_name}': "
                        f"{conflicts} found in both 'params' and config level. "
                        f"Move these parameters to config level only."
                    )
                
                # Create Feature object
                feature_obj = Feature(
                    df=df,
                    feature_type=feature_name,
                    winsorize=config.get('winsorize', None),
                    normalize=config.get('normalize', None),
                    **param_dict
                )
                # Set custom name if provided, otherwise use default get_name()
                if 'naming' in config:
                    feature_obj.name = config['naming'](**param_dict)
                feature_objects.append(feature_obj)
    
    # Check for duplicate feature names
    _check_duplicate_names(feature_objects)
    
    return feature_objects

def _check_duplicate_names(feature_objects: list) -> None:
    """Check for duplicate feature names and warn if found"""
    import warnings
    from collections import defaultdict
    
    name_counts = defaultdict(int)
    name_to_features = defaultdict(list)
    
    for feature_obj in feature_objects:
        name = getattr(feature_obj, 'name', feature_obj.get_name())
        name_counts[name] += 1
        name_to_features[name].append(feature_obj)
    
    # Find duplicates
    duplicates = {name: count for name, count in name_counts.items() if count > 1}
    
    if duplicates:
        warnings.warn(
            f"Duplicate feature names detected! This will cause DataFrame column overwrites:\n" +
            "\n".join([
                f"  '{name}' appears {count} times"
                for name, count in duplicates.items()
            ]) + 
            "\nConsider using different 'naming' functions to create unique names.",
            UserWarning,
            stacklevel=3
        )

# TODO - rolling vs global; other winsorize methods
def winsorize(series: pd.Series, winsor_pct: float | None = None, *, copy: bool = True) -> pd.Series:
    """Clip to [p, 1-p] quantiles; robust to NANs; fast on large inputs."""
    if winsor_pct is None or winsor_pct == 0:
        return series.copy() if copy else series
    
    if not (0 < winsor_pct < 0.5):
        raise ValueError(f"winsor_pct must be in (0, 0.5), got {winsor_pct}")
    
    # Work on a numeric ndarray once; copy controls whether we allocate a new buffer
    s = series.to_numpy(dtype="float64", copy=copy)
    
    # Fast quantiles that ignore NANs
    q = np.nanquantile(s, [winsor_pct, 1 - winsor_pct])
    lower, upper = float(q[0]), float(q[1])
    
    # If quantiles are NaN (empty/all-NaN series), just return input unchanged
    if np.isnan(lower) or np.isnan(upper):
        return series.copy() if copy else series
    
    # In-place clip (if copy=False this mutates original array)
    np.clip(s, lower, upper, out=s)
    return pd.Series(s, index=series.index, name=series.name)

# TODO - other normalization methods
def discretize(series: pd.Series, method: str, threshold: float = 0.0, bin: int = 5) -> pd.Series:
    """
    Convert continuous values to discrete labels using various methods.
    
    Args:
        series: Input pandas Series of numeric values
        method: Discretization method ('binary' or 'quantile')
        threshold: Threshold for binary classification (default: 0.0)
        bin: Number of quantile bins for quantile method (default: 5)
    
    Returns:
        pandas Series with float64 dtype containing discrete labels
        
    Notes:
        - NaN values are preserved in the output (not modified or removed)
        - Binary method: >= threshold -> 1.0, < threshold -> -1.0
        - Quantile method: Equal-frequency bins centered around 0.0
        - Both methods return float64 dtype for consistency with feature pipeline
        - Values are discrete but stored as float for compatibility
    """
    if method == 'binary':
        # Binary classification with threshold - always use >= for 1, < for -1
        # Use np.where for better performance and cleaner code
        discrete_label = np.where(series >= threshold, 1.0, -1.0)
        # Preserve NaN values
        discrete_label = np.where(series.isna(), np.nan, discrete_label)
        return pd.Series(discrete_label, index=series.index, dtype='float64')
        
    elif method == 'quantile':
        # Multi-class quantile-based classification (equal-frequency bins)
        # pd.qcut automatically handles NaN values (preserves them)
        quantile_labels = pd.qcut(series, q=bin, labels=False, duplicates='drop')
        # Center around 0 and convert to float64
        centered_labels = quantile_labels - (bin // 2)
        return centered_labels.astype('float64')
        
        else:
        raise ValueError(f"Unknown method '{method}'. Supported: 'binary', 'quantile'.")



# TODO - add more features
# TODO - use cache to avoid re-calculating the same feature.
# But have to be careful with sr(close) vs sr(close.diff()) vs sr(open)
@Feature.register("maratio")
def feature_maratio(self):
    short = self.PARA.get('short', 5)
    long = self.PARA.get('long', 20)
    
    ma_short = run_SMA(self.df['close'], short)
    ma_long = run_SMA(self.df['close'], long)
    self.feature = ma_short / ma_long - 1

@Feature.register("sr")
def feature_sr(self):
    ma_window = self.PARA.get('ma_window', 5)
    sr_window = self.PARA.get('sr_window', 20)
    
    self.feature = run_SR(self.df['close'], ma_window, sr_window)

@Feature.register("rsi")
def feature_rsi(self):
    """
    RSI (Relative Strength Index) feature
    Value is transformed to be between -1 and +1, corresponding to RSI value between 0 and 100
    """
    window = self.PARA.get("window", 14)
    delta = self.df["close"].diff()
    
    # Optimize: use numpy arrays to avoid multiple pandas operations
    delta_values = delta.values
    gain_values = np.where(delta_values > 0, delta_values, 0.0)
    loss_values = np.where(delta_values < 0, -delta_values, 0.0)
    
    # Convert back to Series for run_SMA (maintains index alignment)
    gain_series = pd.Series(gain_values, index=delta.index)
    loss_series = pd.Series(loss_values, index=delta.index)
    
    gain = run_SMA(gain_series, window)
    loss = run_SMA(loss_series, window)
    
    # RSI calculation with proper edge case handling
    gain_vals = gain.values
    loss_vals = loss.values
    
    # Handle different scenarios correctly:
    # - avg_loss == 0 且 avg_gain > 0 => RS = ∞ => RSI = 100
    # - avg_loss == 0 且 avg_gain == 0 => RSI = 50 (完全无波动)
    # - avg_gain == 0 且 avg_loss > 0 => RSI = 0
    rsi = np.full_like(gain_vals, np.nan)
    
    # Normal case: both gain and loss are non-zero
    normal_mask = (loss_vals != 0)
    rs_normal = gain_vals[normal_mask] / loss_vals[normal_mask]
    rsi[normal_mask] = 100 - (100 / (1 + rs_normal))
    
    # Edge case: loss == 0 but gain > 0 => RSI = 100
    rsi[(loss_vals == 0) & (gain_vals > 0)] = 100.0
    
    # Edge case: loss == 0 and gain == 0 => RSI = 50 (neutral)
    rsi[(loss_vals == 0) & (gain_vals == 0)] = 50.0
    
    # Edge case: gain == 0 but loss > 0 => RSI = 0
    rsi[(gain_vals == 0) & (loss_vals > 0)] = 0.0
    
    # Transform to [-1, +1] range
    self.feature = pd.Series((rsi - 50) / 50, index=delta.index)

@Feature.register("roc")
def feature_roc(self):
    window = self.PARA.get("window", 10)
    self.feature = run_roc(self.df["close"], window)



# ---------- Feature Operators ----------

class FeatureOperator:
    """Base class for feature post-processing operations"""
    def __init__(self, **kwargs):
        self.PARA = kwargs
    
    def apply(self, series: pd.Series) -> pd.Series:
        """Apply transformation to series"""
        raise NotImplementedError
    
    def get_name(self) -> str:
        """Generate operator name for chaining"""
        params = ','.join(f"{k}={v}" for k, v in self.PARA.items())
        return f"{self.__class__.__name__.lower()}[{params}]"

class ReLU(FeatureOperator):
    """ReLU activation operator"""
    def __init__(self, threshold=0.0, direction='both'):
        super().__init__(threshold=threshold, direction=direction)
    
    def apply(self, series: pd.Series) -> pd.Series:
        threshold = self.PARA['threshold']
        direction = self.PARA['direction']
        
        if direction.lower() == 'high':
            result = series.clip(lower=0) * (series > threshold)
        elif direction.lower() == 'low':
            result = -series.clip(upper=0) * (series < threshold)
        elif direction.lower() == 'both':
            result = series.clip(lower=0) * (series > threshold) - series.clip(upper=0) * (series < -threshold)
        else:
            raise ValueError("Unsupported direction")
        
        return pd.Series(result, index=series.index, dtype='float64')

class Clip(FeatureOperator):
    """Clip values to specified range"""
    def __init__(self, low=None, high=None):
        super().__init__(low=low, high=high)
    
    def apply(self, series: pd.Series) -> pd.Series:
        return series.clip(self.PARA['low'], self.PARA['high'])

class Scale(FeatureOperator):
    """Scale values by multiplication and addition"""
    def __init__(self, mul=1.0, add=0.0):
        super().__init__(mul=mul, add=add)
    
    def apply(self, series: pd.Series) -> pd.Series:
        return series * self.PARA['mul'] + self.PARA['add']

class Sign(FeatureOperator):
    """Extract sign of values"""
    def __init__(self):
        super().__init__()
    
    def apply(self, series: pd.Series) -> pd.Series:
        return pd.Series(np.sign(series), index=series.index, dtype='float64')

class Abs(FeatureOperator):
    """Absolute value"""
    def __init__(self):
        super().__init__()
    
    def apply(self, series: pd.Series) -> pd.Series:
        return series.abs()

class Lag(FeatureOperator):
    """Lag/shift values"""
    def __init__(self, k=1):
        super().__init__(k=k)
    
    def apply(self, series: pd.Series) -> pd.Series:
        return series.shift(self.PARA['k'])

class Smooth(FeatureOperator):
    """Smooth using moving average"""
    def __init__(self, window=5, method='sma'):
        super().__init__(window=window, method=method)
    
    def apply(self, series: pd.Series) -> pd.Series:
        window = self.PARA['window']
        method = self.PARA['method']
        
        if method == 'sma':
            return run_SMA(series, window)
        else:
            raise ValueError(f"Unsupported smoothing method: {method}")

class Crossover(FeatureOperator):
    """Crossover signal generator"""
    def __init__(self, threshold=0.0, direction='both'):
        super().__init__(threshold=threshold, direction=direction)
    
    def apply(self, series: pd.Series) -> pd.Series:
        threshold = self.PARA['threshold']
        direction = self.PARA['direction']
        
        if direction == 'high':
            signal = ((series.shift(1) <= threshold) & (series > threshold)).astype(float)
        elif direction == 'low':
            signal = -((series.shift(1) > threshold) & (series <= threshold)).astype(float)
        elif direction == 'both':
            signal = ((series.shift(1) <= threshold) & (series > threshold)).astype(float) - \
                    ((series.shift(1) > -threshold) & (series <= -threshold)).astype(float)
        else:
            raise ValueError("Unsupported direction")
        
        return pd.Series(signal, index=series.index, dtype='float64')

def apply_operators(series: pd.Series, operators: list[FeatureOperator]) -> pd.Series:
    """Apply a list of operators sequentially"""
    result = series
    for op in operators:
        result = op.apply(result)
    return result


if __name__ == "__main__":
    # Test caching functionality
    test_series = generate_test_data(100000, 42)['close']
    
    # Test 1: Basic functionality
    sma_10 = run_SMA(test_series, 10)
    print(f"SMA(10) calculated: {len(sma_10)} points")
    
    # Test 2: run_std
    std_20 = run_std(test_series, 20)
    print(f"STD(20) calculated: {len(std_20)} points")
    
    print("All tests completed")


