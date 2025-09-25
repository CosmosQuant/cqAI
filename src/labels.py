import pandas as pd
import numpy as np
from .features import discretize, winsorize

class Label:
    """
    Base class for label generation with registry pattern.
    
    Be careful - always use forward-looking targets, to avoid look-ahead bias
    """
    _LABEL_REGISTRY = {}
    
    def __init__(self, df: pd.DataFrame, label_type: str, **kwargs):
        self.df = df
        self.label_type = label_type
        self.label = None
        self.PARA = {
            # Common parameters only
            'lag': kwargs.get('lag', 0),                  # Time lag offset
            'LB': kwargs.get('LB', 1),                  # Forward periods
            
            # Processing method parameters (optional)
            'rescale': kwargs.get('rescale', None),  # Risk adjustment method and parameters
            
            'winsorize_pct': kwargs.get('winsorize_pct', None),  # Outlier winsorization
            
            'discretize': kwargs.get('discretize', None),  # Discretization method and parameters
            
        }
        for k, v in kwargs.items():
            self.PARA[k] = v
    
    @classmethod
    def register(cls, name):
        def decorator(fn):
            cls._LABEL_REGISTRY[name] = fn
            return fn
        return decorator
    
    def calculate(self):
        # Step 1: Calculate base forward return (always)
        if self.label_type not in self._LABEL_REGISTRY:
            raise ValueError(f"Unsupported label type: {self.label_type}")
        self._LABEL_REGISTRY[self.label_type](self)
        
        # Step 2: Apply risk adjustment (optional, similar to normalize_feature)
        if self.PARA.get('rescale') is not None:
            self._apply_risk_adjustment()
        
        # Step 3: Apply discretization (optional, final processing step)
        if self.PARA.get('discretize') is not None:
            self.discretize()
        
        # Step 4: Apply winsorization (optional, same as winsorize_feature)
        if self.PARA.get('winsorize_pct') is not None:
            self.winsorize_label()
        
        return self
    
    def _apply_risk_adjustment(self):
        """Apply risk adjustment to raw returns (similar to normalize_feature)."""
        if self.label is None:
            raise ValueError("Label must be calculated before risk adjustment")
        
        rescale_config = self.PARA.get('rescale')
        if rescale_config is not None:
            method = rescale_config.get('method')
            nscale = rescale_config.get('nscale', 20)
            
            if method == 'volatility':
                # Return per unit volatility (Sharpe-like)
                rolling_vol = self.label.rolling(nscale).std()
                self.label = self.label / rolling_vol
            # TODO - add more adj methods later
            else:
                raise ValueError(f"Unknown risk adjustment method: {method}")
    
    def discretize(self):
        """Apply discretization to continuous values (final processing step)."""
        if self.label is None:
            raise ValueError("Label must be calculated before discretization")
        
        discretize_config = self.PARA.get('discretize')
        if discretize_config is not None:
            method = discretize_config.get('method')
            if method:
                # Extract method and pass remaining parameters
                params = {k: v for k, v in discretize_config.items() if k != 'method'}
                self.label = discretize(self.label, method=method, **params)
    
    def winsorize_label(self):
        """Winsorize extreme values using percentile-based approach (same as winsorize_feature)."""
        if self.label is None:
            raise ValueError("Label must be calculated before winsorizing")
        
        winsorize_pct = self.PARA.get('winsorize_pct')
        if winsorize_pct is None:
            return self
        
        self.label = winsorize(self.label, winsorize_pct)
        return self
    
    def get_label(self):
        return self.label
    
    def get_name(self):
        """Automatically generate name from PARA parameters"""
        # Always include label_type
        name_parts = [self.label_type]
        
        # Add parameters (excluding None values and common defaults)
        exclude_keys = {'lag', 'LB'}  # Skip these keys
        exclude_values = {None}  # Skip these values
        
        # Skip common default parameter combinations
        skip_combinations = { }
        
        for key, value in self.PARA.items():
            if isinstance(value, dict):
                # Handle dictionary parameters - group sub-parameters by parent key
                sub_parts = []
                for sub_key, sub_value in value.items():
                    if sub_value not in exclude_values:
                        sub_parts.append(f"{sub_key}={sub_value}")
                if sub_parts:
                    name_parts.append(f"[{key}={{{','.join(sub_parts)}}}]")
            else:
                # Handle regular parameters
                if (key, value) not in skip_combinations and value not in exclude_values:
                    name_parts.append(f"[{key}={value}]")
        
        return "".join(name_parts)

# Base forward return registry - each method gets its own registered function
@Label.register("return")
def label_return(self):
    """Simple return: (p[t+h] - p[t]) / p[t]"""
    LB = self.PARA.get('LB', 1)
    lag = self.PARA.get('lag', 0)
    
    prices = self.df['close']  # Hardcoded to 'close' column
    future_prices = prices.shift(-LB - lag)
    self.label = (future_prices - prices) / prices

@Label.register("log_return")
def label_log_return(self):
    """Log return: log(p[t+h] / p[t])"""
    LB = self.PARA.get('LB', 1)
    lag = self.PARA.get('lag', 0)
    
    prices = self.df['close']  # Hardcoded to 'close' column
    future_prices = prices.shift(-LB - lag)
    self.label = np.log(future_prices / prices)

@Label.register("price_diff")
def label_price_diff(self):
    """Absolute price change: p[t+h] - p[t]"""
    LB = self.PARA.get('LB', 1)
    lag = self.PARA.get('lag', 0)
    
    prices = self.df['close']  # Hardcoded to 'close' column
    future_prices = prices.shift(-LB - lag)
    self.label = future_prices - prices

@Label.register("ma_return")
def label_ma_return(self):
    """MA-smoothed future return: using average of future prices"""
    LB = self.PARA.get('LB', 1)
    lag = self.PARA.get('lag', 0)
    ma_window = self.PARA.get('ma_window', 5)  # Method-specific parameter
    
    prices = self.df['close']  # Hardcoded to 'close' column
    # Calculate MA of future prices from t+LB to t+LB+ma_window-1
    future_prices_list = []
    for i in range(ma_window):
        future_prices_list.append(prices.shift(-LB - lag - i))
    future_ma = pd.concat(future_prices_list, axis=1).mean(axis=1)
    self.label = (future_ma - prices) / prices
