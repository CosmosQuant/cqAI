# Label Engineering Plan

---

## 📋 Current Status Analysis

### **Existing Architecture**
- ✅ **Feature Engineering**: Complete `Feature` class with registry pattern, automatic parameter generation
- ✅ **Signal Generation**: `yhat_to_signals()` converts predictions to trading signals
- ✅ **Backtest Framework**: `simulate()` calculates PnL from signals and prices
- ❌ **Missing**: Label generation for supervised learning (forward returns)

### **Current Gap**
From `cqAI_main.py` line 34-35:
```python
# X = features.make_features(df, FEATURES)  # ✅ Features exist
# y = features.make_labels(df, HORIZON)     # ❌ Labels missing
```

The system needs a `make_labels()` function and supporting architecture to create prediction targets.

---

## 🎯 Label Engineering Architecture

### **Core Philosophy**
- **OOP Design**: Extensible `Label` class similar to `Feature` class
- **Functional Support**: Simple factory functions for common use cases  
- **Registry Pattern**: Decorator-based registration for different label types
- **Time-Aware**: Handle time lags, forward-looking windows, and alignment
- **Performance**: Optimized calculations using existing fast functions

### **Label Types to Support**

1. **Forward Returns**
   - Simple return: `(price[t+h] - price[t]) / price[t]`
   - Log return: `log(price[t+h] / price[t])`
   - Adjusted return: Handle dividends, splits

2. **Price Movements**
   - Absolute price change: `price[t+h] - price[t]`
   - Percentage change: `(price[t+h] - price[t]) / price[t] * 100`

3. **Directional Labels**
   - Binary: `1` if up, `-1` if down, `0` if neutral
   - Multi-class: Quantile-based classification
   - Threshold-based: Custom thresholds for classification

4. **Risk-Adjusted Labels**
   - Return per unit volatility
   - Sharpe-like ratios over forward periods
   - Maximum adverse excursion (MAE) / Maximum favorable excursion (MFE)

---

## 🏗️ Technical Architecture

### **1. Core Label Class**

```python
class Label:
    """
    Base class for label generation with registry pattern.
    
    Similar to Feature class but focused on forward-looking targets.
    """
    _LABEL_REGISTRY = {}
    
    def __init__(self, df: pd.DataFrame, label_type: str, **kwargs):
        self.df = df
        self.label_type = label_type
        self.label = None
        self.PARA = {
            'horizon': kwargs.get('horizon', 1),          # Forward periods
            'column': kwargs.get('column', 'close'),      # Price column
            'method': kwargs.get('method', 'return'),     # Calculation method
            'lag': kwargs.get('lag', 0),                  # Time lag offset
            'clip_pct': kwargs.get('clip_pct', None),     # Outlier clipping
        }
        for k, v in kwargs.items():
            self.PARA[k] = v
    
    @classmethod
    def register(cls, name):
        def decorator(fn):
            cls._LABEL_REGISTRY[name] = fn
            return fn
        return decorator
    
    def calculate_label(self):
        if self.label_type not in self._LABEL_REGISTRY:
            raise ValueError(f"Unsupported label type: {self.label_type}")
        self._LABEL_REGISTRY[self.label_type](self)
        if self.PARA.get('clip_pct') is not None:
            self._clip_outliers()
        return self
    
    def _clip_outliers(self):
        """Clip extreme values using percentile-based approach."""
        clip_pct = self.PARA.get('clip_pct')
        if clip_pct is not None and self.label is not None:
            lower = self.label.quantile(clip_pct)
            upper = self.label.quantile(1 - clip_pct)
            self.label = self.label.clip(lower, upper)
    
    def get_label(self):
        return self.label
    
    def get_name(self):
        return f"{self.label_type}_{self.PARA['horizon']}_{self.PARA['method']}"
```

### **2. Label Registry Functions**

```python
@Label.register("forward_return")
def label_forward_return(self):
    """Calculate forward returns with various methods."""
    horizon = self.PARA.get('horizon', 1)
    column = self.PARA.get('column', 'close')
    method = self.PARA.get('method', 'return')
    lag = self.PARA.get('lag', 0)
    
    prices = self.df[column]
    
    if method == 'return':
        # Simple return: (p[t+h] - p[t]) / p[t]
        future_prices = prices.shift(-horizon - lag)
        self.label = (future_prices - prices) / prices
        
    elif method == 'log_return':
        # Log return: log(p[t+h] / p[t])
        future_prices = prices.shift(-horizon - lag)
        self.label = np.log(future_prices / prices)
        
    elif method == 'price_diff':
        # Absolute price change: p[t+h] - p[t]
        future_prices = prices.shift(-horizon - lag)
        self.label = future_prices - prices
        
    else:
        raise ValueError(f"Unknown method: {method}")

@Label.register("directional")
def label_directional(self):
    """Create directional labels (up/down/neutral)."""
    horizon = self.PARA.get('horizon', 1)
    column = self.PARA.get('column', 'close')
    threshold = self.PARA.get('threshold', 0.001)  # 0.1% threshold
    lag = self.PARA.get('lag', 0)
    
    prices = self.df[column]
    future_prices = prices.shift(-horizon - lag)
    returns = (future_prices - prices) / prices
    
    # Create directional labels
    self.label = pd.Series(0, index=returns.index)
    self.label[returns > threshold] = 1    # Up
    self.label[returns < -threshold] = -1  # Down
    # Neutral (0) for small movements

@Label.register("quantile_class")
def label_quantile_class(self):
    """Create quantile-based classification labels."""
    horizon = self.PARA.get('horizon', 1)
    column = self.PARA.get('column', 'close')
    n_classes = self.PARA.get('n_classes', 3)
    lag = self.PARA.get('lag', 0)
    
    prices = self.df[column]
    future_prices = prices.shift(-horizon - lag)
    returns = (future_prices - prices) / prices
    
    # Create quantile-based labels
    self.label = pd.cut(returns, bins=n_classes, labels=False) - (n_classes // 2)
```

### **3. Label Definition Architecture**

```python
LABEL_DEFINITIONS = {
    'ret_1h': {
        'params': {'horizon': [1, 3, 5, 15], 'method': ['return']},
        'conditions': lambda horizon, method: horizon > 0,
        'naming': lambda horizon, method: f'ret_{horizon}h_{method}',
        'clip_pct': 0.01,  # Clip extreme 1% outliers
    },
    'dir_short': {
        'params': {'horizon': [1, 5], 'threshold': [0.001, 0.005]},
        'conditions': lambda horizon, threshold: True,
        'naming': lambda horizon, threshold: f'dir_{horizon}h_thr{int(threshold*1000)}',
        'clip_pct': None,
    },
    'class_med': {
        'params': {'horizon': [15, 60], 'n_classes': [3, 5]},
        'conditions': lambda horizon, n_classes: n_classes % 2 == 1,  # Odd classes only
        'naming': lambda horizon, n_classes: f'class_{horizon}h_{n_classes}c',
        'clip_pct': None,
    }
}
```

### **4. Factory Functions**

```python
def generate_label_objects(df: pd.DataFrame, label_definitions: dict) -> list:
    """
    Generate all label objects from definitions.
    
    Similar to generate_feature_objects() but for labels.
    """
    import itertools
    label_objects = []
    
    for label_name, config in label_definitions.items():
        param_names = list(config['params'].keys())
        param_values = list(config['params'].values())
        
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            
            if config['conditions'](**param_dict):
                label_obj = Label(
                    df=df,
                    label_type=label_name,
                    clip_pct=config.get('clip_pct', None),
                    **param_dict
                )
                label_obj.name = config['naming'](**param_dict)
                label_objects.append(label_obj)
    
    return label_objects

def make_labels(df: pd.DataFrame, horizon: int = 1, method: str = 'return', 
                column: str = 'close') -> pd.Series:
    """
    Simple factory function for common label creation.
    
    For backward compatibility and simple use cases.
    """
    label_obj = Label(df=df, label_type='forward_return', 
                     horizon=horizon, method=method, column=column)
    return label_obj.calculate_label().get_label()

def create_dataset(df: pd.DataFrame, feature_objects: list, label_objects: list, 
                   min_periods: int = 100) -> dict:
    """
    Create aligned feature-label dataset with proper time handling.
    
    Args:
        df: Raw price data
        feature_objects: List of calculated Feature objects
        label_objects: List of calculated Label objects  
        min_periods: Minimum periods for valid data
        
    Returns:
        dict: {'features': DataFrame, 'labels': DataFrame, 'valid_mask': Series}
    """
    # Calculate all features
    feature_data = {}
    for f_obj in feature_objects:
        feature_name = getattr(f_obj, 'name', f_obj.get_name())
        feature_data[feature_name] = f_obj.calculate_feature().get_feature()
    
    # Calculate all labels  
    label_data = {}
    for l_obj in label_objects:
        label_name = getattr(l_obj, 'name', l_obj.get_name())
        label_data[label_name] = l_obj.calculate_label().get_label()
    
    # Create DataFrames
    features_df = pd.DataFrame(feature_data, index=df.index)
    labels_df = pd.DataFrame(label_data, index=df.index)
    
    # Create valid data mask (no NaN in features or labels)
    valid_mask = ~(features_df.isna().any(axis=1) | labels_df.isna().any(axis=1))
    
    # Apply minimum periods requirement
    if min_periods > 0:
        valid_mask.iloc[:min_periods] = False
    
    return {
        'features': features_df,
        'labels': labels_df, 
        'valid_mask': valid_mask,
        'aligned_features': features_df[valid_mask],
        'aligned_labels': labels_df[valid_mask]
    }
```

---

## 📋 Implementation TODO List

### **Phase 1: Core Label Infrastructure (Day 1-2)**
- [ ] **Create `Label` base class** with registry pattern
- [ ] **Implement basic label types**: `forward_return`, `directional`, `quantile_class`
- [ ] **Add outlier clipping functionality** using percentile-based approach
- [ ] **Create `make_labels()` factory function** for backward compatibility

### **Phase 2: Label Definitions & Generation (Day 2-3)**
- [ ] **Implement `generate_label_objects()`** function with parameter combinations
- [ ] **Create `LABEL_DEFINITIONS`** dictionary with common label configurations
- [ ] **Add label naming conventions** following `{type}_{horizon}h_{method}` format
- [ ] **Test basic label generation** with sample data

### **Phase 3: Dataset Creation & Alignment (Day 3-4)**
- [ ] **Implement `create_dataset()`** function for feature-label alignment
- [ ] **Handle time lags and forward-looking windows** properly
- [ ] **Add data validation** (NaN handling, minimum periods)
- [ ] **Create valid data masking** for train/test splits

### **Phase 4: Advanced Label Types (Day 4-5)**
- [ ] **Add risk-adjusted labels** (Sharpe-like ratios)
- [ ] **Implement multi-horizon labels** (1h, 4h, 1d combinations)
- [ ] **Add volatility-normalized returns** for better ML targets
- [ ] **Create ensemble labels** (combining multiple horizons)

### **Phase 5: Integration & Testing (Day 5-6)**
- [ ] **Update `cqAI_main.py`** to use new label system
- [ ] **Add comprehensive unit tests** for all label types
- [ ] **Performance benchmarking** vs simple implementations
- [ ] **Documentation and examples** in docstrings

### **Phase 6: Advanced Features (Optional)**
- [ ] **Add regime-aware labels** (different labels for different market conditions)
- [ ] **Implement cross-asset labels** (relative performance vs benchmark)
- [ ] **Add option-like payoff labels** (asymmetric risk/reward)
- [ ] **Create adaptive horizon labels** (dynamic based on volatility)

---

## 🔧 Technical Considerations

### **Time Alignment**
- **Forward-looking bias prevention**: Ensure labels use only future information
- **Lag handling**: Support time lags between features and labels for realistic trading
- **Missing data**: Handle weekends, holidays, market closures gracefully

### **Performance Optimization**
- **Vectorized operations**: Use pandas/numpy operations instead of loops
- **Memory efficiency**: Avoid copying large DataFrames unnecessarily
- **Caching**: Consider caching for repeated label calculations (if safe)

### **Data Quality**
- **Outlier handling**: Percentile-based clipping for extreme values
- **Missing data**: Forward-fill, backward-fill, or interpolation strategies
- **Validation**: Ensure feature-label temporal consistency

### **Extensibility**
- **Plugin architecture**: Easy to add new label types via registration
- **Configuration-driven**: YAML/JSON configs for label definitions
- **Composable**: Combine multiple label types for ensemble targets

---

## 🎯 Success Criteria

1. **Functional**: `make_labels(df, horizon=15)` works and integrates with `cqAI_main.py`
2. **Extensible**: New label types can be added with `@Label.register()` decorator
3. **Performant**: Label generation completes in reasonable time for large datasets
4. **Robust**: Handles edge cases (NaN, insufficient data, misaligned timestamps)
5. **Testable**: Comprehensive unit tests cover all label types and edge cases

---

This architecture provides a solid foundation for label engineering while maintaining consistency with the existing feature engineering system. The OOP design allows for easy extension, while the functional interfaces provide backward compatibility and ease of use.
