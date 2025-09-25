# Label Engineering Plan

---

## 📋 Current Status Analysis

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

### **Label Processing Pipeline**

**Architecture**: `Forward Returns (Base) → Risk-Adjustment (Optional) → Directional Transform (Optional)`

#### **1. Forward Returns (Always Base Layer)**
All labels start with forward return calculation using one of 4 methods:
   - **`price_diff`**: `price[t+h] - price[t]` (absolute change)
   - **`return`**: `(price[t+h] - price[t]) / price[t]` (relative change)
   - **`log_return`**: `log(price[t+h] / price[t])` (log difference)
   - **`ma_return`**: Using `ma_price from t+h to t+h+ma_window` (smoothed future price)

#### **2. Risk-Adjustment (Optional Processing Layer)**
Similar to `normalize_feature()` in Feature class - transforms raw returns:
   - **`volatility`**: Return per unit volatility (Sharpe-like)
   - **`rolling_sharpe`**: Rolling Sharpe ratio over forward periods
   - **`vol_normalized`**: Volatility-normalized returns for ML stability
   - **`mae_mfe`**: Maximum adverse/favorable excursion ratios

#### **3. Directional Transform (Optional Final Layer)**
Similar to another processing step - converts continuous values to discrete labels:
   - **`binary`**: `1` if up, `-1` if down, `0` if neutral (threshold-based)
   - **`quantile`**: Multi-class classification based on quantiles (e.g., 3-class, 5-class)
   - **`threshold`**: Custom threshold-based classification with multiple levels

---

## 🏗️ Technical Architecture

### **1. Standalone Utility Functions**

```python
def discretize(series: pd.Series, method: str, threshold: float = 0.0, n_classes: int = 3) -> pd.Series:
    """Convert continuous values to discrete labels using various methods."""
    # ✅ IMPLEMENTED: Binary, quantile methods
```

### **2. Core Label Class**

```python
class Label:
    """Base class for label generation with registry pattern."""
    # ✅ IMPLEMENTED: Registry pattern, calculate pipeline, risk adjustment, discretize, winsorize
```

### **3. Label Registry Functions (Registry Pattern)**

```python
# ✅ IMPLEMENTED: return, log_return, price_diff, ma_return registry functions
```

### **4. Label Definition Architecture**

```python
LABEL_DEFINITIONS = {
    # Basic forward returns
    'ret_basic': {
        'params': {
            'LB': [1, 5, 15]  # Updated parameter name
        },
        'conditions': lambda LB: LB > 0,
        'naming': lambda LB: f'return[LB={LB}]',  # Updated naming format
        'winsorize_pct': None,  # Updated default
    },
    
    'ret_log': {
        'params': {
            'LB': [1, 5, 15]  # Updated parameter name
        },
        'conditions': lambda LB: LB > 0,
        'naming': lambda LB: f'log_return[LB={LB}]',  # Updated naming format
        'winsorize_pct': None,  # Updated default
    },
    
    # MA-smoothed returns
    'ret_ma': {
        'params': {
            'LB': [5, 15], 
            'ma_window': [3, 5]  # Method-specific parameter
        },
        'conditions': lambda LB, ma_window: LB >= ma_window,
        'naming': lambda LB, ma_window: f'ma_return[LB={LB}][ma_window={ma_window}]',  # Updated naming format
        'winsorize_pct': None,  # Updated default
    },
    
    # Risk-adjusted returns
    'ret_risk': {
        'params': {
            'LB': [5, 15], 
            'rescale': [{'method': 'volatility', 'nscale': 10}, {'method': 'volatility', 'nscale': 20}]  # Updated parameter structure
        },
        'conditions': lambda LB, rescale: LB > 1,
        'naming': lambda LB, rescale: f'return[LB={LB}][rescale={rescale}]',  # Updated naming format
        'winsorize_pct': None,  # Updated default
    },
    
    # Directional labels (binary)
    'dir_binary': {
        'params': {
            'LB': [1, 5], 
            'discretize': [{'method': 'binary', 'threshold': 0.001}, {'method': 'binary', 'threshold': 0.005}]  # Updated parameter structure
        },
        'conditions': lambda LB, discretize: True,
        'naming': lambda LB, discretize: f'return[LB={LB}][discretize={discretize}]',  # Updated naming format
        'winsorize_pct': None,
    },
    
    # Quantile-based classification
    'class_quantile': {
        'params': {
            'LB': [5, 15], 
            'discretize': [{'method': 'quantile', 'bin': 3}, {'method': 'quantile', 'bin': 5}]  # Updated parameter structure
        },
        'conditions': lambda LB, discretize: discretize['bin'] % 2 == 1,
        'naming': lambda LB, discretize: f'return[LB={LB}][discretize={discretize}]',  # Updated naming format
        'winsorize_pct': None,
    },
    
    # Combined: Risk-adjusted + Directional
    'risk_dir': {
        'params': {
            'LB': [15], 
            'rescale': [{'method': 'volatility', 'nscale': 10}],  # Updated parameter structure
            'discretize': [{'method': 'binary', 'threshold': 0.0}]  # Updated parameter structure
        },
        'conditions': lambda LB, rescale, discretize: True,
        'naming': lambda LB, rescale, discretize: f'return[LB={LB}][rescale={rescale}][discretize={discretize}]',  # Updated naming format
        'winsorize_pct': None,
    }
}
```

### **5. Factory Functions**

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
                    label_type=param_dict.get('method', 'return'),  # Use method as label_type (registry key)
                    winsorize_pct=config.get('winsorize_pct', None),
                    **param_dict
                )
                label_obj.name = config['naming'](**param_dict)
                label_objects.append(label_obj)
    
    return label_objects

def make_labels(df: pd.DataFrame, horizon: int = 1, method: str = 'return', 
                **kwargs) -> pd.Series:
    """
    Simple factory function for common label creation.
    
    For backward compatibility and simple use cases.
    Supports all processing options: risk_adjust, discretize, etc.
    """
    label_obj = Label(df=df, label_type=method,  # Use method as registry key
                     horizon=horizon, **kwargs)
    return label_obj.calculate().get_label()

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
        label_data[label_name] = l_obj.calculate().get_label()
    
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
- [x] **Create `Label` base class** with registry pattern (✅ IMPLEMENTED & TESTED)
- [x] **Implement basic label types**: `return`, `log_return`, `price_diff`, `ma_return` (✅ IMPLEMENTED & TESTED)
- [x] **Add discretize functionality** using standalone function (✅ IMPLEMENTED & TESTED)
- [x] **Add outlier clipping functionality** using percentile-based approach (✅ IMPLEMENTED & TESTED)
- [ ] **Create `make_labels()` factory function** for backward compatibility

### **Phase 2: Label Definitions & Generation (Day 2-3)**
- [ ] **Implement `generate_label_objects()`** function with parameter combinations
- [ ] **Create `LABEL_DEFINITIONS`** dictionary with common label configurations
- [ ] **Add label naming conventions** following `{type}_{horizon}h_{method}` format
- [ ] **Test basic label generation** with sample data

### **Phase 3: Dataset Creation & Alignment (Day 3-4)**
- [ ] **Implement `create_dataset()`** function for feature-label alignment
- [I] **Handle time lags and forward-looking windows** properly
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
