# Features Engineering Plan

---

## 📋 Current Implementation Analysis

### **Existing Architecture**
- ✅ **Fast functions**: `fast_SMA`, `fast_std`, `fast_zscore`, `fast_rank` with Numba optimization
- ✅ **Factor system**: `Factor` class with registry pattern for factor calculations
- ✅ **Signal generation**: `SignalGenerator` class for trading signals
- ✅ **Registered factors**: MAX, RSI, ROC factors already implemented


---

## 🎯 Implementation Plan

### **Phase 1: Add Caching to Core Functions**
```python
# Modify existing functions with caching support
def fast_SMA(series: pd.Series, window: int, exact=False, use_cache=True) -> pd.Series:
    """Ultra-fast moving average with optional caching."""
    # Add cache logic to existing function

def fast_std(series: pd.Series, window: int, type_exact=False, sample=True, use_cache=True) -> pd.Series:
    """Fast standard deviation with optional caching."""
    # Add cache logic to existing function

# Add helper functions for complex calculations
def _calculate_sr(series: pd.Series, window: int) -> pd.Series:
    """Calculate Sharpe ratio."""
    
def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Calculate Average True Range."""
```

### **Phase 2: Create Feature Definition Architecture**
```python
# Feature definition architecture
FEATURE_DEFINITIONS = {
    'ratio_ma': {
        'func': lambda df, short, long: fast_SMA(df['close'], short) / fast_SMA(df['close'], long) - 1,
        'params': {
            'short': [1, 2, 5, 10, 40, 100],
            'long': [1, 2, 5, 10, 40, 100]
        },
        'conditions': lambda short, long: short < long,
        'naming': lambda short, long: f'ratio_ma{short}_ma{long}',
        'ranked': True,
        'rank_window': 1000
    },
    
    'sr_ma': {
        'func': lambda df, ma_window, sr_window: _calculate_sr(fast_SMA(df['close'], ma_window).diff(), sr_window),
        'params': {
            'ma_window': [1, 3, 10, 40, 100],
            'sr_window': [2, 5, 10, 40, 100]
        },
        'conditions': lambda ma_window, sr_window: sr_window > 1,
        'naming': lambda ma_window, sr_window: f'sr_ma{ma_window}_{sr_window}',
        'ranked': True,
        'rank_window': 1000
    }
    
    # Add all 10 feature categories from R code...
}
```

### **Phase 3: Auto-Generate Feature Registry**
```python
# Auto-generate all features from definitions
def generate_features():
    """Generate all features from definitions with parameter combinations."""
    FEATURES = {}
    
    for feature_name, config in FEATURE_DEFINITIONS.items():
        param_names = list(config['params'].keys())
        param_values = list(config['params'].values())
        
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            
            if config['conditions'](**param_dict):
                # Original feature
                feature_key = config['naming'](**param_dict)
                FEATURES[feature_key] = lambda df, params=param_dict, func=config['func']: func(df, **params)
                
                # Ranked feature
                if config.get('ranked', False):
                    rank_key = f'rank_{feature_key}'
                    rank_window = config.get('rank_window', 1000)
                    FEATURES[rank_key] = lambda df, params=param_dict, func=config['func'], window=rank_window: fast_rank(func(df, **params), window)
    
    return FEATURES

# Generate all 300+ features
FEATURES = generate_features()
```

### **Phase 4: Update make_features Function**
```python
def make_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """Create feature matrix with automatic caching."""
    global _CACHE
    _CACHE.clear()
    
    feature_data = {}
    for name in feature_list:
        if name in FEATURES:
            feature_data[name] = FEATURES[name](df)
        else:
            print(f"Warning: Unknown feature '{name}', skipping")
    
    return pd.DataFrame(feature_data, index=df.index).fillna(method='ffill').dropna()
```

---

## 📝 TODO List

### **High Priority**
- [ ] **Add caching to fast_SMA** - Add use_cache parameter to existing function
- [ ] **Add caching to fast_std** - Add use_cache parameter to existing function
- [ ] **Add helper functions** - _calculate_sr(), _calculate_atr() for complex calculations
- [ ] **Create FEATURE_DEFINITIONS architecture** - Define all 10 feature categories with parameters
- [ ] **Implement generate_features()** - Auto-generate all 300+ features from definitions
- [ ] **Update make_features()** - Add cache clearing and feature processing

### **Medium Priority**
- [ ] **Complete all 10 feature categories** - Add remaining 8 categories from R code
- [ ] **Add parameter validation** - Validate parameter combinations and conditions
- [ ] **Performance testing** - Benchmark caching vs non-caching performance
- [ ] **Feature selection utilities** - Helper functions to select features by category

### **Low Priority**
- [ ] **Add more technical indicators** - Stochastic, Williams %R, etc.
- [ ] **Add custom features** - User-defined feature functions
- [ ] **Feature selection** - Automatic feature selection methods
- [ ] **Feature importance** - Feature importance analysis

---

## 🚀 Benefits

- ✅ **Architecture-driven**: Feature definitions with parameters, conditions, and naming
- ✅ **Auto-generation**: All 300+ features generated from definitions
- ✅ **Parameterized**: Each feature category supports multiple parameter combinations
- ✅ **Conditional**: Filter parameter combinations with custom conditions
- ✅ **Caching-aware**: Automatic caching for performance optimization
- ✅ **Extensible**: Easy to add new feature categories
- ✅ **Systematic**: Consistent naming and structure across all features

---

## 📊 Migration Strategy

1. **Phase 1**: Add caching to core functions (fast_SMA, fast_std) and helper functions
2. **Phase 2**: Create FEATURE_DEFINITIONS architecture with all 10 categories
3. **Phase 3**: Implement generate_features() to auto-generate all 300+ features
4. **Phase 4**: Update make_features() with cache management
5. **Phase 5**: Test integration with cqAI_main.py and validate performance

---

## 🚨 CRITICAL CACHE SAFETY ISSUE

### **Problem Identified**
Current caching implementation has a **critical safety issue**:

```python
# DANGEROUS: Current cache key only uses window size
cache_key = f"sma_{window}"  # Only window, no data identifier

# PROBLEM: These would incorrectly share cache
fast_SMA(df['close'], 10)    # Close prices
fast_SMA(df['volume'], 10)   # Volume data  
fast_SMA(df['high'], 10)     # High prices
fast_SMA(df['low'], 10)      # Low prices
```

### **Safety Risks**
1. **Data Confusion**: Different data sources (close, volume, high, low) sharing same cache
2. **Time Window Issues**: Different time periods with same window size
3. **Preprocessing Conflicts**: Raw data vs normalized/transformed data
4. **Silent Errors**: Wrong results without obvious failure

