import time, pandas as pd, numpy as np
from src.data import generate_test_data
from src.features import run_SMA, run_std, run_SR, fast_rank, create_features

# Set pandas display options to show more columns
pd.set_option('display.max_columns', 10); pd.set_option('display.width', 1000); pd.set_option('display.max_colwidth', 1000)

df = generate_test_data(100000, 42)

# # Test 1: Basic functionality
# sma_10 = fast_SMA(test_series, 10, use_cache=True)
# print(f"SMA(10) calculated: {len(sma_10)} points")

# # Test 2: Cache hit vs No cache
# t0 = time.time(); sma_10_cached = fast_SMA(test_series, 10, use_cache=True); cache_time = time.time() - t0
# t0 = time.time(); sma_10_no_cache = fast_SMA(test_series, 10, use_cache=False); no_cache_time = time.time() - t0
# print(f"Cached SMA(10): {cache_time:.6f}s, No cache: {no_cache_time:.6f}s")

# # Test 3: fast_std with cache
# t0 = time.time(); std_20 = fast_std(test_series, 20, use_cache=True); std_time = time.time() - t0
# print(f"STD(20) calculated: {std_time:.6f}s, Cache entries: {len(_CACHE)}")

# # Test 4: _calculate_sr function
# sr_result = _calculate_sr(df['close'], smooth_window=5, sr_window=20, threshold=100000)
# print(sr_result.tail(10))

# Test 5: New Feature Object Architecture
FEATURE_COMBO = {
    'maratio': {
        'params': {'short': [1, 3], 'long': [10, 40, 100]},
        'conditions': lambda short, long: short < long,
        'naming': lambda short, long: f'maratio_{short}_{long}',
        'winsorize': {'pct': 0.01},
        'normalize': {'method': 'rank', 'window': 1000},
    },
    'sr': {
        'params': {'ma_window': [1, 3], 'sr_window': [5, 22, 100]},
        'conditions': lambda ma_window, sr_window: sr_window > 1,
        'naming': lambda ma_window, sr_window: f'sr_{ma_window}_{sr_window}',
        'winsorize': None,
        'normalize': {'method': 'rank', 'window': 1000},
    }
}

# Generate all possible Feature objects
feature_objects = create_features(df, feature_combo=FEATURE_COMBO)
# Show feature names
print([getattr(f, 'name', f.get_name()) for f in feature_objects])  # Show first 5

feature_obj = create_features(df, feature_combo=FEATURE_COMBO)
t0 = time.time()
results = {getattr(f, 'name', f.get_name()): f.calculate().get_feature() for f in feature_obj}
print(f"Feature calculation time: {time.time() - t0:.3f}s")
# results to dataframe
res = pd.DataFrame(results)
print(res.tail(10))
# TODO - parallelize the calculation of the features to speed up the process
# TODO - add a progress bar to the calculation of the features
# TODO - add a progress bar to the calculation of the features  