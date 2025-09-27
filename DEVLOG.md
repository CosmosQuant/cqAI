DEVLOG.md

-- -- -- TEMPLATE
######################################################################################
## yyyy-mm-dd

### Goals

### Work done

### Decisions
- **D-0001 — XXX**: 
- **D-0002 — XXX**: 

### TODO / Next
- XX
- XX

### Future Work (not urgent)


######################################################################################
## 2025-09-22

### Goals
- Implement OOP architecture for data sources
- Add standardized data format with UTC time conversion
- Optimize performance for large datasets

### Work done
- **Implemented MVP data loading functions** in `src/data.py`:
  - `read_xts(path)` -  read_xts('data/btc_csv/btc_20250120.csv')
  - `combine_dataframes(dfs)` - Smart DataFrame merger with timestamp sorting and deduplication
  - `read_folder_data(folder_path, file_extension, market_keyword, exact_match)` - Batch folder reader
- **Built complete OOP architecture**:
  - `DataSource` abstract base class with standard column format `['datetime', 'open', 'high', 'low', 'close', 'volume', 'oi']`
  - `BinanceDataSource` with Binance-specific configuration
  - `DataManager` for coordinating multiple data sources
- **Added comprehensive UTC time handling**:
  - Automatic timestamp conversion (milliseconds/seconds/string formats)
  - UTC timezone standardization
  - Configurable timestamp units via `timestamp_unit` parameter
- **Implemented smart column mapping system**:
  - Automatic column filtering (only mapped columns kept)
  - Missing standard columns auto-added with value 0
  - Column reordering to standard format

### Decisions
- **D-0003 — Use constants for timestamp columns**: Moved hardcoded timestamp column names to `TIMESTAMP_COLUMNS` constant for better maintainability
- **D-0004 — Config-driven architecture**: DataSource subclasses only need configuration, no method overrides needed
- **D-0005 — UTC standardization**: All datetime columns automatically converted to UTC for consistency
- **D-0006 — Performance over readability**: Chose optimized pandas operations (`sort=False, copy=False`) for better performance with large datasets
- **D-0007 — Functional approach for file reading**: Used list comprehensions instead of loops for more Pythonic and R-like elegance

### Usages:
```python
# Basic functional usage
df = read_xts('data/btc_csv/btc_20250120.csv', show_info=False)
df = read_folder_data('data/btc_csv/', market_keyword='btc', show_info=False)

# OOP usage with DataManager
dm = DataManager()
dm.register_source('binance', BinanceDataSource(), set_default=True)
df = dm.load_data('binance', folder_path='data/btc_csv/')

# Result: Standardized format with UTC timestamps
# Columns: ['datetime', 'open', 'high', 'low', 'close', 'volume', 'oi']
# datetime: 2020-02-13 00:00:00+00:00 (UTC timezone)
```

### Future Work (not urgent)
- Implement data quality validation and reporting
- Add data caching mechanisms for faster re-loading
- Performance benchmarking and memory usage optimization

######################################################################################
## 2025-09-23 Finish the features/factors creations

### Work done
- **Removed DataManager class** - Eliminated unnecessary complexity following YAGNI principle
- **Moved load_data to DataSource ABC** - Direct usage: `BinanceDataSource().load_data(folder_path)`
- **Created IBDataSource placeholder** - Ready for future IB data implementation
- **Updated cqAI_main.py** - Replaced `data.load_ohlcv()` with new `BinanceDataSource().load_data()`
- **Implemented feature caching** - Added `use_cache=True` parameter to `fast_SMA()` and `fast_std()` functions with global `_CACHE` dictionary
- **Implemented Feature engineering architecture** - `Feature` class with registry pattern, `generate_feature_objects()` for auto parameter combinations
- **CRITICAL: Removed all caching** - Eliminated `_CACHE` and `use_cache` parameters due to data confusion safety issues


### Decisions
- **D-0008 — Remove DataManager**: DataManager was over-engineered for current needs, direct DataSource usage is simpler
- **D-0009 — Feature naming convention**: Standardized to `{feature}_{param1}_{param2}` format (e.g., `maratio_5_20`). feature name doesn't allow '_'
- **D-0010 — OOP Feature architecture**: Class-based approach for better extensibility and parameter management
- **D-0011 — Registry pattern**: Decorator-based registration for clean feature function organization
- **D-0012 — CRITICAL: Disable caching for safety**: Cache keys (`sma_{window}`) caused data confusion between close/volume/high/low

######################################################################################
## 2025-09-24 Finish the labels creations

### Work done
- [x] Implement discretize function in features.py
- [x] Implement Label class with registry pattern in labels.py and test in test_label.py
- [x] Fix discretize function to preserve NaN values in binary classification
- [x] Combine discretize method and parameters into single discretize dictionary
- [x] Change discretize default from {} to None for consistency
- [x] Rename n_classes parameter to bin in discretize function
- [x] Refactor risk adjustment parameters: rename risk_adjust to method, vol_window to nscale, combine into rescale dict
- [x] Update get_name method to group dictionary parameters: return[rescale={method=volatility,nscale=10}][discretize={method=quantile,bin=5}]
- [x] Fix quantile method to use pd.qcut for equal-frequency bins instead of pd.cut for equal-width bins
- [x] Complete Phase 1 core infrastructure: Label class, 4 label types, discretize/winsorize functions, risk adjustment
- [x] Refactor fast_SMA to run_SMA using optimized implementation from temp.md reference
- [x] Rename fast_std to run_std and fast_zscore to run_zscore for consistency
- [x] Update all function references and imports across the codebase
- [x] Implement sophisticated run_std with Numba JIT optimization and proper edge case handling
- [x] Improve run_zscore with proactive zero-volatility handling, input validation, and comprehensive documentation
- [x] Add mu parameter to run_zscore for constant mean option and refactor _calculate_sr to use run_zscore
- [x] Rename _calculate_sr to run_SR for consistency with other run_* functions
- [x] Improve winsorize function: rename pct to winsor_pct, add parameter validation (0 <= winsor_pct < 0.5), optimize performance with single quantile call


### Decisions

######################################################################################
## 2025-09-25

### Work done

### Decisions


### TODO
- [ ] test all customized functions
- [ ] check all features class
- [ ] check all operators class
- [ ] combine X and Y - need align function? - maybe backfill, forwardfill as well? - pandas have
- [ ] need a lag function which I am familiar with - to avoid mistakes
- [ ] for all customized TA functions, verify with some quantlib package
- [I] a real test of all features - compare results vs R or excel outputs
- [ ] furtherly increase the speed of SMA and others (like using parrallel)
- [ ] in def winsorize: rolling vs global; other winsorize methods

- [ ] need to go back to handle practical backtest data
- [ ] discuss the plan to mimic R + Cpp backtest engine - for old fashion test

- [ ] single feature - overall analysis - references: [worldquant_alphathon] [old_Chinese_stock_codes]

- deal with 'lag': kwargs.get('lag', 0), # Time lag offset
- Add remaining features categories from R code
- Implement actual IB data format when data becomes available
- Improve SR calculation: Better handling of std_series=0 cases (currently sets to 0, should consider mean value)

### TODO - optional
- for large n rank, use approximation method
- Consider adding factory functions only if multiple dynamic source selection is needed
- Implement safe caching solution (Feature-level caching recommended)
- Add data validation and error handling

- other winsorize methods
- rolling methods for winsorize, normalize, etc

