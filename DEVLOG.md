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
## 2025-01-23

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
## 2025-01-24

### Goals

### Work done
- **Removed DataManager class** - Eliminated unnecessary complexity following YAGNI principle
- **Moved load_data to DataSource ABC** - Direct usage: `BinanceDataSource().load_data(folder_path)`
- **Created IBDataSource placeholder** - Ready for future IB data implementation

### Decisions
- **D-0008 — Remove DataManager**: DataManager was over-engineered for current needs, direct DataSource usage is 

### TODO
- Implement actual IB data format when data becomes available
- Add data validation and error handling
- Consider adding factory functions only if multiple dynamic source selection is needed