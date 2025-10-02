DEVLOG.md

-- -- -- TEMPLATE
######################################################################################
## yyyy-mm-dd

### Goals

### Work done

### Decisions
- **D-0001 鈥?XXX**: 
- **D-0002 鈥?XXX**: 

### TODO / Next
- XX
- XX

### Random Thoughts


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
- **D-0003 鈥?Use constants for timestamp columns**: Moved hardcoded timestamp column names to `TIMESTAMP_COLUMNS` constant for better maintainability
- **D-0004 鈥?Config-driven architecture**: DataSource subclasses only need configuration, no method overrides needed
- **D-0005 鈥?UTC standardization**: All datetime columns automatically converted to UTC for consistency
- **D-0006 鈥?Performance over readability**: Chose optimized pandas operations (`sort=False, copy=False`) for better performance with large datasets
- **D-0007 鈥?Functional approach for file reading**: Used list comprehensions instead of loops for more Pythonic and R-like elegance

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
- **D-0008 鈥?Remove DataManager**: DataManager was over-engineered for current needs, direct DataSource usage is simpler
- **D-0009 鈥?Feature naming convention**: Standardized to `{feature}_{param1}_{param2}` format (e.g., `maratio_5_20`). feature name doesn't allow '_'
- **D-0010 鈥?OOP Feature architecture**: Class-based approach for better extensibility and parameter management
- **D-0011 鈥?Registry pattern**: Decorator-based registration for clean feature function organization
- **D-0012 鈥?CRITICAL: Disable caching for safety**: Cache keys (`sma_{window}`) caused data confusion between close/volume/high/low

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
## 2025-09-29
### Goals
- quickly read relevant papers - *淇濊瘉鑷繁涓嶆槸鍦ㄨ嚜濞辫嚜涔愮殑灏忓湀瀛愰噷鑷猦igh* *maximize the leverage - 鍓嶄汉宸叉湁鐨勫伐浣滃綋鐒惰鍏堢湅*
- first model - VB or EOM flow or PPT (Knowledge-driven AI)
  - **SD1 - which universe, single or multiple** - probably single
  - **SD2 - model choice**

### Work done
- [X] Skim *From Deep Learning to LLMs: A Survey of AI in Quant* 鈫?extract 2鈥? key ideas relevant to targets/labeling in new framework
      - development of alpha strategies typically includes four steps: data processing, model prediction, portfolio optimization, and order execution
      - *manual labeling* of trading signals -> the use of *deep learning* models -> ultimately to an era of agent interaction and decision-making between LLM agents
      - *long-term temporal dependence* TCN(Temporal Convolutional Networks) a Gated Recurrent Unit (GRU) 
      - "LLMs primarily serve two roles - predictors and agents" - actually might be very useful in feature architecturing
      - data - pairwise edges or hyperedges, usually represented as a graph 饾挗=V脳E,  where V is the set of nodes and E is the set of edges. static (LT trend) and dynamic (event based) relation
      - Synthetic data
      - [Symbolic factors (limited by operand and operator) vs ML factors] - encoder-decoder
        - latent representation or ouput could be used as feature, "A graph-based framework for stock trend forecasting via mining Concept"
        - temporal patterns (CNN, CNN-LSTM), spatial pattern (Implicit methods use self-attention mechanism, explicit methods use graphic structure) or Hybrid
- [ ] Read *Quant 4.0* intro & conclusion only 鈫?note how it frames pipeline reformation; map to my State 脳 Driver 脳 Quality
      - Encoder-decoder
- [ ] For labeling: review *SAFE Machine Learning in Quant Trading* 鈫?focus only on calibration/robustness sections
- [ ] Decide SD1: which universe for first MVP model (SPX only vs multi-asset futures)
- [ ] Decide SD2: first model choice (simple CNN/Transformer vs AlphaNet baseline)
- [ ] Define first target: distributional return (fwd_ret with cost-adjustment, 1 horizon only)
- [ ] Implement minimal lag function + align X and Y for MVP dataset
- [ ] Test pipeline with one signal family (VB or PPT) 鈫?verify data flow, Y alignment, evaluation metric (IC/RankIC)

### Decisions

### Random Thoughts
- AI+ in small sample world -- knowledge-driven AI, explainable AI 
  - 鏃㈠彲浠ユ槸缁欏畾閫昏緫(e.g. PPT or end-of-day flow)璁〢I瀛︿範pattern锛屼篃鍙互鏄AI鏇碼uto鐨勫涔犳綔鍦ㄧ殑閫昏緫锛岀劧鍚巋uman鍘籸eview閫昏緫
  - Symbolic Reasoning: "璐у竵瀹芥澗 鈫?閫氳儉棰勬湡涓婂崌 鈫?榛勯噾浠锋牸涓婃定" or Neural Reasoning or hybrid
  - Event Embedding
- 璺熻繃鍘荤殑鏃朵唬涓€鏍凤紝瀛︽湳鐣岀殑瀛﹁€呬滑渚濈劧鍦ㄥ鍕囩殑鎶婃渶鏂扮殑绉戠爺鐮旂┒灏濊瘯鐢ㄥ埌閲戣瀺閲岄潰鍘伙紙澶у閮介渶瑕侀挶鍟?cry锛?- 濡傛灉璇翠箣鍓嶆墽琛屽姏鍜岄€熷害鏄?010s棰嗚窇鑰呯殑edge锛岄偅涔堢幇鍦ˋI鐨勫簲鐢?(闈炰綋鍔涘伐浣滀笂锛岃€屾槸瀵箂mall sample渚濈劧寮哄ぇ鐨勪俊鎭彁鍙栬兘鍔涘拰Flexibility) - e.g. dynamic cor; LT temporal cor; 
- [ ] call MOM at 10am

### 2024-12-XX: Implemented PPT (Parametric Prediction Trading) MVP
- **Core Architecture**: Gate 脳 Driver 脳 Quality 鈫?Heads model structure
- **NumPy Implementation**: Pure NumPy/SciPy version for compatibility (27 parameters)
- **PyTorch Implementation**: Created `run_PPT_torch.py` with automatic differentiation, GPU acceleration, and 60% code reduction
- **Feature Engineering**: OHLC 鈫?returns, ATR, volatility, run-length, compression, momentum
- **Training Pipeline**: L-BFGS-B optimization with Huber + BCE loss functions
- **Multi-horizon Prediction**: Simultaneous prediction for H 鈭?{1,2,3} day horizons
- **Successful Testing**: Converged training (loss: 1.064鈫?.033) with reasonable predictions
- **Modular Design**: Extensible for monotonic splines, calibration, rolling training

######################################################################################
## 2025-09-29

### Goals
- Align PPT implementation with updated Gate × Driver × Quality spec

### Work done
- Refactored `run_PPT.py` feature pipeline to generate ret4 drawdown measures, 60-day compression ranks, and LT momentum inputs per the new PPT plan.
- Redesigned PPTModel parameters and forward pass for run-length gating, convex drawdown mixing, ReLU driver, and bounded quality scalar with configurable heads.
- Added guard standardisation and NaN-safe rolling ranks to stabilise optimisation, then verified synthetic training converges without warnings.

### TODO / Next
- [ ] Extend PPT flow with rolling training and backtesting once evaluation utilities are ready.
- [ ] Review guard and quality scaling on real datasets to tune parameter bounds.
- [ ] Consider migrating to PyTorch version for GPU acceleration when scaling to larger datasets.

### Decisions
- None
######################################################################################
## TODO

### STUDY


### CODING
- [ ] in data standard columns, have to allow additional columns (those not contained in standard mapping)
- [ ] test all customized functions
- [ ] check all features class
- [ ] check all operators class
- [ ] combine X and Y - need align function? - maybe backfill, forwardfill as well? - pandas have
- [ ] need a lag function which I am familiar with - to avoid mistakes
- [ ] for all customized TA functions, verify with some quantlib package
- [I] a real test of all features - compare results vs R or excel outputs
- [ ] furtherly increase the speed of SMA and others (like using parrallel)
- [ ] in def winsorize: rolling vs global; other winsorize methods
- [ ] add neutralize member function

- [ ] need to go back to handle practical backtest data
- [ ] discuss the plan to mimic R + Cpp backtest engine - for old fashion test

- [ ] single feature - overall analysis - references: [worldquant_alphathon] [old_Chinese_stock_codes]

- deal with 'lag': kwargs.get('lag', 0), # Time lag offset
- Add remaining features categories from R code
- Implement actual IB data format when data becomes available
- Improve SR calculation: Better handling of std_series=0 cases (currently sets to 0, should consider mean value)

### Random Thoughts
- for large n rank, use approximation method
- Consider adding factory functions only if multiple dynamic source selection is needed
- Implement safe caching solution (Feature-level caching recommended)
- Add data validation and error handling

- other winsorize methods
- rolling methods for winsorize, normalize, etc

- Implement data quality validation and reporting
- Add data caching mechanisms for faster re-loading
- Performance benchmarking and memory usage optimization

