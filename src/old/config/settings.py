# src/config/settings.py
## base config for the main project
BASE_CONFIG = {
    "ticker": "btc",
    "data_folder": "data/btc_csv",
    "windows": [3, 7, 14],
    "hold_periods": [10, 30, 60],
    "thresholds": [5, 10, 20, 30],
    "num_of_share": 1,
    "cost_ratio": 0.001,
    "slippage_ratio": 0.0,
    "longshort": "long",
    "signal_type": "RSI",  
    "plot": False        
}
