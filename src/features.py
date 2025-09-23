# src/features.py - Feature engineering and label creation
import pandas as pd
import numpy as np
import talib

def make_features(df, feature_list):
    """
    Create features from OHLCV data.
    
    Args:
        df (pd.DataFrame): OHLCV dataframe with datetime index
        feature_list (list): List of feature names to create
        
    Returns:
        np.ndarray: Feature matrix
    """
    features_df = pd.DataFrame(index=df.index)
    
    for feature_name in feature_list:
        if feature_name.startswith('sma_'):
            # Simple Moving Average
            period = int(feature_name.split('_')[1])
            features_df[feature_name] = df['close'].rolling(window=period).mean()
            
        elif feature_name.startswith('ema_'):
            # Exponential Moving Average
            period = int(feature_name.split('_')[1])
            features_df[feature_name] = df['close'].ewm(span=period).mean()
            
        elif feature_name.startswith('rsi_'):
            # Relative Strength Index
            period = int(feature_name.split('_')[1])
            features_df[feature_name] = talib.RSI(df['close'].values, timeperiod=period)
            
        elif feature_name.startswith('bb_'):
            # Bollinger Bands
            period = int(feature_name.split('_')[1])
            if len(feature_name.split('_')) > 2:
                band_type = feature_name.split('_')[2]  # upper, lower, width
            else:
                band_type = 'width'
                
            upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=period)
            
            if band_type == 'upper':
                features_df[feature_name] = upper
            elif band_type == 'lower':
                features_df[feature_name] = lower
            elif band_type == 'width':
                features_df[feature_name] = (upper - lower) / middle
            else:
                features_df[feature_name] = (df['close'] - lower) / (upper - lower)  # %B
                
        elif feature_name.startswith('macd'):
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
            if feature_name == 'macd':
                features_df[feature_name] = macd
            elif feature_name == 'macd_signal':
                features_df[feature_name] = macd_signal
            elif feature_name == 'macd_hist':
                features_df[feature_name] = macd_hist
                
        elif feature_name.startswith('atr_'):
            # Average True Range
            period = int(feature_name.split('_')[1])
            features_df[feature_name] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
            
        elif feature_name == 'volume_sma_ratio':
            # Volume to SMA ratio
            if 'volume' in df.columns:
                volume_sma = df['volume'].rolling(window=20).mean()
                features_df[feature_name] = df['volume'] / volume_sma
            else:
                features_df[feature_name] = 1.0
                
        elif feature_name.startswith('return_'):
            # Price returns
            period = int(feature_name.split('_')[1])
            features_df[feature_name] = df['close'].pct_change(period)
            
        elif feature_name.startswith('volatility_'):
            # Rolling volatility
            period = int(feature_name.split('_')[1])
            returns = df['close'].pct_change()
            features_df[feature_name] = returns.rolling(window=period).std()
            
        else:
            print(f"Warning: Unknown feature '{feature_name}', skipping")
            continue
    
    # Forward fill and backward fill missing values
    features_df = features_df.fillna(method='ffill').fillna(method='bfill')
    
    # Drop any remaining NaN rows
    features_df = features_df.dropna()
    
    print(f"Created {len(features_df.columns)} features with {len(features_df)} valid rows")
    
    return features_df.values

def make_labels(df, horizon):
    """
    Create labels for prediction (forward returns).
    
    Args:
        df (pd.DataFrame): OHLCV dataframe
        horizon (int): Number of periods ahead to predict
        
    Returns:
        np.ndarray: Label array (forward returns)
    """
    # Calculate forward returns
    forward_returns = df['close'].shift(-horizon) / df['close'] - 1
    
    # Remove the last 'horizon' rows (no future data available)
    labels = forward_returns[:-horizon].values
    
    print(f"Created {len(labels)} labels for {horizon}-period forward returns")
    
    return labels

def create_technical_features(df, include_volume=True):
    """
    Create a comprehensive set of technical indicators.
    
    Args:
        df (pd.DataFrame): OHLCV dataframe
        include_volume (bool): Whether to include volume-based features
        
    Returns:
        pd.DataFrame: DataFrame with technical features
    """
    features = pd.DataFrame(index=df.index)
    
    # Price-based features
    features['sma_5'] = df['close'].rolling(5).mean()
    features['sma_10'] = df['close'].rolling(10).mean()
    features['sma_20'] = df['close'].rolling(20).mean()
    features['sma_50'] = df['close'].rolling(50).mean()
    
    features['ema_12'] = df['close'].ewm(span=12).mean()
    features['ema_26'] = df['close'].ewm(span=26).mean()
    
    # RSI
    features['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].values, timeperiod=20)
    features['bb_upper'] = bb_upper
    features['bb_lower'] = bb_lower
    features['bb_width'] = (bb_upper - bb_lower) / bb_middle
    features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    
    # MACD
    macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
    features['macd'] = macd
    features['macd_signal'] = macd_signal
    features['macd_hist'] = macd_hist
    
    # ATR
    features['atr_14'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
    
    # Returns and volatility
    features['return_1'] = df['close'].pct_change(1)
    features['return_5'] = df['close'].pct_change(5)
    features['volatility_20'] = df['close'].pct_change().rolling(20).std()
    
    # Volume features
    if include_volume and 'volume' in df.columns:
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']
        
        # On-Balance Volume
        obv = talib.OBV(df['close'].values, df['volume'].values)
        features['obv'] = obv
        features['obv_sma_20'] = pd.Series(obv, index=df.index).rolling(20).mean()
    
    return features



