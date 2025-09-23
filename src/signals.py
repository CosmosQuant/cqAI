# src/signals.py - Signal generation from predictions
import numpy as np
import pandas as pd

def yhat_to_signals(predictions, threshold=0.0):
    """
    Convert model predictions to trading signals.
    
    Args:
        predictions (array-like): Model predictions (expected returns)
        threshold (float): Threshold for signal generation
        
    Returns:
        np.ndarray: Trading signals (-1, 0, 1)
    """
    predictions = np.array(predictions)
    signals = np.zeros_like(predictions)
    
    # Generate signals based on threshold
    signals[predictions > threshold] = 1   # Long signal
    signals[predictions < -threshold] = -1  # Short signal
    # signals == 0 means no position (neutral)
    
    return signals

def predictions_to_positions(predictions, threshold=0.0, position_sizing='binary'):
    """
    Convert predictions to position sizes.
    
    Args:
        predictions (array-like): Model predictions
        threshold (float): Minimum threshold for taking positions
        position_sizing (str): Method for position sizing
            - 'binary': Binary positions (-1, 0, 1)
            - 'proportional': Proportional to prediction strength
            - 'quantile': Based on prediction quantiles
            
    Returns:
        np.ndarray: Position sizes
    """
    predictions = np.array(predictions)
    
    if position_sizing == 'binary':
        return yhat_to_signals(predictions, threshold)
    
    elif position_sizing == 'proportional':
        # Scale predictions to [-1, 1] range
        positions = np.clip(predictions / np.std(predictions), -3, 3) / 3
        
        # Apply threshold
        positions[np.abs(positions) < threshold] = 0
        
        return positions
    
    elif position_sizing == 'quantile':
        # Use quantile-based position sizing
        positions = np.zeros_like(predictions)
        
        # Calculate quantiles
        q_high = np.percentile(predictions, 80)
        q_low = np.percentile(predictions, 20)
        
        # Assign positions based on quantiles
        positions[predictions >= q_high] = 1
        positions[predictions <= q_low] = -1
        
        return positions
    
    else:
        raise ValueError(f"Unknown position sizing method: {position_sizing}")

def apply_signal_filters(signals, prices, min_holding_period=1, max_position_size=1.0):
    """
    Apply filters to trading signals to improve quality.
    
    Args:
        signals (array-like): Raw trading signals
        prices (array-like): Price series for context
        min_holding_period (int): Minimum periods to hold a position
        max_position_size (float): Maximum absolute position size
        
    Returns:
        np.ndarray: Filtered signals
    """
    signals = np.array(signals)
    filtered_signals = signals.copy()
    
    # Apply minimum holding period
    if min_holding_period > 1:
        filtered_signals = apply_min_holding_period(filtered_signals, min_holding_period)
    
    # Apply maximum position size constraint
    filtered_signals = np.clip(filtered_signals, -max_position_size, max_position_size)
    
    return filtered_signals

def apply_min_holding_period(signals, min_periods):
    """
    Ensure minimum holding period for positions.
    
    Args:
        signals (array-like): Trading signals
        min_periods (int): Minimum holding periods
        
    Returns:
        np.ndarray: Signals with minimum holding period applied
    """
    signals = np.array(signals)
    filtered_signals = signals.copy()
    
    i = 0
    while i < len(signals):
        if signals[i] != 0:  # Position change detected
            # Hold this position for minimum period
            current_signal = signals[i]
            end_idx = min(i + min_periods, len(signals))
            
            # Fill the minimum holding period
            for j in range(i, end_idx):
                filtered_signals[j] = current_signal
            
            i = end_idx
        else:
            i += 1
    
    return filtered_signals

def calculate_signal_metrics(signals, returns):
    """
    Calculate metrics for trading signals.
    
    Args:
        signals (array-like): Trading signals
        returns (array-like): Asset returns
        
    Returns:
        dict: Signal quality metrics
    """
    signals = np.array(signals)
    returns = np.array(returns)
    
    # Ensure same length
    min_len = min(len(signals), len(returns))
    signals = signals[:min_len]
    returns = returns[:min_len]
    
    # Strategy returns
    strategy_returns = signals * returns
    
    # Basic metrics
    total_return = np.sum(strategy_returns)
    hit_rate = np.mean(strategy_returns > 0) if len(strategy_returns) > 0 else 0
    
    # Signal statistics
    long_signals = np.sum(signals > 0)
    short_signals = np.sum(signals < 0)
    neutral_signals = np.sum(signals == 0)
    
    # Turnover (position changes)
    position_changes = np.sum(np.diff(signals) != 0)
    turnover = position_changes / len(signals) if len(signals) > 0 else 0
    
    metrics = {
        'total_return': total_return,
        'hit_rate': hit_rate,
        'long_signals': long_signals,
        'short_signals': short_signals,
        'neutral_signals': neutral_signals,
        'signal_ratio': (long_signals + short_signals) / len(signals) if len(signals) > 0 else 0,
        'turnover': turnover,
        'avg_strategy_return': np.mean(strategy_returns) if len(strategy_returns) > 0 else 0,
        'strategy_volatility': np.std(strategy_returns) if len(strategy_returns) > 0 else 0
    }
    
    return metrics

def create_ensemble_signals(signal_list, method='average', weights=None):
    """
    Combine multiple signal series into ensemble signals.
    
    Args:
        signal_list (list): List of signal arrays
        method (str): Ensemble method ('average', 'majority', 'weighted')
        weights (list, optional): Weights for weighted ensemble
        
    Returns:
        np.ndarray: Ensemble signals
    """
    if not signal_list:
        raise ValueError("Signal list cannot be empty")
    
    # Convert to numpy arrays and ensure same length
    signals_array = []
    min_length = min(len(signals) for signals in signal_list)
    
    for signals in signal_list:
        signals_array.append(np.array(signals)[:min_length])
    
    signals_matrix = np.array(signals_array)
    
    if method == 'average':
        # Simple average
        ensemble_signals = np.mean(signals_matrix, axis=0)
        
    elif method == 'majority':
        # Majority vote
        ensemble_signals = np.zeros(min_length)
        for i in range(min_length):
            votes = signals_matrix[:, i]
            long_votes = np.sum(votes > 0)
            short_votes = np.sum(votes < 0)
            
            if long_votes > short_votes:
                ensemble_signals[i] = 1
            elif short_votes > long_votes:
                ensemble_signals[i] = -1
            # else remains 0
                
    elif method == 'weighted':
        # Weighted average
        if weights is None:
            weights = np.ones(len(signal_list)) / len(signal_list)
        weights = np.array(weights)
        
        ensemble_signals = np.average(signals_matrix, axis=0, weights=weights)
        
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return ensemble_signals

def signal_correlation_analysis(signal_list, signal_names=None):
    """
    Analyze correlations between different signal series.
    
    Args:
        signal_list (list): List of signal arrays
        signal_names (list, optional): Names for each signal series
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    if signal_names is None:
        signal_names = [f"Signal_{i+1}" for i in range(len(signal_list))]
    
    # Ensure same length
    min_length = min(len(signals) for signals in signal_list)
    
    signal_df = pd.DataFrame()
    for i, signals in enumerate(signal_list):
        signal_df[signal_names[i]] = np.array(signals)[:min_length]
    
    return signal_df.corr()

def optimize_signal_threshold(predictions, returns, threshold_range=None, metric='sharpe'):
    """
    Optimize signal threshold based on historical performance.
    
    Args:
        predictions (array-like): Model predictions
        returns (array-like): Asset returns
        threshold_range (tuple, optional): Range of thresholds to test
        metric (str): Optimization metric ('sharpe', 'return', 'hit_rate')
        
    Returns:
        dict: Optimization results with best threshold
    """
    predictions = np.array(predictions)
    returns = np.array(returns)
    
    if threshold_range is None:
        threshold_range = (0.0, np.std(predictions) * 2)
    
    thresholds = np.linspace(threshold_range[0], threshold_range[1], 50)
    results = []
    
    for threshold in thresholds:
        signals = yhat_to_signals(predictions, threshold)
        signal_metrics = calculate_signal_metrics(signals, returns)
        
        # Calculate Sharpe ratio
        strategy_returns = signals * returns
        if np.std(strategy_returns) > 0:
            sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        result = {
            'threshold': threshold,
            'total_return': signal_metrics['total_return'],
            'hit_rate': signal_metrics['hit_rate'],
            'sharpe_ratio': sharpe_ratio,
            'turnover': signal_metrics['turnover']
        }
        results.append(result)
    
    # Find best threshold based on chosen metric
    results_df = pd.DataFrame(results)
    
    if metric == 'sharpe':
        best_idx = results_df['sharpe_ratio'].idxmax()
    elif metric == 'return':
        best_idx = results_df['total_return'].idxmax()
    elif metric == 'hit_rate':
        best_idx = results_df['hit_rate'].idxmax()
    else:
        raise ValueError(f"Unknown optimization metric: {metric}")
    
    best_result = results_df.iloc[best_idx].to_dict()
    
    return {
        'best_threshold': best_result['threshold'],
        'best_metrics': best_result,
        'all_results': results_df
    }



