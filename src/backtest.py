# src/backtest.py - Backtesting and simulation engine
import numpy as np
import pandas as pd

def simulate(df, signals, cost_bp=0.0, slip_bp=0.0):
    """
    Simulate trading strategy with transaction costs and slippage.
    
    Args:
        df (pd.DataFrame): OHLCV dataframe with datetime index
        signals (array-like): Trading signals (-1, 0, 1)
        cost_bp (float): Transaction cost in basis points
        slip_bp (float): Slippage in basis points
        
    Returns:
        dict: Simulation results with PnL and metrics
    """
    signals = np.array(signals)
    
    # Ensure signals and df have same length
    min_len = min(len(df), len(signals))
    df_sim = df.iloc[:min_len].copy()
    signals = signals[:min_len]
    
    # Initialize tracking arrays
    positions = np.zeros(len(df_sim))
    trades = np.zeros(len(df_sim))
    gross_pnl = np.zeros(len(df_sim))
    transaction_costs = np.zeros(len(df_sim))
    net_pnl = np.zeros(len(df_sim))
    cumulative_pnl = np.zeros(len(df_sim))
    
    # Get price series
    prices = df_sim['close'].values
    
    # Track previous position
    prev_position = 0
    
    for i in range(len(df_sim)):
        # Current signal becomes current position
        current_position = signals[i]
        positions[i] = current_position
        
        # Calculate trade size (position change)
        trade_size = current_position - prev_position
        trades[i] = trade_size
        
        # Calculate gross PnL from position
        if i > 0:
            price_return = (prices[i] - prices[i-1]) / prices[i-1]
            gross_pnl[i] = prev_position * price_return
        
        # Calculate transaction costs
        if trade_size != 0:
            # Cost is proportional to trade size
            cost_rate = (cost_bp + slip_bp) / 10000  # Convert basis points to decimal
            transaction_costs[i] = abs(trade_size) * cost_rate
        
        # Net PnL = Gross PnL - Transaction Costs
        net_pnl[i] = gross_pnl[i] - transaction_costs[i]
        
        # Cumulative PnL
        if i == 0:
            cumulative_pnl[i] = net_pnl[i]
        else:
            cumulative_pnl[i] = cumulative_pnl[i-1] + net_pnl[i]
        
        prev_position = current_position
    
    # Create results dictionary
    results = {
        'positions': positions,
        'trades': trades,
        'gross_pnl': gross_pnl,
        'transaction_costs': transaction_costs,
        'net_pnl': net_pnl,
        'cumulative_pnl': cumulative_pnl,
        'final_pnl': cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0,
        'total_trades': np.sum(np.abs(trades)),
        'total_costs': np.sum(transaction_costs)
    }
    
    return results

def calculate_performance_metrics(pnl_series, positions=None, benchmark_returns=None):
    """
    Calculate comprehensive performance metrics.
    
    Args:
        pnl_series (array-like): PnL time series
        positions (array-like, optional): Position time series
        benchmark_returns (array-like, optional): Benchmark returns for comparison
        
    Returns:
        dict: Performance metrics
    """
    pnl_series = np.array(pnl_series)
    
    if len(pnl_series) == 0:
        return {'error': 'Empty PnL series'}
    
    # Basic metrics
    total_return = pnl_series[-1] if len(pnl_series) > 0 else 0
    daily_returns = np.diff(pnl_series) if len(pnl_series) > 1 else np.array([0])
    
    # Return statistics
    mean_return = np.mean(daily_returns)
    std_return = np.std(daily_returns)
    
    # Sharpe ratio (annualized, assuming daily data)
    if std_return > 0:
        sharpe_ratio = mean_return / std_return * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # Maximum drawdown
    cumulative = np.cumsum(daily_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
    
    # Win rate
    win_rate = np.mean(daily_returns > 0) if len(daily_returns) > 0 else 0
    
    # Calmar ratio (return/max_drawdown)
    if abs(max_drawdown) > 0:
        calmar_ratio = total_return / abs(max_drawdown)
    else:
        calmar_ratio = 0
    
    # Sortino ratio (downside deviation)
    negative_returns = daily_returns[daily_returns < 0]
    if len(negative_returns) > 0:
        downside_std = np.std(negative_returns)
        sortino_ratio = mean_return / downside_std * np.sqrt(252) if downside_std > 0 else 0
    else:
        sortino_ratio = sharpe_ratio
    
    metrics = {
        'total_return': total_return,
        'mean_daily_return': mean_return,
        'volatility': std_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'num_periods': len(pnl_series)
    }
    
    # Position-based metrics
    if positions is not None:
        positions = np.array(positions)
        long_periods = np.sum(positions > 0)
        short_periods = np.sum(positions < 0)
        neutral_periods = np.sum(positions == 0)
        
        metrics.update({
            'long_periods': long_periods,
            'short_periods': short_periods,
            'neutral_periods': neutral_periods,
            'position_utilization': (long_periods + short_periods) / len(positions)
        })
    
    # Benchmark comparison
    if benchmark_returns is not None:
        benchmark_returns = np.array(benchmark_returns)
        min_len = min(len(daily_returns), len(benchmark_returns))
        
        if min_len > 0:
            strategy_rets = daily_returns[:min_len]
            bench_rets = benchmark_returns[:min_len]
            
            # Information ratio
            active_returns = strategy_rets - bench_rets
            tracking_error = np.std(active_returns)
            
            if tracking_error > 0:
                information_ratio = np.mean(active_returns) / tracking_error * np.sqrt(252)
            else:
                information_ratio = 0
            
            # Beta
            if np.var(bench_rets) > 0:
                beta = np.cov(strategy_rets, bench_rets)[0, 1] / np.var(bench_rets)
            else:
                beta = 0
            
            metrics.update({
                'information_ratio': information_ratio,
                'tracking_error': tracking_error,
                'beta': beta
            })
    
    return metrics

def run_backtest(df, strategy_func, **strategy_params):
    """
    Run a complete backtest for a trading strategy.
    
    Args:
        df (pd.DataFrame): OHLCV dataframe
        strategy_func (callable): Function that generates signals
        **strategy_params: Parameters for the strategy function
        
    Returns:
        dict: Complete backtest results
    """
    # Generate signals using strategy function
    signals = strategy_func(df, **strategy_params)
    
    # Run simulation
    sim_results = simulate(df, signals)
    
    # Calculate performance metrics
    performance = calculate_performance_metrics(
        sim_results['cumulative_pnl'],
        sim_results['positions']
    )
    
    # Combine results
    backtest_results = {
        'simulation': sim_results,
        'performance': performance,
        'signals': signals,
        'strategy_params': strategy_params
    }
    
    return backtest_results

def compare_strategies(df, strategy_configs):
    """
    Compare multiple trading strategies.
    
    Args:
        df (pd.DataFrame): OHLCV dataframe
        strategy_configs (list): List of strategy configurations
            Each config should have 'name', 'func', and 'params'
        
    Returns:
        dict: Comparison results for all strategies
    """
    results = {}
    
    for config in strategy_configs:
        name = config['name']
        strategy_func = config['func']
        params = config.get('params', {})
        
        try:
            backtest_result = run_backtest(df, strategy_func, **params)
            results[name] = backtest_result
        except Exception as e:
            print(f"Error running strategy '{name}': {e}")
            results[name] = {'error': str(e)}
    
    # Create comparison summary
    comparison_df = pd.DataFrame()
    
    for name, result in results.items():
        if 'error' not in result:
            performance = result['performance']
            comparison_df[name] = pd.Series(performance)
    
    return {
        'individual_results': results,
        'comparison_table': comparison_df
    }

def portfolio_simulation(df, signal_dict, weights=None):
    """
    Simulate a portfolio of multiple strategies.
    
    Args:
        df (pd.DataFrame): OHLCV dataframe
        signal_dict (dict): Dictionary of {strategy_name: signals}
        weights (dict, optional): Dictionary of {strategy_name: weight}
        
    Returns:
        dict: Portfolio simulation results
    """
    if weights is None:
        # Equal weights
        weights = {name: 1.0/len(signal_dict) for name in signal_dict.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {name: w/total_weight for name, w in weights.items()}
    
    # Calculate portfolio signals
    portfolio_signals = np.zeros(len(df))
    
    for name, signals in signal_dict.items():
        weight = weights.get(name, 0)
        signals_array = np.array(signals)
        
        # Ensure same length
        min_len = min(len(portfolio_signals), len(signals_array))
        portfolio_signals[:min_len] += weight * signals_array[:min_len]
    
    # Run portfolio simulation
    portfolio_results = simulate(df, portfolio_signals)
    
    # Calculate individual strategy results for comparison
    individual_results = {}
    for name, signals in signal_dict.items():
        individual_results[name] = simulate(df, signals)
    
    return {
        'portfolio': portfolio_results,
        'individual_strategies': individual_results,
        'weights': weights,
        'portfolio_signals': portfolio_signals
    }

def walk_forward_analysis(df, strategy_func, train_period=252, test_period=63, **strategy_params):
    """
    Perform walk-forward analysis of a trading strategy.
    
    Args:
        df (pd.DataFrame): OHLCV dataframe
        strategy_func (callable): Strategy function
        train_period (int): Training period length
        test_period (int): Test period length
        **strategy_params: Strategy parameters
        
    Returns:
        dict: Walk-forward analysis results
    """
    results = []
    
    start_idx = train_period
    while start_idx + test_period <= len(df):
        # Define train and test periods
        train_start = start_idx - train_period
        train_end = start_idx
        test_start = start_idx
        test_end = start_idx + test_period
        
        # Split data
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]
        
        try:
            # Generate signals on test period (using train period for any optimization)
            test_signals = strategy_func(test_df, **strategy_params)
            
            # Run simulation on test period
            test_results = simulate(test_df, test_signals)
            
            # Calculate performance
            performance = calculate_performance_metrics(
                test_results['cumulative_pnl'],
                test_results['positions']
            )
            
            # Store results
            period_result = {
                'train_period': (train_df.index[0], train_df.index[-1]),
                'test_period': (test_df.index[0], test_df.index[-1]),
                'simulation': test_results,
                'performance': performance
            }
            
            results.append(period_result)
            
        except Exception as e:
            print(f"Error in walk-forward period {len(results)+1}: {e}")
        
        # Move to next period
        start_idx += test_period
    
    # Aggregate results
    if results:
        total_pnl = sum(r['simulation']['final_pnl'] for r in results)
        avg_sharpe = np.mean([r['performance']['sharpe_ratio'] for r in results])
        
        aggregate_metrics = {
            'total_pnl': total_pnl,
            'avg_sharpe_ratio': avg_sharpe,
            'num_periods': len(results),
            'consistency': np.std([r['simulation']['final_pnl'] for r in results])
        }
    else:
        aggregate_metrics = {}
    
    return {
        'period_results': results,
        'aggregate_metrics': aggregate_metrics
    }



