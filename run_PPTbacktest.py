"""
PPT Model Backtesting MVP
Implements basic signal generation, single window backtest, and performance comparison
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from run_PPT_torch import PPTModelTorch, PPTConfigTorch, train_ppt_torch
from run_PPT import prepare_data, compute_run_length

# ==================== Signal Generation ====================

def generate_ppt_signals(model: PPTModelTorch, df: pd.DataFrame, config: PPTConfigTorch) -> Dict[int, pd.Series]:
    """Generate trading signals from PPT model predictions"""
    features, _, _, mask = prepare_data(df, config)
    feat_tensors = {k: torch.FloatTensor(v[mask]) for k, v in features.items()}
    
    with torch.no_grad():
        outputs = model(feat_tensors)
    
    signals = {}
    for h in config.H_SET:
        mu_hat, p_hat = outputs[h]
        mu_hat, p_hat = mu_hat.numpy(), p_hat.numpy()
        
        # Decision: score_H = p_H * μ_H, trade if score > threshold and p > p_threshold
        score = p_hat * mu_hat
        # Create histogram analysis of model outputs
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].hist(score, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(config.score_threshold, color='red', linestyle='--', label=f'Threshold: {config.score_threshold}')
        axes[0].set_title(f'Score Distribution (H={h})')
        axes[0].set_xlabel('Score = p × μ')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(p_hat, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(config.p_threshold, color='red', linestyle='--', label=f'Threshold: {config.p_threshold}')
        axes[1].set_title(f'Probability Distribution (H={h})')
        axes[1].set_xlabel('Predicted Probability')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].hist(mu_hat, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[2].axvline(0, color='red', linestyle='--', label='Zero Return')
        axes[2].set_title(f'Return Distribution (H={h})')
        axes[2].set_xlabel('Predicted Return')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'signal_distribution_H{h}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create full signal array (with NaN for invalid data)
        sig_full = np.zeros(len(df))
        sig_masked = np.where((score > config.score_threshold) & (p_hat > config.p_threshold), 1, 0)
        sig_full[mask] = sig_masked
        
        # Stats on valid data only
        valid_score = score[np.isfinite(score)]
        valid_p = p_hat[np.isfinite(p_hat)]
        print(f"  H={h}: score [{valid_score.min():.4f}, {valid_score.max():.4f}], p [{valid_p.min():.4f}, {valid_p.max():.4f}], signals: {sig_masked.sum()}/{len(sig_masked)}")
        
        signals[h] = pd.Series(sig_full, index=df.index)
    
    return signals

def generate_baseline_signals(df: pd.DataFrame, runlen_threshold=4, ret_threshold=0) -> pd.Series:
    """Baseline: hard PPT rule (runlen>=4 & ret_4<0)"""
    close = df['close'].values
    ret1 = np.diff(close, prepend=close[0]) / close
    
    # Compute run length of down days using run_PPT function
    down_mask = ret1 < 0
    runlen = compute_run_length(down_mask)
    
    # Compute 4-day return
    ret4 = np.full(len(close), np.nan)
    ret4[4:] = close[4:] / close[:-4] - 1
    
    # Signal: 1 if runlen>=4 and ret4<0
    signals = np.where((runlen >= runlen_threshold) & (ret4 < ret_threshold), 1, 0)
    return pd.Series(signals, index=df.index)

# ==================== Backtesting ====================

def simulate_strategy(df: pd.DataFrame, signals: pd.Series, holding_period: int = 1, cost_bp: float = 0) -> Dict:
    """
    Simulate trading strategy with fixed holding period
    
    Args:
        df: OHLCV dataframe
        signals: Trading signals (0 or 1)
        holding_period: Days to hold position (H)
        cost_bp: Transaction cost in basis points
        
    Returns:
        dict with pnl, positions, trades, nav
    """
    close = df['close'].values
    n = len(close)
    
    positions = np.zeros(n)
    pnl = np.zeros(n)
    trades = np.zeros(n)
    
    # Simple strategy: enter on signal, hold for H days
    for i in range(n - holding_period):
        if signals.iloc[i] == 1:
            # Enter position
            entry_price = close[i]
            exit_price = close[min(i + holding_period, n - 1)]
            
            # Calculate return
            ret = (exit_price - entry_price) / entry_price
            
            # Apply transaction cost
            cost = cost_bp / 10000 * 2  # Entry + exit
            net_ret = ret - cost
            
            pnl[i + holding_period] = net_ret
            positions[i:i + holding_period] = 1
            trades[i] = 1
    
    # Calculate cumulative PnL (NAV)
    nav = np.cumsum(pnl)
    
    return {
        'pnl': pnl,
        'nav': nav,
        'positions': positions,
        'trades': trades,
        'signals': signals.values
    }

def calculate_metrics(sim_results: Dict, df: pd.DataFrame) -> Dict:
    """Calculate performance metrics"""
    pnl = sim_results['pnl']
    nav = sim_results['nav']
    trades = sim_results['trades']
    
    # Basic metrics
    total_pnl = nav[-1]
    n_trades = int(trades.sum())
    
    # Returns for Sharpe calculation
    returns = pnl[pnl != 0]
    if len(returns) > 0:
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        avg_ret_per_trade = np.mean(returns)
    else:
        sharpe = 0
        avg_ret_per_trade = 0
    
    # Max drawdown
    running_max = np.maximum.accumulate(nav)
    drawdown = nav - running_max
    max_dd = np.min(drawdown)
    
    # Win rate
    wins = returns > 0
    win_rate = np.mean(wins) if len(returns) > 0 else 0
    
    # Turnover
    n_days = len(df)
    turnover = n_trades / n_days if n_days > 0 else 0
    
    return {
        'total_pnl': total_pnl,
        'n_trades': n_trades,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'avg_return_per_trade': avg_ret_per_trade,
        'turnover': turnover,
        'final_nav': nav[-1]
    }

# ==================== Comparison & Visualization ====================

def compare_strategies(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """Compare multiple strategies and return metrics table"""
    metrics_list = []
    
    for name, result in results_dict.items():
        metrics = result['metrics']
        metrics['strategy'] = name
        metrics_list.append(metrics)
    
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics = df_metrics.set_index('strategy')
    
    return df_metrics

def plot_nav_comparison(results_dict: Dict[str, Dict], save_path: str = 'nav_comparison.png'):
    """Plot NAV curves for all strategies"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: NAV curves
    for name, result in results_dict.items():
        nav = result['simulation']['nav']
        axes[0].plot(nav, label=name, linewidth=2)
    
    axes[0].set_title('Net Asset Value (NAV) Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Days')
    axes[0].set_ylabel('Cumulative PnL')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Metrics comparison
    metrics_df = compare_strategies(results_dict)
    metrics_to_plot = ['sharpe', 'win_rate', 'turnover']
    
    x = np.arange(len(metrics_df))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        axes[1].bar(x + i * width, metrics_df[metric], width, label=metric)
    
    axes[1].set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Strategy')
    axes[1].set_ylabel('Value')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(metrics_df.index)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"NAV comparison plot saved to {save_path}")

def print_metrics_table(results_dict: Dict[str, Dict]):
    """Print formatted metrics table"""
    df_metrics = compare_strategies(results_dict)
    
    print("\n" + "="*80)
    print("PERFORMANCE METRICS COMPARISON")
    print("="*80)
    print(df_metrics.to_string())
    print("="*80)

# ==================== Main Backtest Pipeline ====================

def run_backtest_mvp(df_train: pd.DataFrame, df_test: pd.DataFrame, config: PPTConfigTorch):
    """
    MVP backtest pipeline: train on train set, test on test set
    
    Args:
        df_train: Training data
        df_test: Test data
        config: PPT configuration
        
    Returns:
        dict with all results
    """
    print("="*80)
    print("PPT MODEL BACKTEST - MVP")
    print("="*80)
    
    # 1. Train PPT model
    print(f"\n[1/4] Training PPT model on {len(df_train)} days...")
    features_train, mu_train, p_train, mask_train = prepare_data(df_train, config)
    model = PPTModelTorch(config)
    model, history = train_ppt_torch(model, features_train, mu_train, p_train, mask_train, config)
    print(f"Training completed. Final loss: {history['loss'][-1]:.6f}")
    
    # Plot parameter analysis
    from run_PPT_torch import plot_parameter_analysis
    plot_parameter_analysis(history)
    
    # 2. Generate signals
    print(f"\n[2/4] Generating signals on test set ({len(df_test)} days)...")
    ppt_signals = generate_ppt_signals(model, df_test, config)
    baseline_signals = generate_baseline_signals(df_test)
    
    # 3. Run simulations
    print("\n[3/4] Running backtests...")
    results = {}
    
    for h in config.H_SET:
        print(f"  - PPT H={h}...")
        sim = simulate_strategy(df_test, ppt_signals[h], holding_period=h, cost_bp=0.0)
        metrics = calculate_metrics(sim, df_test)
        results[f'PPT_H{h}'] = {'simulation': sim, 'metrics': metrics}
    
    print(f"  - Baseline (hard PPT)...")
    sim_baseline = simulate_strategy(df_test, baseline_signals, holding_period=4, cost_bp=0.0)
    metrics_baseline = calculate_metrics(sim_baseline, df_test)
    results['Baseline'] = {'simulation': sim_baseline, 'metrics': metrics_baseline}
    
    # 4. Compare and visualize
    print("\n[4/4] Analyzing results...")
    print_metrics_table(results)
    plot_nav_comparison(results)
    
    return {
        'model': model,
        'history': history,
        'results': results,
        'signals': {'ppt': ppt_signals, 'baseline': baseline_signals}
    }

# ==================== Entry Point ====================

if __name__ == "__main__":
    # Load data (example with SPX)
    print("Loading data...")
    df = pd.read_csv('spx.csv', parse_dates=['date'], index_col='date')
    df.columns = [c.lower() for c in df.columns]
    
    # Split: 1980-2005 train, 2005-2015 test (simplified for MVP)
    df_train = df.loc['1990':'2011']  # Using 1990 for faster MVP demo
    df_test = df.loc['2011':'2025']   # 5 years test
    
    print(f"Train: {df_train.index[0]} to {df_train.index[-1]} ({len(df_train)} days)")
    print(f"Test:  {df_test.index[0]} to {df_test.index[-1]} ({len(df_test)} days)")
    
    # Configure
    config = PPTConfigTorch(
        H_SET=(1, 1),
        learning_rate=0.01,
        epochs=2000,
        early_stop_threshold=0.00001,
        early_stop_patience=100,
        lambda_L2=1e-4,
        lambda_p=0.5,
        score_threshold=0.0,
        p_threshold=0.55
    )
    
    # Run backtest
    backtest_results = run_backtest_mvp(df_train, df_test, config)
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETED SUCCESSFULLY")
    print("="*80)

