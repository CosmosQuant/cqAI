import os, subprocess, numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union


def annualized_sharpe(returns: pd.Series) -> float:
    """Calculate annualized Sharpe ratio"""
    cleaned = returns.dropna()
    if cleaned.empty:
        return 0.0
    std = cleaned.std(ddof=0)
    if std == 0:
        return 0.0
    return float((cleaned.mean() / std) * np.sqrt(252))


def retmd_ratio(returns: pd.Series) -> float:
    """Calculate Return to Max Drawdown ratio"""
    cleaned = returns.dropna()
    if cleaned.empty:
        return 0.0
    
    # Calculate cumulative returns (no compounding)
    cumulative = cleaned.cumsum()
    
    # Calculate max drawdown
    rolling_max = cumulative.expanding().max()
    drawdown = cumulative - rolling_max
    max_dd = abs(drawdown.min())
    
    if max_dd == 0:
        return float('inf') if cleaned.mean() > 0 else 0.0
    
    # Annualized return
    total_return = cumulative.iloc[-1] if not cumulative.empty else 0
    years = len(cleaned) / 252
    annualized_return = total_return / years if years > 0 else 0
    
    return float(annualized_return / max_dd) if max_dd > 0 else 0.0


def max_drawdown(nav: pd.Series) -> float:
    """Calculate maximum drawdown"""
    if nav.empty:
        return 0.0
    rolling_max = nav.cummax()
    drawdown = nav - rolling_max
    return float(drawdown.min())


def calculate_pnl_metrics(pnl_series: pd.Series) -> Dict[str, float]:
    """Calculate performance metrics from daily PnL series"""
    nav = pnl_series.cumsum()
    return {
        'Return': float(nav.iloc[-1]) if not nav.empty else 0.0,
        'Sharpe': annualized_sharpe(pnl_series),
        'RetMD': retmd_ratio(pnl_series),
        'MaxDD': max_drawdown(nav),
        'Win Rate': float((pnl_series > 0).mean()) if not pnl_series.empty else 0.0,
    }


def standard_pnl_analysis(pnl_df: pd.DataFrame, save_path: str = 'ml_result.png') -> None:
    """Generate standard PnL analysis plot with NAV curves and metrics table
    
    Args:
        pnl_df: DataFrame where each column is daily returns for a strategy
        save_path: Path to save the output image
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot 1: NAV curves
    for col in pnl_df.columns:
        nav = pnl_df[col].cumsum()
        if col == 'BnH':
            axes[0].plot(nav.index, nav.values, label=col, linewidth=3, linestyle='--', alpha=0.8, color='gray')
        else:
            axes[0].plot(nav.index, nav.values, label=col, linewidth=2)
    
    axes[0].set_title('Net Asset Value (NAV) Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Cumulative PnL')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Metrics table
    metrics_list = []
    for col in pnl_df.columns:
        metrics = calculate_pnl_metrics(pnl_df[col])
        metrics['Strategy'] = col
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df[['Strategy', 'Return', 'Sharpe', 'RetMD', 'MaxDD', 'Win Rate']]
    
    # Create table
    axes[1].axis('tight')
    axes[1].axis('off')
    
    # Format the data for table
    table_data = []
    for _, row in metrics_df.iterrows():
        table_data.append([
            row['Strategy'],
            f"{row['Return']:.2%}",
            f"{row['Sharpe']:.2f}",
            f"{row['RetMD']:.2f}",
            f"{row['MaxDD']:.2%}",
            f"{row['Win Rate']:.2%}"
        ])
    
    table = axes[1].table(cellText=table_data,
                         colLabels=['Strategy', 'Return', 'Sharpe', 'RetMD', 'Max DD', 'Win Rate'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header
    for i in range(len(metrics_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(metrics_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved to {save_path}")
    
    # Auto-open the image
    if os.path.exists(save_path):
        if os.name == 'nt':
            os.startfile(save_path)
        elif os.name == 'posix':
            subprocess.run(['open' if os.uname().sysname == 'Darwin' else 'xdg-open', save_path])
        print(f"Opening {save_path}...")

def create_holdout_split(df: pd.DataFrame, holdout_year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask = df.index.year >= holdout_year
    df_holdout = df.loc[mask].copy()
    df_model = df.loc[~mask].copy()
    if df_model.empty:
        raise ValueError("Modeling dataset is empty after applying holdout split")
    return df_model, df_holdout


def tsdata_split(df_model: pd.DataFrame, *, method: str = "fixed", train_end: str = "2015-12-31", test_size=None,
    train_window: int = 252 * 5, test_window: int = 252,step_size: int = 63, 
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], List[Tuple[pd.DataFrame, pd.DataFrame]]]:
    """
    Time series data split
    Args:
        df_model: DataFrame to split
        method: Method to split the data
        train_end: End date of the training data
        test_size: Test size
        train_window: Window size of the training data
        test_window: Window size of the test data
        step_size: Step size of the training data
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame] or List[Tuple[pd.DataFrame, pd.DataFrame]]
    """
    df_model = df_model.sort_index()
    if method == "fixed":
        if test_size is not None:
            train_end_ts = df_model.index[-int(len(df_model) * test_size)]
        else:
            train_end_ts = pd.Timestamp(train_end)
        df_train = df_model.loc[:train_end_ts]
        df_test = df_model.loc[train_end_ts + pd.Timedelta(days=1):]
        if df_train.empty or df_test.empty:
            raise ValueError("Fixed split produced empty train/test sets; adjust train_end")
        return df_train, df_test
    if method not in {"expanding", "rolling"}:
        raise ValueError(f"Unsupported split method: {method}")
    n = len(df_model)
    splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    idx_start = 0
    idx_end = train_window
    while idx_end + test_window <= n:
        if method == "expanding":
            df_train = df_model.iloc[:idx_end]
        else:
            df_train = df_model.iloc[idx_start:idx_end]
        df_test = df_model.iloc[idx_end:idx_end + test_window]
        if df_train.empty or df_test.empty:
            break
        splits.append((df_train, df_test))
        if method == "expanding":
            idx_end += step_size
        else:
            idx_start += step_size
            idx_end = idx_start + train_window
    if not splits:
        raise ValueError("Walk-forward split produced no windows; adjust window parameters")
    return splits
