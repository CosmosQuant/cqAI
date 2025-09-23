# src/analysis.py - Results analysis and reporting
import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt

def compute_cv_metrics(pnl, signals, predictions):
    """
    Compute cross-validation metrics for a single CV fold.
    
    Args:
        pnl (dict): PnL results from simulation
        signals (array-like): Trading signals
        predictions (array-like): Model predictions
        
    Returns:
        dict: CV metrics for this fold
    """
    metrics = {}
    
    # PnL metrics
    if isinstance(pnl, dict):
        metrics['final_pnl'] = pnl.get('final_pnl', 0)
        metrics['total_trades'] = pnl.get('total_trades', 0)
        metrics['total_costs'] = pnl.get('total_costs', 0)
        
        # Calculate Sharpe ratio from net PnL
        net_pnl = pnl.get('net_pnl', [])
        if len(net_pnl) > 0:
            daily_returns = np.array(net_pnl)
            if np.std(daily_returns) > 0:
                metrics['sharpe_ratio'] = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            else:
                metrics['sharpe_ratio'] = 0
            
            # Maximum drawdown
            cumulative = np.cumsum(daily_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            metrics['max_drawdown'] = np.min(drawdown) if len(drawdown) > 0 else 0
        else:
            metrics['sharpe_ratio'] = 0
            metrics['max_drawdown'] = 0
    
    # Signal metrics
    signals = np.array(signals)
    if len(signals) > 0:
        metrics['signal_count'] = len(signals)
        metrics['long_signals'] = np.sum(signals > 0)
        metrics['short_signals'] = np.sum(signals < 0)
        metrics['neutral_signals'] = np.sum(signals == 0)
        metrics['signal_utilization'] = (metrics['long_signals'] + metrics['short_signals']) / len(signals)
        
        # Signal turnover
        position_changes = np.sum(np.diff(signals) != 0) if len(signals) > 1 else 0
        metrics['turnover'] = position_changes / len(signals)
    
    # Prediction metrics
    predictions = np.array(predictions)
    if len(predictions) > 0:
        metrics['pred_mean'] = np.mean(predictions)
        metrics['pred_std'] = np.std(predictions)
        metrics['pred_min'] = np.min(predictions)
        metrics['pred_max'] = np.max(predictions)
        
        # Prediction distribution
        metrics['pred_positive_ratio'] = np.mean(predictions > 0)
    
    return metrics

def analyze_all_results(all_results):
    """
    Analyze all cross-validation results and create comprehensive summary.
    
    Args:
        all_results (list): List of CV results from all experiments
        
    Returns:
        dict: Comprehensive analysis results
    """
    if not all_results:
        return {'error': 'No results to analyze'}
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(all_results)
    
    # Extract metrics into separate DataFrame
    metrics_list = []
    for result in all_results:
        metrics = result.get('metrics', {})
        metrics['combination_id'] = result['combination_id']
        metrics['model_id'] = result['model_id']
        metrics['algorithm'] = result['algorithm']
        metrics['cv_id'] = result['cv_id']
        metrics['cv_strategy'] = result['cv_strategy']
        metrics['cv_idx'] = result['cv_idx']
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # Summary statistics by combination
    combination_summary = metrics_df.groupby('combination_id').agg({
        'final_pnl': ['mean', 'std', 'min', 'max'],
        'sharpe_ratio': ['mean', 'std', 'min', 'max'],
        'max_drawdown': ['mean', 'std', 'min', 'max'],
        'turnover': ['mean', 'std'],
        'signal_utilization': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    combination_summary.columns = ['_'.join(col).strip() for col in combination_summary.columns]
    
    # Model performance summary
    model_summary = metrics_df.groupby('model_id').agg({
        'final_pnl': ['mean', 'std'],
        'sharpe_ratio': ['mean', 'std'],
        'max_drawdown': ['mean', 'std']
    }).round(4)
    
    model_summary.columns = ['_'.join(col).strip() for col in model_summary.columns]
    
    # CV strategy summary
    cv_summary = metrics_df.groupby('cv_strategy').agg({
        'final_pnl': ['mean', 'std'],
        'sharpe_ratio': ['mean', 'std'],
        'max_drawdown': ['mean', 'std']
    }).round(4)
    
    cv_summary.columns = ['_'.join(col).strip() for col in cv_summary.columns]
    
    # Best performing combinations
    best_combinations = combination_summary.sort_values('sharpe_ratio_mean', ascending=False).head(10)
    
    # Risk-adjusted rankings
    combination_summary['risk_adjusted_return'] = (
        combination_summary['final_pnl_mean'] / 
        (combination_summary['max_drawdown_mean'].abs() + 0.01)  # Add small constant to avoid division by zero
    )
    
    risk_adjusted_rankings = combination_summary.sort_values('risk_adjusted_return', ascending=False)
    
    analysis_results = {
        'summary': {
            'total_experiments': len(all_results),
            'unique_combinations': len(combination_summary),
            'unique_models': len(model_summary),
            'unique_cv_strategies': len(cv_summary)
        },
        'combination_summary': combination_summary,
        'model_summary': model_summary,
        'cv_summary': cv_summary,
        'best_combinations': best_combinations,
        'risk_adjusted_rankings': risk_adjusted_rankings,
        'detailed_metrics': metrics_df,
        'raw_results': results_df
    }
    
    return analysis_results

def save_experiment_results(analysis_results, output_dir):
    """
    Save experiment results to files.
    
    Args:
        analysis_results (dict): Analysis results from analyze_all_results
        output_dir (str): Output directory path
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save summary statistics
    if 'combination_summary' in analysis_results:
        analysis_results['combination_summary'].to_csv(
            output_path / 'combination_summary.csv'
        )
    
    if 'model_summary' in analysis_results:
        analysis_results['model_summary'].to_csv(
            output_path / 'model_summary.csv'
        )
    
    if 'cv_summary' in analysis_results:
        analysis_results['cv_summary'].to_csv(
            output_path / 'cv_summary.csv'
        )
    
    # Save detailed metrics
    if 'detailed_metrics' in analysis_results:
        analysis_results['detailed_metrics'].to_csv(
            output_path / 'detailed_metrics.csv', index=False
        )
    
    # Save best combinations
    if 'best_combinations' in analysis_results:
        analysis_results['best_combinations'].to_csv(
            output_path / 'best_combinations.csv'
        )
    
    # Save risk-adjusted rankings
    if 'risk_adjusted_rankings' in analysis_results:
        analysis_results['risk_adjusted_rankings'].to_csv(
            output_path / 'risk_adjusted_rankings.csv'
        )
    
    # Save summary as JSON
    summary_dict = {
        'experiment_summary': analysis_results.get('summary', {}),
        'output_directory': str(output_path),
        'files_created': [
            'combination_summary.csv',
            'model_summary.csv', 
            'cv_summary.csv',
            'detailed_metrics.csv',
            'best_combinations.csv',
            'risk_adjusted_rankings.csv'
        ]
    }
    
    with open(output_path / 'experiment_summary.json', 'w') as f:
        json.dump(summary_dict, f, indent=2)
    
    # Create performance plots
    create_performance_plots(analysis_results, output_path)
    
    print(f"Results saved to: {output_path}")
    return str(output_path)

def create_performance_plots(analysis_results, output_path):
    """
    Create performance visualization plots.
    
    Args:
        analysis_results (dict): Analysis results
        output_path (Path): Output directory path
    """
    try:
        # Plot 1: Sharpe ratio comparison by model
        if 'model_summary' in analysis_results:
            plt.figure(figsize=(12, 6))
            
            model_summary = analysis_results['model_summary']
            if 'sharpe_ratio_mean' in model_summary.columns:
                plt.subplot(1, 2, 1)
                model_summary['sharpe_ratio_mean'].plot(kind='bar')
                plt.title('Average Sharpe Ratio by Model')
                plt.ylabel('Sharpe Ratio')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
            
            # Plot 2: PnL comparison by model
            if 'final_pnl_mean' in model_summary.columns:
                plt.subplot(1, 2, 2)
                model_summary['final_pnl_mean'].plot(kind='bar')
                plt.title('Average PnL by Model')
                plt.ylabel('Final PnL')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'model_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 3: Risk-Return scatter plot
        if 'combination_summary' in analysis_results:
            combination_summary = analysis_results['combination_summary']
            
            if 'final_pnl_mean' in combination_summary.columns and 'max_drawdown_mean' in combination_summary.columns:
                plt.figure(figsize=(10, 8))
                
                x = combination_summary['max_drawdown_mean'].abs()
                y = combination_summary['final_pnl_mean']
                
                plt.scatter(x, y, alpha=0.7)
                plt.xlabel('Average Max Drawdown (Absolute)')
                plt.ylabel('Average Final PnL')
                plt.title('Risk-Return Profile by Combination')
                plt.grid(True, alpha=0.3)
                
                # Add labels for best combinations
                for i, (idx, row) in enumerate(combination_summary.head(5).iterrows()):
                    plt.annotate(idx, (abs(row['max_drawdown_mean']), row['final_pnl_mean']), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                plt.savefig(output_path / 'risk_return_scatter.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Plot 4: Performance distribution
        if 'detailed_metrics' in analysis_results:
            detailed_metrics = analysis_results['detailed_metrics']
            
            if 'sharpe_ratio' in detailed_metrics.columns:
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                detailed_metrics['sharpe_ratio'].hist(bins=30, alpha=0.7)
                plt.title('Sharpe Ratio Distribution')
                plt.xlabel('Sharpe Ratio')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                
                if 'final_pnl' in detailed_metrics.columns:
                    plt.subplot(1, 3, 2)
                    detailed_metrics['final_pnl'].hist(bins=30, alpha=0.7)
                    plt.title('Final PnL Distribution')
                    plt.xlabel('Final PnL')
                    plt.ylabel('Frequency')
                    plt.grid(True, alpha=0.3)
                
                if 'max_drawdown' in detailed_metrics.columns:
                    plt.subplot(1, 3, 3)
                    detailed_metrics['max_drawdown'].hist(bins=30, alpha=0.7)
                    plt.title('Max Drawdown Distribution')
                    plt.xlabel('Max Drawdown')
                    plt.ylabel('Frequency')
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_path / 'performance_distributions.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print("Performance plots created successfully")
        
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")

def generate_report(analysis_results, output_path=None):
    """
    Generate a comprehensive text report of the analysis.
    
    Args:
        analysis_results (dict): Analysis results
        output_path (str, optional): Path to save the report
        
    Returns:
        str: Report text
    """
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("QUANTITATIVE TRADING STRATEGY ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Summary
    if 'summary' in analysis_results:
        summary = analysis_results['summary']
        report_lines.append("EXPERIMENT SUMMARY:")
        report_lines.append(f"  Total Experiments: {summary.get('total_experiments', 'N/A')}")
        report_lines.append(f"  Unique Model×CV Combinations: {summary.get('unique_combinations', 'N/A')}")
        report_lines.append(f"  Unique Models: {summary.get('unique_models', 'N/A')}")
        report_lines.append(f"  Unique CV Strategies: {summary.get('unique_cv_strategies', 'N/A')}")
        report_lines.append("")
    
    # Best combinations
    if 'best_combinations' in analysis_results:
        best_combinations = analysis_results['best_combinations']
        report_lines.append("TOP 5 COMBINATIONS (by Sharpe Ratio):")
        report_lines.append("-" * 40)
        
        for i, (combination_id, row) in enumerate(best_combinations.head(5).iterrows(), 1):
            report_lines.append(f"{i}. {combination_id}")
            if 'sharpe_ratio_mean' in row:
                report_lines.append(f"   Sharpe Ratio: {row['sharpe_ratio_mean']:.4f}")
            if 'final_pnl_mean' in row:
                report_lines.append(f"   Avg PnL: {row['final_pnl_mean']:.4f}")
            if 'max_drawdown_mean' in row:
                report_lines.append(f"   Avg Max DD: {row['max_drawdown_mean']:.4f}")
            report_lines.append("")
    
    # Model comparison
    if 'model_summary' in analysis_results:
        model_summary = analysis_results['model_summary']
        report_lines.append("MODEL PERFORMANCE COMPARISON:")
        report_lines.append("-" * 40)
        
        for model_id, row in model_summary.iterrows():
            report_lines.append(f"{model_id}:")
            if 'sharpe_ratio_mean' in row:
                report_lines.append(f"  Avg Sharpe: {row['sharpe_ratio_mean']:.4f} (±{row.get('sharpe_ratio_std', 0):.4f})")
            if 'final_pnl_mean' in row:
                report_lines.append(f"  Avg PnL: {row['final_pnl_mean']:.4f} (±{row.get('final_pnl_std', 0):.4f})")
            report_lines.append("")
    
    # CV strategy comparison
    if 'cv_summary' in analysis_results:
        cv_summary = analysis_results['cv_summary']
        report_lines.append("CROSS-VALIDATION STRATEGY COMPARISON:")
        report_lines.append("-" * 40)
        
        for cv_strategy, row in cv_summary.iterrows():
            report_lines.append(f"{cv_strategy}:")
            if 'sharpe_ratio_mean' in row:
                report_lines.append(f"  Avg Sharpe: {row['sharpe_ratio_mean']:.4f} (±{row.get('sharpe_ratio_std', 0):.4f})")
            if 'final_pnl_mean' in row:
                report_lines.append(f"  Avg PnL: {row['final_pnl_mean']:.4f} (±{row.get('final_pnl_std', 0):.4f})")
            report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    # Save to file if path provided
    if output_path:
        output_file = Path(output_path) / 'analysis_report.txt'
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {output_file}")
    
    return report_text

def compare_model_stability(analysis_results):
    """
    Analyze model stability across different CV folds.
    
    Args:
        analysis_results (dict): Analysis results
        
    Returns:
        dict: Stability analysis results
    """
    if 'detailed_metrics' not in analysis_results:
        return {'error': 'Detailed metrics not available'}
    
    detailed_metrics = analysis_results['detailed_metrics']
    
    # Calculate coefficient of variation (CV) for each combination
    stability_metrics = detailed_metrics.groupby('combination_id').agg({
        'final_pnl': ['mean', 'std'],
        'sharpe_ratio': ['mean', 'std'],
        'max_drawdown': ['mean', 'std']
    })
    
    # Calculate coefficient of variation
    stability_metrics['pnl_cv'] = stability_metrics[('final_pnl', 'std')] / stability_metrics[('final_pnl', 'mean')].abs()
    stability_metrics['sharpe_cv'] = stability_metrics[('sharpe_ratio', 'std')] / stability_metrics[('sharpe_ratio', 'mean')].abs()
    
    # Sort by stability (lower CV is better)
    most_stable = stability_metrics.sort_values('sharpe_cv').head(10)
    
    return {
        'stability_metrics': stability_metrics,
        'most_stable_combinations': most_stable
    }



