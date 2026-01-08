"""
Backtest Output Organization and Visualization Module

Creates organized output folders with:
- Summary CSVs
- OLS coefficient analysis
- Visualizations (coefficient charts, return distributions, scatter plots)
- HTML report summary
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server/headless
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualizations will be skipped.")


def create_output_folder(base_name="backtest_results"):
    """Create a dated output folder for this run."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    folder_name = f"{base_name}_{timestamp}"
    
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(os.path.join(folder_name, "data"), exist_ok=True)
    os.makedirs(os.path.join(folder_name, "visualizations"), exist_ok=True)
    
    return folder_name


def save_summary_csv(output_folder, summary_df):
    """Save the scenario summary to CSV."""
    if summary_df is not None and not summary_df.empty:
        path = os.path.join(output_folder, "data", "scenario_summary.csv")
        summary_df.to_csv(path, index=False)
        return path
    return None


def save_ols_results(output_folder, ols_result, prefix="ols_all_stocks"):
    """Save OLS coefficients and analysis to CSV."""
    if ols_result is None:
        return None
    
    results_saved = []
    
    # Save coefficients
    if 'coefficients' in ols_result:
        coef_df = ols_result['coefficients']
        path = os.path.join(output_folder, "data", f"{prefix}_coefficients.csv")
        coef_df.to_csv(path, index=False)
        results_saved.append(path)
    
    # Save analysis dataframe if present
    if 'analysis_df' in ols_result:
        path = os.path.join(output_folder, "data", f"{prefix}_analysis.csv")
        ols_result['analysis_df'].to_csv(path, index=False)
        results_saved.append(path)
    
    # Save summary stats
    summary = {
        'r_squared': ols_result.get('r_squared', np.nan),
        'n_samples': ols_result.get('n_samples', 0),
        'buy_year': ols_result.get('buy_year', 'N/A'),
    }
    summary_df = pd.DataFrame([summary])
    path = os.path.join(output_folder, "data", f"{prefix}_summary.csv")
    summary_df.to_csv(path, index=False)
    results_saved.append(path)
    
    return results_saved


def save_scenario_details(output_folder, scenario_details):
    """Save detailed results for each scenario."""
    saved = []
    for key, result in scenario_details.items():
        if 'analysis_df' in result:
            path = os.path.join(output_folder, "data", f"scenario_{key}_details.csv")
            result['analysis_df'].to_csv(path, index=False)
            saved.append(path)
    return saved


def plot_ols_coefficients(output_folder, ols_result, title="OLS Coefficients", 
                          filename="ols_coefficients.png", top_n=20):
    """Create a horizontal bar chart of OLS coefficients."""
    if not HAS_MATPLOTLIB or ols_result is None:
        return None
    
    if 'coefficients' not in ols_result:
        return None
    
    coef_df = ols_result['coefficients'].head(top_n).copy()
    
    if coef_df.empty:
        return None
    
    # Sort by coefficient value for better visualization
    coef_df = coef_df.sort_values('coefficient')
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(coef_df) * 0.35)))
    
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coef_df['coefficient']]
    
    bars = ax.barh(coef_df['metric'], coef_df['coefficient'], color=colors, edgecolor='white')
    
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Coefficient (Standardized)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # Add R-squared annotation
    r_sq = ols_result.get('r_squared', 0)
    n_samples = ols_result.get('n_samples', 0)
    ax.annotate(f'R² = {r_sq:.4f}\nn = {n_samples}', 
                xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    path = os.path.join(output_folder, "visualizations", filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return path


def plot_return_distribution(output_folder, ols_result, title="Return Distribution",
                             filename="return_distribution.png"):
    """Create a histogram of returns."""
    if not HAS_MATPLOTLIB or ols_result is None:
        return None
    
    if 'analysis_df' not in ols_result:
        return None
    
    df = ols_result['analysis_df']
    if 'total_return' not in df.columns:
        return None
    
    returns = df['total_return'].dropna()
    
    # Winsorize for display
    returns_display = returns.clip(-2, 5)  # -200% to +500%
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(returns_display * 100, bins=50, edgecolor='white', alpha=0.7)
    
    # Color bars based on positive/negative
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0:
            patch.set_facecolor('#e74c3c')
        else:
            patch.set_facecolor('#2ecc71')
    
    # Add vertical lines for mean and median
    mean_ret = returns.mean() * 100
    median_ret = returns.median() * 100
    
    ax.axvline(mean_ret, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_ret:.1f}%')
    ax.axvline(median_ret, color='orange', linestyle='-', linewidth=2, label=f'Median: {median_ret:.1f}%')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Total Return (%)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Add stats annotation
    pos_pct = (returns > 0).mean() * 100
    ax.annotate(f'n = {len(returns)}\n{pos_pct:.0f}% positive', 
                xy=(0.02, 0.98), xycoords='axes fraction',
                ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    path = os.path.join(output_folder, "visualizations", filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return path


def plot_top_predictors_scatter(output_folder, ols_result, n_plots=6,
                                 filename_prefix="scatter"):
    """Create scatter plots of top predictors vs returns."""
    if not HAS_MATPLOTLIB or ols_result is None:
        return None
    
    if 'analysis_df' not in ols_result or 'coefficients' not in ols_result:
        return None
    
    df = ols_result['analysis_df']
    coef_df = ols_result['coefficients']
    
    if 'total_return' not in df.columns:
        return None
    
    # Get top predictors by absolute coefficient
    top_metrics = coef_df.head(n_plots)['metric'].tolist()
    top_metrics = [m for m in top_metrics if m in df.columns]
    
    if not top_metrics:
        return None
    
    # Create subplot grid
    n_cols = min(3, len(top_metrics))
    n_rows = (len(top_metrics) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    paths = []
    for idx, metric in enumerate(top_metrics):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        # Get data
        x = df[metric].replace([np.inf, -np.inf], np.nan)
        y = df['total_return'].clip(-2, 5)  # Winsorize for display
        
        valid = ~(x.isna() | y.isna())
        x, y = x[valid], y[valid]
        
        if len(x) < 5:
            ax.set_visible(False)
            continue
        
        # Scatter plot
        colors = ['#2ecc71' if ret > 0 else '#e74c3c' for ret in y]
        ax.scatter(x, y * 100, c=colors, alpha=0.5, s=20)
        
        # Trend line
        try:
            z = np.polyfit(x, y * 100, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "b--", linewidth=2, alpha=0.8)
        except:
            pass
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel(metric, fontsize=10)
        ax.set_ylabel('Return (%)', fontsize=10)
        
        # Get coefficient
        coef_row = coef_df[coef_df['metric'] == metric]
        if not coef_row.empty:
            coef_val = coef_row['coefficient'].values[0]
            ax.set_title(f'{metric}\n(coef: {coef_val:+.3f})', fontsize=10)
    
    # Hide empty subplots
    for idx in range(len(top_metrics), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('Top Predictors vs Returns', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    path = os.path.join(output_folder, "visualizations", f"{filename_prefix}_top_predictors.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return path


def plot_scenario_comparison(output_folder, summary_df, filename="scenario_comparison.png"):
    """Create a bar chart comparing scenarios."""
    if not HAS_MATPLOTLIB or summary_df is None or summary_df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create labels
    labels = [f"Buy {int(row['buy_year'])}\n{int(row['eval_years'])}yr" 
              for _, row in summary_df.iterrows()]
    
    x = np.arange(len(labels))
    width = 0.35
    
    means = summary_df['mean_return'] * 100
    medians = summary_df['median_return'] * 100
    
    bars1 = ax.bar(x - width/2, means, width, label='Mean Return', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, medians, width, label='Median Return', color='#2ecc71', alpha=0.8)
    
    ax.set_ylabel('Return (%)', fontsize=11)
    ax.set_xlabel('Scenario', fontsize=11)
    ax.set_title('Backtest Scenario Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    plt.tight_layout()
    path = os.path.join(output_folder, "visualizations", filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return path


def generate_html_report(output_folder, results_dict):
    """Generate an HTML summary report."""
    
    html_parts = []
    
    # Header
    html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Results Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f1f1f1; }
        .positive { color: #27ae60; font-weight: bold; }
        .negative { color: #e74c3c; font-weight: bold; }
        .metric-box { background: #ecf0f1; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .stat-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }
        .stat-card h4 { margin: 0; font-size: 14px; opacity: 0.9; }
        .stat-card .value { font-size: 28px; font-weight: bold; margin: 10px 0; }
        img { max-width: 100%; height: auto; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .timestamp { color: #95a5a6; font-size: 12px; }
    </style>
</head>
<body>
<div class="container">
""")
    
    # Title and timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html_parts.append(f"""
    <h1>Backtest Results Report</h1>
    <p class="timestamp">Generated: {timestamp}</p>
""")
    
    # Summary stats cards
    if 'summary' in results_dict and results_dict['summary'] is not None and not results_dict['summary'].empty:
        summary_df = results_dict['summary']
        best_mean = summary_df['mean_return'].max() * 100
        best_median = summary_df['median_return'].max() * 100
        n_scenarios = len(summary_df)
        
        html_parts.append(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <h4>Best Mean Return</h4>
            <div class="value">{best_mean:.1f}%</div>
        </div>
        <div class="stat-card">
            <h4>Best Median Return</h4>
            <div class="value">{best_median:.1f}%</div>
        </div>
        <div class="stat-card">
            <h4>Scenarios Tested</h4>
            <div class="value">{n_scenarios}</div>
        </div>
    </div>
""")
    
    # Scenario summary table
    if 'summary' in results_dict and results_dict['summary'] is not None and not results_dict['summary'].empty:
        html_parts.append("<h2>Scenario Summary</h2>")
        summary_df = results_dict['summary'].copy()
        
        # Format percentages
        for col in ['mean_return', 'median_return', 'std_return', 'min_return', 'max_return']:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
        if 'positive_pct' in summary_df.columns:
            summary_df['positive_pct'] = summary_df['positive_pct'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "N/A")
        
        html_parts.append(summary_df.to_html(index=False, classes='summary-table'))
    
    # Scenario comparison chart
    if os.path.exists(os.path.join(output_folder, "visualizations", "scenario_comparison.png")):
        html_parts.append("""
    <h2>Scenario Comparison</h2>
    <img src="visualizations/scenario_comparison.png" alt="Scenario Comparison">
""")
    
    # OLS Results - All Stocks
    if 'ols_analysis' in results_dict and results_dict['ols_analysis'] is not None:
        ols = results_dict['ols_analysis']
        html_parts.append(f"""
    <h2>OLS Analysis: All Stocks</h2>
    <div class="metric-box">
        <p><strong>R-squared:</strong> {ols.get('r_squared', 0):.4f}</p>
        <p><strong>Samples:</strong> {ols.get('n_samples', 0):,}</p>
        <p><strong>Buy Year:</strong> {ols.get('buy_year', 'N/A')}</p>
    </div>
""")
        
        if os.path.exists(os.path.join(output_folder, "visualizations", "ols_all_stocks_coefficients.png")):
            html_parts.append('<img src="visualizations/ols_all_stocks_coefficients.png" alt="OLS Coefficients">')
        
        if os.path.exists(os.path.join(output_folder, "visualizations", "ols_all_stocks_returns.png")):
            html_parts.append('<img src="visualizations/ols_all_stocks_returns.png" alt="Return Distribution">')
    
    # OLS Results - Top 100
    if 'top100_ols' in results_dict and results_dict['top100_ols'] is not None:
        ols = results_dict['top100_ols']
        html_parts.append(f"""
    <h2>OLS Analysis: Top 100 PostRank Stocks</h2>
    <div class="metric-box">
        <p><strong>R-squared:</strong> {ols.get('r_squared', 0):.4f}</p>
        <p><strong>Samples:</strong> {ols.get('n_samples', 0):,}</p>
    </div>
""")
        
        # Coefficient table
        if 'coefficients' in ols:
            coef_df = ols['coefficients'].copy()
            coef_df['coefficient'] = coef_df['coefficient'].apply(
                lambda x: f'<span class="positive">{x:+.4f}</span>' if x > 0 
                else f'<span class="negative">{x:+.4f}</span>'
            )
            html_parts.append("<h3>PostRank Metric Coefficients</h3>")
            html_parts.append(coef_df[['metric', 'coefficient']].to_html(index=False, escape=False))
        
        if os.path.exists(os.path.join(output_folder, "visualizations", "ols_top100_coefficients.png")):
            html_parts.append('<img src="visualizations/ols_top100_coefficients.png" alt="Top 100 OLS Coefficients">')
        
        if os.path.exists(os.path.join(output_folder, "visualizations", "ols_top100_returns.png")):
            html_parts.append('<img src="visualizations/ols_top100_returns.png" alt="Top 100 Return Distribution">')
        
        if os.path.exists(os.path.join(output_folder, "visualizations", "scatter_top100_top_predictors.png")):
            html_parts.append('<img src="visualizations/scatter_top100_top_predictors.png" alt="Top Predictors Scatter">')
    
    # Footer
    html_parts.append("""
</div>
</body>
</html>
""")
    
    # Write HTML file
    html_content = "\n".join(html_parts)
    path = os.path.join(output_folder, "report.html")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return path


def save_all_outputs(results_dict, verbose=True):
    """
    Main function to save all outputs from a backtest run.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys: 'summary', 'scenarios', 'ols_analysis', 'postrank_ols', 'top100_ols'
    verbose : bool
        Print progress
        
    Returns:
    --------
    str : Path to the output folder
    """
    
    # Create output folder
    output_folder = create_output_folder()
    
    if verbose:
        print(f"\n{'='*70}")
        print("SAVING OUTPUTS")
        print('='*70)
        print(f"Output folder: {output_folder}")
    
    # Save CSVs
    if 'summary' in results_dict:
        save_summary_csv(output_folder, results_dict['summary'])
        if verbose:
            print("  - Saved scenario summary CSV")
    
    if 'scenarios' in results_dict:
        save_scenario_details(output_folder, results_dict['scenarios'])
        if verbose:
            print("  - Saved scenario detail CSVs")
    
    if 'ols_analysis' in results_dict and results_dict['ols_analysis'] is not None:
        save_ols_results(output_folder, results_dict['ols_analysis'], "ols_all_stocks")
        if verbose:
            print("  - Saved all-stocks OLS results")
    
    if 'top100_ols' in results_dict and results_dict['top100_ols'] is not None:
        save_ols_results(output_folder, results_dict['top100_ols'], "ols_top100")
        if verbose:
            print("  - Saved top-100 OLS results")
    
    # Generate visualizations
    if HAS_MATPLOTLIB:
        if verbose:
            print("\nGenerating visualizations...")
        
        # Scenario comparison
        if 'summary' in results_dict:
            path = plot_scenario_comparison(output_folder, results_dict['summary'])
            if path and verbose:
                print(f"  - {os.path.basename(path)}")
        
        # All-stocks OLS visualizations
        if 'ols_analysis' in results_dict and results_dict['ols_analysis'] is not None:
            path = plot_ols_coefficients(
                output_folder, results_dict['ols_analysis'],
                title="OLS Coefficients: All Stocks",
                filename="ols_all_stocks_coefficients.png"
            )
            if path and verbose:
                print(f"  - {os.path.basename(path)}")
            
            path = plot_return_distribution(
                output_folder, results_dict['ols_analysis'],
                title="Return Distribution: All Stocks",
                filename="ols_all_stocks_returns.png"
            )
            if path and verbose:
                print(f"  - {os.path.basename(path)}")
        
        # Top-100 OLS visualizations
        if 'top100_ols' in results_dict and results_dict['top100_ols'] is not None:
            path = plot_ols_coefficients(
                output_folder, results_dict['top100_ols'],
                title="OLS Coefficients: Top 100 PostRank Stocks",
                filename="ols_top100_coefficients.png",
                top_n=25
            )
            if path and verbose:
                print(f"  - {os.path.basename(path)}")
            
            path = plot_return_distribution(
                output_folder, results_dict['top100_ols'],
                title="Return Distribution: Top 100 PostRank Stocks",
                filename="ols_top100_returns.png"
            )
            if path and verbose:
                print(f"  - {os.path.basename(path)}")
            
            path = plot_top_predictors_scatter(
                output_folder, results_dict['top100_ols'],
                n_plots=6,
                filename_prefix="scatter_top100"
            )
            if path and verbose:
                print(f"  - {os.path.basename(path)}")
    
    # Generate HTML report
    if verbose:
        print("\nGenerating HTML report...")
    
    report_path = generate_html_report(output_folder, results_dict)
    if verbose:
        print(f"  - {os.path.basename(report_path)}")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"All outputs saved to: {output_folder}")
        print(f"Open report: {os.path.join(output_folder, 'report.html')}")
        print('='*70)
    
    return output_folder
