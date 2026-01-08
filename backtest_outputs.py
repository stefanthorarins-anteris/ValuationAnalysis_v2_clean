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
    """Create a dated output folder inside the main 'output' directory."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    
    # Create main output folder if it doesn't exist
    main_output = "output"
    os.makedirs(main_output, exist_ok=True)
    
    folder_name = os.path.join(main_output, f"{base_name}_{timestamp}")
    
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
    """Create a histogram of returns with proper outlier handling."""
    if not HAS_MATPLOTLIB or ols_result is None:
        return None
    
    if 'analysis_df' not in ols_result:
        return None
    
    df = ols_result['analysis_df']
    if 'total_return' not in df.columns:
        return None
    
    returns = df['total_return'].dropna()
    
    # Winsorize for display (cap at -100% to +300% for readable histogram)
    returns_display = returns.clip(-1, 3) * 100  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram with proper bins
    n, bins, patches = ax.hist(returns_display, bins=40, edgecolor='white', alpha=0.7)
    
    # Color bars based on positive/negative
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0:
            patch.set_facecolor('#e74c3c')
        else:
            patch.set_facecolor('#2ecc71')
    
    # Add vertical lines for median (use median, not mean due to outliers)
    median_ret = returns.median() * 100
    
    ax.axvline(median_ret, color='orange', linestyle='-', linewidth=2, label=f'Median: {median_ret:.1f}%')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Total Return (%) [capped at -100% to +300%]', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Add stats annotation
    pos_pct = (returns > 0).mean() * 100
    n_outliers = ((returns < -1) | (returns > 3)).sum()
    ax.annotate(f'n = {len(returns)}\n{pos_pct:.0f}% positive\n{n_outliers} outliers clipped', 
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


def compute_cohort_analysis(analysis_df, score_col='score', return_col='total_return'):
    """
    Analyze returns by ranking cohorts to validate the ranking algorithm.
    
    Cohorts: Bottom 50%, 50-25%, 25-10%, 10-5%, 5-1%, Top 1%
    
    Returns:
    --------
    dict with:
        - cohort_stats: DataFrame with mean/median returns per cohort
        - ratio_matrix: Upper triangular matrix of return ratios
        - marginal_improvement: Step-by-step improvement between adjacent cohorts
    """
    if analysis_df is None or analysis_df.empty:
        return None
    
    if return_col not in analysis_df.columns:
        return None
    
    df = analysis_df.copy()
    
    # Ensure we have valid returns
    df = df[df[return_col].notna() & np.isfinite(df[return_col])]
    
    if len(df) < 100:
        return None
    
    # Rank stocks (1 = best)
    if score_col in df.columns:
        df['_rank'] = df[score_col].rank(ascending=False)
    else:
        # If no score column, assume already sorted
        df['_rank'] = range(1, len(df) + 1)
    
    n = len(df)
    
    # Define cohorts by percentile cutoffs
    cohorts = {
        'Bottom 50%': (0.50, 1.00),      # Ranks 50%-100% (worst half)
        'Top 50-25%': (0.25, 0.50),      # Ranks 25%-50%
        'Top 25-10%': (0.10, 0.25),      # Ranks 10%-25%
        'Top 10-5%': (0.05, 0.10),       # Ranks 5%-10%
        'Top 5-1%': (0.01, 0.05),        # Ranks 1%-5%
        'Top 1%': (0.00, 0.01),          # Top 1%
    }
    
    # Calculate stats for each cohort
    cohort_stats = []
    for name, (lower_pct, upper_pct) in cohorts.items():
        lower_rank = int(n * lower_pct) + 1
        upper_rank = int(n * upper_pct)
        
        if lower_rank > upper_rank:
            lower_rank, upper_rank = upper_rank, lower_rank
        
        # Handle edge case for top 1%
        if lower_pct == 0.00:
            lower_rank = 1
        
        cohort_df = df[(df['_rank'] >= lower_rank) & (df['_rank'] <= upper_rank)]
        
        if len(cohort_df) == 0:
            continue
        
        returns = cohort_df[return_col]
        
        cohort_stats.append({
            'cohort': name,
            'n_stocks': len(cohort_df),
            'rank_range': f"{lower_rank}-{upper_rank}",
            'mean_return': returns.mean(),
            'median_return': returns.median(),
            'std_return': returns.std(),
            'positive_pct': (returns > 0).mean() * 100,
            'min_return': returns.min(),
            'max_return': returns.max(),
        })
    
    if not cohort_stats:
        return None
    
    stats_df = pd.DataFrame(cohort_stats)
    
    # Create ratio matrix (row / column)
    # Upper triangular: how much better is each cohort vs lower cohorts
    cohort_names = stats_df['cohort'].tolist()
    n_cohorts = len(cohort_names)
    
    # Use median returns for ratios (more robust)
    medians = stats_df['median_return'].values
    
    ratio_matrix = np.zeros((n_cohorts, n_cohorts))
    for i in range(n_cohorts):
        for j in range(n_cohorts):
            if medians[j] != 0 and not np.isnan(medians[j]):
                ratio_matrix[i, j] = medians[i] / medians[j]
            else:
                ratio_matrix[i, j] = np.nan
    
    ratio_df = pd.DataFrame(ratio_matrix, index=cohort_names, columns=cohort_names)
    
    # Calculate marginal improvement (each tier vs the one below)
    marginal = []
    for i in range(1, len(stats_df)):
        prev_median = stats_df.iloc[i-1]['median_return']
        curr_median = stats_df.iloc[i]['median_return']
        
        if prev_median != 0 and not np.isnan(prev_median):
            improvement = (curr_median - prev_median) / abs(prev_median) * 100
        else:
            improvement = np.nan
        
        marginal.append({
            'from_cohort': stats_df.iloc[i-1]['cohort'],
            'to_cohort': stats_df.iloc[i]['cohort'],
            'median_improvement_pct': improvement,
            'absolute_improvement': curr_median - prev_median,
        })
    
    marginal_df = pd.DataFrame(marginal)
    
    # Summary statistics
    # How much better is Top 1% vs Bottom 50%?
    top1_median = stats_df[stats_df['cohort'] == 'Top 1%']['median_return'].values
    bottom50_median = stats_df[stats_df['cohort'] == 'Bottom 50%']['median_return'].values
    
    summary = {
        'top1_vs_bottom50_ratio': top1_median[0] / bottom50_median[0] if len(top1_median) > 0 and len(bottom50_median) > 0 and bottom50_median[0] != 0 else np.nan,
        'monotonic': all(marginal_df['absolute_improvement'] > 0) if len(marginal_df) > 0 else False,
        'avg_marginal_improvement': marginal_df['median_improvement_pct'].mean() if len(marginal_df) > 0 else np.nan,
    }
    
    return {
        'cohort_stats': stats_df,
        'ratio_matrix': ratio_df,
        'marginal_improvement': marginal_df,
        'summary': summary,
    }


def save_cohort_analysis(output_folder, cohort_results, prefix="cohort"):
    """Save cohort analysis results to CSV files."""
    if cohort_results is None:
        return []
    
    saved = []
    
    # Save cohort stats
    if 'cohort_stats' in cohort_results:
        path = os.path.join(output_folder, "data", f"{prefix}_stats.csv")
        cohort_results['cohort_stats'].to_csv(path, index=False)
        saved.append(path)
    
    # Save ratio matrix
    if 'ratio_matrix' in cohort_results:
        path = os.path.join(output_folder, "data", f"{prefix}_ratio_matrix.csv")
        cohort_results['ratio_matrix'].to_csv(path)
        saved.append(path)
    
    # Save marginal improvement
    if 'marginal_improvement' in cohort_results:
        path = os.path.join(output_folder, "data", f"{prefix}_marginal.csv")
        cohort_results['marginal_improvement'].to_csv(path, index=False)
        saved.append(path)
    
    return saved


def plot_cohort_analysis(output_folder, cohort_results, filename="cohort_analysis.png"):
    """Create visualization of cohort analysis."""
    if not HAS_MATPLOTLIB or cohort_results is None:
        return None
    
    if 'cohort_stats' not in cohort_results:
        return None
    
    stats_df = cohort_results['cohort_stats']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Mean and Median returns by cohort
    ax1 = axes[0]
    x = range(len(stats_df))
    width = 0.35
    
    means = stats_df['mean_return'] * 100
    medians = stats_df['median_return'] * 100
    
    bars1 = ax1.bar([i - width/2 for i in x], means, width, label='Mean', color='#3498db', alpha=0.8)
    bars2 = ax1.bar([i + width/2 for i in x], medians, width, label='Median', color='#2ecc71', alpha=0.8)
    
    ax1.set_ylabel('Return (%)')
    ax1.set_xlabel('Cohort')
    ax1.set_title('Returns by Ranking Cohort', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_df['cohort'], rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot 2: Marginal improvement
    ax2 = axes[1]
    if 'marginal_improvement' in cohort_results and len(cohort_results['marginal_improvement']) > 0:
        marginal_df = cohort_results['marginal_improvement']
        labels = [f"{row['from_cohort']}\n→\n{row['to_cohort']}" for _, row in marginal_df.iterrows()]
        improvements = marginal_df['median_improvement_pct']
        
        colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
        ax2.bar(range(len(labels)), improvements, color=colors, alpha=0.8)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, fontsize=8)
        ax2.set_ylabel('Median Improvement (%)')
        ax2.set_title('Marginal Improvement Between Cohorts', fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot 3: Ratio to Bottom 50% (how much better is each tier)
    ax3 = axes[2]
    if 'ratio_matrix' in cohort_results:
        ratio_df = cohort_results['ratio_matrix']
        if 'Bottom 50%' in ratio_df.columns:
            ratios = ratio_df['Bottom 50%'].values[1:]  # Skip bottom 50% itself
            cohort_labels = ratio_df.index.tolist()[1:]
            
            colors = ['#2ecc71' if r > 1 else '#e74c3c' for r in ratios]
            ax3.bar(range(len(cohort_labels)), ratios, color=colors, alpha=0.8)
            ax3.set_xticks(range(len(cohort_labels)))
            ax3.set_xticklabels(cohort_labels, rotation=45, ha='right', fontsize=9)
            ax3.set_ylabel('Ratio (vs Bottom 50%)')
            ax3.set_title('Performance vs Bottom 50%', fontweight='bold')
            ax3.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Equal')
            
            # Add value labels
            for i, (ratio, label) in enumerate(zip(ratios, cohort_labels)):
                if not np.isnan(ratio):
                    ax3.annotate(f'{ratio:.2f}x', xy=(i, ratio), xytext=(0, 5),
                                textcoords='offset points', ha='center', fontsize=9)
    
    plt.tight_layout()
    path = os.path.join(output_folder, "visualizations", filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return path


def analyze_winners_vs_losers(analysis_df, return_col='total_return', min_samples_for_ols=30):
    """
    Compare metrics between winning and losing stocks.
    
    Winners: stocks with positive returns
    Losers: stocks with negative returns
    
    If enough losers, runs OLS on losers to see what predicts losses.
    Otherwise, shows comparison of median metrics.
    
    Returns:
    --------
    dict with comparison stats and optional loser OLS
    """
    if analysis_df is None or analysis_df.empty:
        return None
    
    if return_col not in analysis_df.columns:
        return None
    
    df = analysis_df.copy()
    df = df[df[return_col].notna() & np.isfinite(df[return_col])]
    
    if len(df) < 20:
        return None
    
    # Split into winners and losers
    winners = df[df[return_col] > 0]
    losers = df[df[return_col] <= 0]
    
    n_winners = len(winners)
    n_losers = len(losers)
    
    if n_losers < 5:
        return None
    
    # Identify metric columns
    exclude_cols = ['source', 'date', 'total_return', 'price_return', 'div_return', 
                    'symbol', 'score', '_rank', 'Unnamed']
    metric_cols = [c for c in df.columns 
                   if c not in exclude_cols 
                   and df[c].dtype in ['float64', 'int64', 'float32', 'int32']
                   and not any(excl in c for excl in exclude_cols)]
    
    # Filter to metrics with enough non-NaN values
    valid_metrics = []
    for col in metric_cols:
        col_data = df[col].replace([np.inf, -np.inf], np.nan)
        if col_data.notna().mean() >= 0.3:  # At least 30% non-NaN
            valid_metrics.append(col)
            df[col] = col_data
    
    if not valid_metrics:
        return None
    
    # Compare metrics between winners and losers
    comparison = []
    for metric in valid_metrics:
        w_values = winners[metric].dropna()
        l_values = losers[metric].dropna()
        
        if len(w_values) < 3 or len(l_values) < 3:
            continue
        
        w_median = w_values.median()
        l_median = l_values.median()
        w_mean = w_values.mean()
        l_mean = l_values.mean()
        
        # Calculate difference/ratio carefully
        # For normalized metrics (roughly mean 0), use difference
        # For positive metrics, use ratio
        
        is_normalized = abs(df[metric].mean()) < 0.5 and df[metric].std() > 0.5
        
        if is_normalized:
            # Use difference for normalized metrics
            diff = w_median - l_median
            comparison_value = diff
            comparison_type = 'difference'
        else:
            # Use ratio for other metrics, handle zero/negative carefully
            if l_median != 0 and not np.isnan(l_median):
                if l_median > 0 and w_median > 0:
                    ratio = w_median / l_median
                elif l_median < 0 and w_median < 0:
                    ratio = l_median / w_median  # Flip for negative values (less negative = better)
                else:
                    # Mixed signs - use difference instead
                    ratio = w_median - l_median
                    comparison_type = 'difference'
                comparison_value = ratio
                comparison_type = 'ratio'
            else:
                comparison_value = w_median - l_median
                comparison_type = 'difference'
        
        # Simple effect size (Cohen's d approximation)
        pooled_std = np.sqrt((w_values.std()**2 + l_values.std()**2) / 2)
        if pooled_std > 0:
            effect_size = (w_mean - l_mean) / pooled_std
        else:
            effect_size = 0
        
        comparison.append({
            'metric': metric,
            'winner_median': w_median,
            'loser_median': l_median,
            'winner_mean': w_mean,
            'loser_mean': l_mean,
            'comparison_value': comparison_value,
            'comparison_type': comparison_type,
            'effect_size': effect_size,
            'winner_n': len(w_values),
            'loser_n': len(l_values),
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    # Sort by absolute effect size
    if not comparison_df.empty:
        comparison_df['abs_effect'] = comparison_df['effect_size'].abs()
        comparison_df = comparison_df.sort_values('abs_effect', ascending=False)
    
    result = {
        'n_winners': n_winners,
        'n_losers': n_losers,
        'winner_pct': n_winners / (n_winners + n_losers) * 100,
        'winner_median_return': winners[return_col].median(),
        'loser_median_return': losers[return_col].median(),
        'comparison': comparison_df,
    }
    
    # Run OLS on losers if we have enough samples
    n_features = len(valid_metrics)
    if n_losers >= max(min_samples_for_ols, n_features * 2):
        try:
            from sklearn.linear_model import Ridge
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            
            X = losers[valid_metrics].copy()
            y = losers[return_col].clip(-5, 5)  # Winsorize
            
            mask = y.notna()
            X, y = X[mask], y[mask]
            
            if len(y) >= min_samples_for_ols:
                imputer = SimpleImputer(strategy='median')
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(imputer.fit_transform(X))
                
                model = Ridge(alpha=1.0)
                model.fit(X_scaled, y)
                r_squared = model.score(X_scaled, y)
                
                loser_coef_df = pd.DataFrame({
                    'metric': valid_metrics,
                    'coefficient': model.coef_,
                    'abs_coef': np.abs(model.coef_)
                }).sort_values('abs_coef', ascending=False)
                
                result['loser_ols'] = {
                    'r_squared': r_squared,
                    'n_samples': len(y),
                    'coefficients': loser_coef_df,
                }
        except Exception as e:
            result['loser_ols_error'] = str(e)
    else:
        result['loser_ols_note'] = f"Not enough losers for OLS ({n_losers} < {max(min_samples_for_ols, n_features * 2)})"
    
    return result


def save_winners_losers_analysis(output_folder, wl_results, prefix="winners_losers"):
    """Save winners vs losers analysis to CSV."""
    if wl_results is None:
        return []
    
    saved = []
    
    # Save comparison
    if 'comparison' in wl_results and not wl_results['comparison'].empty:
        path = os.path.join(output_folder, "data", f"{prefix}_comparison.csv")
        wl_results['comparison'].to_csv(path, index=False)
        saved.append(path)
    
    # Save loser OLS coefficients if available
    if 'loser_ols' in wl_results and 'coefficients' in wl_results['loser_ols']:
        path = os.path.join(output_folder, "data", f"{prefix}_loser_ols.csv")
        wl_results['loser_ols']['coefficients'].to_csv(path, index=False)
        saved.append(path)
    
    return saved


def plot_winners_losers(output_folder, wl_results, filename="winners_vs_losers.png"):
    """Visualize winners vs losers comparison."""
    if not HAS_MATPLOTLIB or wl_results is None:
        return None
    
    if 'comparison' not in wl_results or wl_results['comparison'].empty:
        return None
    
    comp_df = wl_results['comparison'].head(15)  # Top 15 by effect size
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Effect sizes (what differentiates winners from losers)
    ax1 = axes[0]
    colors = ['#2ecc71' if e > 0 else '#e74c3c' for e in comp_df['effect_size']]
    y_pos = range(len(comp_df))
    
    ax1.barh(y_pos, comp_df['effect_size'], color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(comp_df['metric'], fontsize=9)
    ax1.axvline(x=0, color='black', linewidth=0.8)
    ax1.set_xlabel('Effect Size (Cohen\'s d)')
    ax1.set_title('What Differentiates Winners from Losers\n(positive = winners higher)', fontweight='bold')
    
    # Add interpretation
    ax1.annotate(f"Winners: {wl_results['n_winners']:,}\nLosers: {wl_results['n_losers']:,}",
                xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Median comparison
    ax2 = axes[1]
    
    # Normalize for comparison (use z-scores)
    w_medians = comp_df['winner_median']
    l_medians = comp_df['loser_median']
    
    x = np.arange(len(comp_df))
    width = 0.35
    
    # Standardize for visual comparison
    all_vals = pd.concat([w_medians, l_medians])
    mean_val = all_vals.mean()
    std_val = all_vals.std()
    if std_val > 0:
        w_norm = (w_medians - mean_val) / std_val
        l_norm = (l_medians - mean_val) / std_val
    else:
        w_norm = w_medians
        l_norm = l_medians
    
    ax2.bar(x - width/2, w_norm, width, label='Winners', color='#2ecc71', alpha=0.8)
    ax2.bar(x + width/2, l_norm, width, label='Losers', color='#e74c3c', alpha=0.8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(comp_df['metric'], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Standardized Median')
    ax2.set_title('Metric Medians: Winners vs Losers', fontweight='bold')
    ax2.legend()
    ax2.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    path = os.path.join(output_folder, "visualizations", filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return path


def compute_ols_weighted_ranking(postrank_df, ols_coefficients):
    """
    Re-rank the top stocks using OLS coefficients as weights.
    
    This creates an 'OLS-weighted score' by multiplying each metric by its
    empirically-derived coefficient and summing.
    """
    if postrank_df is None or postrank_df.empty:
        return None
    if ols_coefficients is None or ols_coefficients.empty:
        return None
    
    df = postrank_df.copy()
    
    # Get metrics that are in both the postrank and the coefficients
    coef_dict = dict(zip(ols_coefficients['metric'], ols_coefficients['coefficient']))
    available_metrics = [m for m in coef_dict.keys() if m in df.columns]
    
    if not available_metrics:
        return None
    
    # Compute OLS-weighted score
    ols_score = pd.Series(0.0, index=df.index)
    
    for metric in available_metrics:
        # Standardize the metric (z-score)
        col = df[metric].replace([np.inf, -np.inf], np.nan)
        col_mean = col.mean()
        col_std = col.std()
        if col_std > 0:
            standardized = (col - col_mean) / col_std
        else:
            standardized = 0
        
        # Multiply by coefficient
        ols_score += standardized.fillna(0) * coef_dict[metric]
    
    df['OLS_Score'] = ols_score
    df['OLS_Rank'] = df['OLS_Score'].rank(ascending=False)
    
    # Sort by OLS score (higher is better based on coefficients)
    df = df.sort_values('OLS_Score', ascending=False)
    
    return df


def save_stock_picks(output_folder, postrank_df, ols_reranked_df=None):
    """Save the stock picks to CSV with rankings."""
    if postrank_df is None or postrank_df.empty:
        return None
    
    # Select key columns for the stock picks output
    key_cols = ['source', 'BoScore', 'AggScore', 'rankOfRanks', 
                'Altman-Z', 'Piotroski', 'CycleHeat', 'moatScore',
                'grahamNumberToPrice', 'earnYield', 'returnOnEquity']
    
    available_cols = ['source'] + [c for c in key_cols[1:] if c in postrank_df.columns]
    
    picks_df = postrank_df[available_cols].copy()
    picks_df['Original_Rank'] = range(1, len(picks_df) + 1)
    
    # Add OLS ranking if available
    if ols_reranked_df is not None and 'OLS_Rank' in ols_reranked_df.columns:
        ols_ranks = ols_reranked_df[['source', 'OLS_Rank', 'OLS_Score']].copy()
        picks_df = picks_df.merge(ols_ranks, on='source', how='left')
    
    path = os.path.join(output_folder, "data", "stock_picks.csv")
    picks_df.to_csv(path, index=False)
    
    return picks_df


def generate_html_report(output_folder, results_dict, stock_picks_df=None, ols_reranked_df=None):
    """Generate an HTML summary report including stock picks."""
    
    html_parts = []
    
    # Header
    html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <title>Valuation Analysis Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; }
        h2 { color: #34495e; margin-top: 30px; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
        h3 { color: #7f8c8d; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #3498db; color: white; position: sticky; top: 0; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #e8f4f8; }
        .positive { color: #27ae60; font-weight: bold; }
        .negative { color: #e74c3c; font-weight: bold; }
        .metric-box { background: #ecf0f1; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .stat-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }
        .stat-card h4 { margin: 0; font-size: 14px; opacity: 0.9; }
        .stat-card .value { font-size: 28px; font-weight: bold; margin: 10px 0; }
        .stat-card.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
        .stat-card.orange { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
        img { max-width: 100%; height: auto; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .timestamp { color: #95a5a6; font-size: 12px; }
        .stock-table { max-height: 600px; overflow-y: auto; display: block; }
        .stock-table table { display: table; }
        .top-pick { background-color: #d5f5e3 !important; }
        .nav { background: #34495e; padding: 10px 20px; margin: -30px -30px 30px -30px; border-radius: 10px 10px 0 0; }
        .nav a { color: white; margin-right: 20px; text-decoration: none; }
        .nav a:hover { text-decoration: underline; }
    </style>
</head>
<body>
<div class="container">
<div class="nav">
    <a href="#summary">Summary</a>
    <a href="#stocks">Stock Picks</a>
    <a href="#backtest">Backtest</a>
    <a href="#ols">OLS Analysis</a>
</div>
""")
    
    # Title and timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html_parts.append(f"""
    <h1 id="summary">Valuation Analysis Report</h1>
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
    
    # Stock Picks Section
    if stock_picks_df is not None and not stock_picks_df.empty:
        n_stocks = len(stock_picks_df)
        html_parts.append(f"""
    <h2 id="stocks">Stock Picks ({n_stocks} stocks)</h2>
    <p>Stocks ranked by the algorithm. OLS_Score uses empirically-derived coefficients to re-weight metrics based on historical return prediction.</p>
""")
        
        # Show top 20 stocks in the report (full list in CSV)
        display_df = stock_picks_df.head(20).copy()
        
        # Format numeric columns
        for col in display_df.columns:
            if display_df[col].dtype in ['float64', 'float32']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        
        html_parts.append('<div class="stock-table">')
        html_parts.append(display_df.to_html(index=False, escape=False))
        html_parts.append('</div>')
        html_parts.append(f'<p><em>Showing top 20 of {n_stocks} stocks. Full list in data/stock_picks.csv</em></p>')
    
    # OLS-Weighted Re-ranking
    if ols_reranked_df is not None and not ols_reranked_df.empty:
        html_parts.append("""
    <h3>OLS-Weighted Re-Ranking</h3>
    <p>Stocks re-ranked using OLS coefficients as weights. This prioritizes stocks with metrics that historically predicted better returns.</p>
""")
        # Show comparison of original vs OLS ranking
        compare_cols = ['source', 'OLS_Rank', 'OLS_Score']
        if 'AggScore' in ols_reranked_df.columns:
            compare_cols.insert(2, 'AggScore')
        
        available_compare = [c for c in compare_cols if c in ols_reranked_df.columns]
        compare_df = ols_reranked_df[available_compare].head(15).copy()
        
        # Format
        for col in compare_df.columns:
            if compare_df[col].dtype in ['float64', 'float32']:
                compare_df[col] = compare_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        
        html_parts.append(compare_df.to_html(index=False))
    
    # Scenario summary table
    html_parts.append('<h2 id="backtest">Backtest Results</h2>')
    if 'summary' in results_dict and results_dict['summary'] is not None and not results_dict['summary'].empty:
        html_parts.append("<h3>Scenario Summary</h3>")
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
    <h3>Scenario Comparison</h3>
    <img src="visualizations/scenario_comparison.png" alt="Scenario Comparison">
""")
    
    # Cohort Analysis Section
    if 'cohort_analysis' in results_dict and results_dict['cohort_analysis'] is not None:
        cohort = results_dict['cohort_analysis']
        html_parts.append("""
    <h2>Ranking Cohort Analysis</h2>
    <p>This analysis validates the ranking algorithm by comparing returns across different percentile groups.
    If the algorithm works, better-ranked stocks should have higher returns.</p>
""")
        
        # Summary stats
        summary = cohort.get('summary', {})
        if summary:
            ratio = summary.get('top1_vs_bottom50_ratio', np.nan)
            monotonic = summary.get('monotonic', False)
            avg_marg = summary.get('avg_marginal_improvement', np.nan)
            
            html_parts.append(f"""
    <div class="stat-grid">
        <div class="stat-card {'green' if ratio > 1 else 'orange'}">
            <h4>Top 1% vs Bottom 50%</h4>
            <div class="value">{ratio:.2f}x</div>
        </div>
        <div class="stat-card {'green' if monotonic else 'orange'}">
            <h4>Monotonic Improvement</h4>
            <div class="value">{'Yes' if monotonic else 'No'}</div>
        </div>
        <div class="stat-card">
            <h4>Avg Marginal Improvement</h4>
            <div class="value">{avg_marg:.1f}%</div>
        </div>
    </div>
""")
        
        # Cohort stats table
        if 'cohort_stats' in cohort:
            stats_df = cohort['cohort_stats'].copy()
            
            # Format numeric columns
            for col in ['mean_return', 'median_return', 'std_return', 'min_return', 'max_return']:
                if col in stats_df.columns:
                    stats_df[col] = stats_df[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
            if 'positive_pct' in stats_df.columns:
                stats_df['positive_pct'] = stats_df['positive_pct'].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "N/A")
            
            html_parts.append("<h3>Returns by Cohort</h3>")
            html_parts.append(stats_df.to_html(index=False))
        
        # Ratio matrix (make it upper triangular for display)
        if 'ratio_matrix' in cohort:
            ratio_df = cohort['ratio_matrix'].copy()
            
            # Format values
            for col in ratio_df.columns:
                ratio_df[col] = ratio_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) and x != 0 else "")
            
            html_parts.append("""
    <h3>Return Ratio Matrix</h3>
    <p><em>Each cell shows: row cohort median / column cohort median. Values >1 mean the row outperforms the column.</em></p>
""")
            html_parts.append(ratio_df.to_html())
        
        # Cohort visualization
        if os.path.exists(os.path.join(output_folder, "visualizations", "cohort_analysis.png")):
            html_parts.append('<img src="visualizations/cohort_analysis.png" alt="Cohort Analysis">')
    
    # Winners vs Losers Analysis
    if 'winners_losers' in results_dict and results_dict['winners_losers'] is not None:
        wl = results_dict['winners_losers']
        html_parts.append(f"""
    <h2>Winners vs Losers Analysis</h2>
    <p>Comparing metrics between stocks that had positive returns (winners) vs negative returns (losers).
    This reveals which metrics differentiate success from failure.</p>
    
    <div class="stat-grid">
        <div class="stat-card green">
            <h4>Winners</h4>
            <div class="value">{wl['n_winners']:,}</div>
            <p style="margin:0;font-size:12px;">Median: {wl['winner_median_return']*100:.1f}%</p>
        </div>
        <div class="stat-card orange">
            <h4>Losers</h4>
            <div class="value">{wl['n_losers']:,}</div>
            <p style="margin:0;font-size:12px;">Median: {wl['loser_median_return']*100:.1f}%</p>
        </div>
        <div class="stat-card">
            <h4>Win Rate</h4>
            <div class="value">{wl['winner_pct']:.0f}%</div>
        </div>
    </div>
""")
        
        # Show comparison table (top differences)
        if 'comparison' in wl and not wl['comparison'].empty:
            comp_df = wl['comparison'].head(15).copy()
            
            # Format for display
            display_cols = ['metric', 'winner_median', 'loser_median', 'effect_size']
            display_df = comp_df[display_cols].copy()
            display_df['winner_median'] = display_df['winner_median'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            display_df['loser_median'] = display_df['loser_median'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            display_df['effect_size'] = display_df['effect_size'].apply(
                lambda x: f'<span class="positive">{x:+.3f}</span>' if x > 0.2 
                else f'<span class="negative">{x:+.3f}</span>' if x < -0.2 
                else f"{x:+.3f}"
            )
            display_df.columns = ['Metric', 'Winner Median', 'Loser Median', 'Effect Size']
            
            html_parts.append("""
    <h3>Metrics That Differentiate Winners from Losers</h3>
    <p><em>Effect Size (Cohen's d): >0.2 small, >0.5 medium, >0.8 large. Positive = winners have higher values.</em></p>
""")
            html_parts.append(display_df.to_html(index=False, escape=False))
        
        # Loser OLS results if available
        if 'loser_ols' in wl:
            loser_ols = wl['loser_ols']
            html_parts.append(f"""
    <h3>What Predicts Losses? (OLS on Losers Only)</h3>
    <div class="metric-box">
        <p><strong>R-squared:</strong> {loser_ols['r_squared']:.4f}</p>
        <p><strong>Samples:</strong> {loser_ols['n_samples']:,} losing stocks</p>
    </div>
    <p><em>Negative coefficients = higher values led to worse (more negative) returns among losers.</em></p>
""")
            if 'coefficients' in loser_ols:
                coef_df = loser_ols['coefficients'].head(10).copy()
                coef_df['coefficient'] = coef_df['coefficient'].apply(
                    lambda x: f'<span class="negative">{x:+.4f}</span>' if x < 0 
                    else f'<span class="positive">{x:+.4f}</span>'
                )
                html_parts.append(coef_df[['metric', 'coefficient']].to_html(index=False, escape=False))
        elif 'loser_ols_note' in wl:
            html_parts.append(f'<p><em>{wl["loser_ols_note"]}</em></p>')
        
        # Visualization
        if os.path.exists(os.path.join(output_folder, "visualizations", "winners_vs_losers.png")):
            html_parts.append('<img src="visualizations/winners_vs_losers.png" alt="Winners vs Losers">')
    
    # OLS Results - All Stocks
    if 'ols_analysis' in results_dict and results_dict['ols_analysis'] is not None:
        ols = results_dict['ols_analysis']
        html_parts.append(f"""
    <h2 id="ols">OLS Analysis: All Stocks</h2>
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
    
    # Cohort analysis on all-stocks OLS
    cohort_results = None
    if 'ols_analysis' in results_dict and results_dict['ols_analysis'] is not None:
        ols_df = results_dict['ols_analysis'].get('analysis_df')
        if ols_df is not None and not ols_df.empty:
            # Use BoScore if available, otherwise just use return ranking
            score_col = 'score' if 'score' in ols_df.columns else None
            cohort_results = compute_cohort_analysis(ols_df, score_col=score_col, return_col='total_return')
            
            if cohort_results is not None:
                save_cohort_analysis(output_folder, cohort_results, "cohort_all_stocks")
                results_dict['cohort_analysis'] = cohort_results
                if verbose:
                    print("  - Saved cohort analysis")
                    
                    # Print quick summary
                    summary = cohort_results.get('summary', {})
                    if summary.get('top1_vs_bottom50_ratio'):
                        ratio = summary['top1_vs_bottom50_ratio']
                        print(f"    Top 1% vs Bottom 50% median return ratio: {ratio:.2f}x")
                    if summary.get('monotonic'):
                        print(f"    Monotonic improvement: Yes")
    
    # Winners vs Losers analysis
    wl_results = None
    if 'ols_analysis' in results_dict and results_dict['ols_analysis'] is not None:
        ols_df = results_dict['ols_analysis'].get('analysis_df')
        if ols_df is not None and not ols_df.empty:
            wl_results = analyze_winners_vs_losers(ols_df, return_col='total_return')
            
            if wl_results is not None:
                save_winners_losers_analysis(output_folder, wl_results)
                results_dict['winners_losers'] = wl_results
                if verbose:
                    print("  - Saved winners vs losers analysis")
                    print(f"    Winners: {wl_results['n_winners']:,} ({wl_results['winner_pct']:.0f}%)")
                    print(f"    Losers: {wl_results['n_losers']:,}")
                    
                    if 'loser_ols' in wl_results:
                        print(f"    Loser OLS R²: {wl_results['loser_ols']['r_squared']:.4f}")
    
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
        
        # Cohort analysis visualization
        if 'cohort_analysis' in results_dict and results_dict['cohort_analysis'] is not None:
            path = plot_cohort_analysis(output_folder, results_dict['cohort_analysis'])
            if path and verbose:
                print(f"  - {os.path.basename(path)}")
        
        # Winners vs Losers visualization
        if 'winners_losers' in results_dict and results_dict['winners_losers'] is not None:
            path = plot_winners_losers(output_folder, results_dict['winners_losers'])
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
    
    # Load postRank data for stock picks
    stock_picks_df = None
    ols_reranked_df = None
    
    try:
        import glob
        postrank_files = glob.glob('postRank_*.pickle')
        if postrank_files:
            postrank_files.sort(reverse=True)
            postrank_data = pd.read_pickle(postrank_files[0])
            postrank_df = postrank_data.get('postRank', pd.DataFrame())
            
            if not postrank_df.empty:
                if verbose:
                    print(f"\nLoaded postRank: {postrank_files[0]}")
                
                # Compute OLS-weighted ranking if we have top100 coefficients
                if 'top100_ols' in results_dict and results_dict['top100_ols'] is not None:
                    coefficients = results_dict['top100_ols'].get('coefficients')
                    if coefficients is not None:
                        ols_reranked_df = compute_ols_weighted_ranking(postrank_df, coefficients)
                        if ols_reranked_df is not None and verbose:
                            print("  - Computed OLS-weighted re-ranking")
                
                # Save stock picks
                stock_picks_df = save_stock_picks(output_folder, postrank_df, ols_reranked_df)
                if stock_picks_df is not None and verbose:
                    print("  - Saved stock picks CSV")
    except Exception as e:
        if verbose:
            print(f"  - Warning: Could not load postRank: {e}")
    
    # Generate HTML report
    if verbose:
        print("\nGenerating HTML report...")
    
    report_path = generate_html_report(output_folder, results_dict, stock_picks_df, ols_reranked_df)
    if verbose:
        print(f"  - {os.path.basename(report_path)}")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"All outputs saved to: {output_folder}")
        print(f"Open report: {os.path.join(output_folder, 'report.html')}")
        print('='*70)
    
    return output_folder
