"""
Backtest OLS Analysis

Answers: Which postRank metrics actually predict future returns?

Process:
1. Take rankings from N years ago (point-in-time data only)
2. Extract all postRank metric values for top stocks at that time
3. Get actual returns over the holding period
4. Run OLS: return ~ postRank_metrics
5. Report which metrics are statistically significant predictors

Usage:
  python backtest_ols_analysis.py --buy_year 2020 --eval_years 3 --topn 100
"""

import pandas as pd
import numpy as np
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

import configuration as cf
import utils as utils
import calcScore as csf
import postBo as pb
import postBoRank as pbr
import gains
from tqdm import tqdm

def run_ols_analysis(loadfname=None, buy_year=2020, eval_years=3, topn=100, verbose=True):
    """
    Run OLS analysis to determine which postRank metrics predict returns.
    
    Parameters:
    -----------
    loadfname : str
        Pickle file to load
    buy_year : int
        Year to "buy" stocks (uses only data available at that time)
    eval_years : int
        Years to hold before evaluating returns
    topn : int
        Number of top-ranked stocks to analyze
    verbose : bool
        Print progress
        
    Returns:
    --------
    dict with OLS results, raw data, and summary
    """
    
    # Load configuration and data
    cfg = cf.getDataFetchConfiguration([])
    if loadfname is None:
        loadfname = cfg.get('loadBoMetricfname')
    
    baseurl = cfg.get('baseurl', 'https://financialmodelingprep.com/api/')
    api_key = cfg.get('api_key')
    
    if verbose:
        print(f"{'='*70}")
        print(f"OLS BACKTEST ANALYSIS")
        print(f"{'='*70}")
        print(f"Buy year: {buy_year}")
        print(f"Evaluation period: {eval_years} years")
        print(f"Top N stocks: {topn}")
        print(f"Loading: {loadfname}")
    
    # Load data
    load_dic = {'loadBoMetric': 1, 'loadBoMetricfname': loadfname}
    dmdic = utils.loadWrapper('metric', load_dic)
    
    BoMetric_df = dmdic['BoMetric_df'].copy()
    cdx_df = dmdic['cdx_df'].copy()
    
    # Ensure dates are datetime
    BoMetric_df['date'] = pd.to_datetime(BoMetric_df['date'])
    cdx_df['date'] = pd.to_datetime(cdx_df['date'])
    
    # Set buy date (end of buy year)
    buy_date = pd.Timestamp(f"{buy_year}-12-31")
    eval_date = pd.Timestamp(f"{buy_year + eval_years}-12-31")
    
    if verbose:
        print(f"\nBuy date: {buy_date.date()}")
        print(f"Eval date: {eval_date.date()}")
    
    # Filter data to ONLY what was available at buy date
    bm_filtered = BoMetric_df[BoMetric_df['date'] <= buy_date].copy()
    cdx_filtered = cdx_df[cdx_df['date'] <= buy_date].copy()
    
    if bm_filtered.empty:
        print(f"ERROR: No data available before {buy_date}")
        return None
    
    n_stocks = bm_filtered['source'].nunique()
    if verbose:
        print(f"Stocks available at {buy_date.date()}: {n_stocks}")
    
    if n_stocks < topn:
        print(f"WARNING: Only {n_stocks} stocks available, less than topn={topn}")
        topn = n_stocks
    
    # ==========================================================================
    # STEP 1: Run the ranking algorithm using point-in-time data
    # ==========================================================================
    if verbose:
        print(f"\n{'='*70}")
        print("STEP 1: Running ranking algorithm (point-in-time)")
        print('='*70)
    
    # Recalculate averages for filtered data
    meandic = csf.getAves2(bm_filtered)
    
    # Build dmdic for postBoWrapper
    temp_dmdic = {
        'BoMetric_df': bm_filtered,
        'BoMetric_ave': meandic['BoMetric_ave'],
        'BoMetric_dateAve': meandic['BoMetric_dateAve'],
        'cdx_df': cdx_filtered,
        'nrScorePeriods': dmdic.get('nrScorePeriods', 8),
        'baseurl': baseurl,
        'api_key': api_key,
        'period': dmdic.get('period', 'quarter')
    }
    
    # Run postBoWrapper to get rankings and metrics
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        resdic = pb.postBoWrapper(temp_dmdic)
    finally:
        sys.stdout = old_stdout
    
    if 'postRank' not in resdic:
        print("ERROR: postRank not in results")
        return None
    
    postRank = resdic['postRank'].head(topn).copy()
    
    if verbose:
        print(f"Got top {len(postRank)} stocks from ranking")
        print(f"Columns in postRank: {list(postRank.columns)}")
    
    # ==========================================================================
    # STEP 2: Extract metric values (these are point-in-time values)
    # ==========================================================================
    if verbose:
        print(f"\n{'='*70}")
        print("STEP 2: Extracting postRank metric values")
        print('='*70)
    
    # Identify metric columns (numeric, not identifiers)
    id_cols = ['source', 'date', 'symbol', 'name', 'Unnamed', 'index']
    metric_cols = [c for c in postRank.columns 
                   if postRank[c].dtype in ['float64', 'int64', 'float32', 'int32']
                   and not any(id_str in c for id_str in id_cols)]
    
    if verbose:
        print(f"Metric columns found: {metric_cols}")
    
    # ==========================================================================
    # STEP 3: Get actual returns over holding period
    # ==========================================================================
    if verbose:
        print(f"\n{'='*70}")
        print(f"STEP 3: Fetching actual {eval_years}-year returns")
        print('='*70)
    
    symbols = postRank['source'].tolist()
    returns_data = []
    
    for symbol in tqdm(symbols, disable=not verbose, desc="Fetching prices"):
        try:
            # Get historical prices
            hist_prices = gains.getHistPrices(symbol, api_key, baseurl)
            if hist_prices.empty:
                continue
            
            # Get buy price
            buy_price = gains.getPrice(symbol, hist_prices, buy_date)
            if pd.isna(buy_price) or buy_price == 0:
                continue
            
            # Get eval price
            eval_price = gains.getPrice(symbol, hist_prices, eval_date)
            if pd.isna(eval_price):
                continue
            
            # Get dividends
            hist_divs = gains.getHistDivs(symbol, api_key, baseurl)
            divs = gains.getDividends(symbol, hist_divs, buy_date, eval_date)
            
            # Total return
            total_return = (eval_price - buy_price + divs) / buy_price
            
            returns_data.append({
                'symbol': symbol,
                'buy_price': buy_price,
                'eval_price': eval_price,
                'dividends': divs,
                'total_return': total_return
            })
            
        except Exception as e:
            continue
    
    if not returns_data:
        print("ERROR: No return data fetched")
        return None
    
    returns_df = pd.DataFrame(returns_data)
    
    if verbose:
        print(f"Got returns for {len(returns_df)} stocks")
        print(f"Average return: {returns_df['total_return'].mean()*100:.2f}%")
        print(f"Median return: {returns_df['total_return'].median()*100:.2f}%")
    
    # ==========================================================================
    # STEP 4: Merge metrics with returns
    # ==========================================================================
    if verbose:
        print(f"\n{'='*70}")
        print("STEP 4: Merging metrics with returns")
        print('='*70)
    
    # Merge postRank metrics with returns
    analysis_df = postRank.merge(returns_df, left_on='source', right_on='symbol', how='inner')
    
    if verbose:
        print(f"Merged dataset: {len(analysis_df)} stocks")
    
    # ==========================================================================
    # STEP 5: Run OLS regression
    # ==========================================================================
    if verbose:
        print(f"\n{'='*70}")
        print("STEP 5: OLS Regression Analysis")
        print('='*70)
    
    try:
        import statsmodels.api as sm
        has_statsmodels = True
    except ImportError:
        has_statsmodels = False
        print("WARNING: statsmodels not installed. Using simple correlations instead.")
    
    # Filter to available metrics with sufficient non-NaN values
    valid_metrics = []
    for col in metric_cols:
        if col in analysis_df.columns:
            non_nan = analysis_df[col].notna().sum()
            if non_nan >= len(analysis_df) * 0.5:  # At least 50% non-NaN
                valid_metrics.append(col)
    
    if verbose:
        print(f"Valid metrics (>=50% non-NaN): {valid_metrics}")
    
    # Simple correlations first
    correlations = []
    for col in valid_metrics:
        if col in analysis_df.columns:
            corr = analysis_df[[col, 'total_return']].dropna().corr().iloc[0, 1]
            correlations.append({
                'metric': col,
                'correlation': corr,
                'abs_corr': abs(corr)
            })
    
    corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)
    
    if verbose:
        print(f"\nCorrelations with {eval_years}-year return:")
        print(corr_df.to_string())
    
    # Full OLS if statsmodels available
    ols_results = None
    if has_statsmodels and len(valid_metrics) > 0:
        # Prepare data for OLS
        X = analysis_df[valid_metrics].copy()
        y = analysis_df['total_return'].copy()
        
        # Drop rows with any NaN
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) >= 10:
            # Standardize X for easier interpretation
            X_standardized = (X_clean - X_clean.mean()) / X_clean.std()
            X_standardized = sm.add_constant(X_standardized)
            
            # Fit OLS
            model = sm.OLS(y_clean, X_standardized).fit()
            ols_results = model
            
            if verbose:
                print(f"\n{'='*70}")
                print("OLS REGRESSION RESULTS")
                print('='*70)
                print(model.summary())
                
                # Extract significant predictors
                sig_predictors = model.pvalues[model.pvalues < 0.05].index.tolist()
                if 'const' in sig_predictors:
                    sig_predictors.remove('const')
                
                print(f"\n{'='*70}")
                print("SIGNIFICANT PREDICTORS (p < 0.05)")
                print('='*70)
                for pred in sig_predictors:
                    coef = model.params[pred]
                    pval = model.pvalues[pred]
                    direction = "positive" if coef > 0 else "negative"
                    print(f"  {pred}: coef={coef:.4f} ({direction}), p={pval:.4f}")
                
                if not sig_predictors:
                    print("  No statistically significant predictors at p<0.05")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    if verbose:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print('='*70)
        print(f"Buy year: {buy_year}")
        print(f"Eval period: {eval_years} years")
        print(f"Stocks analyzed: {len(analysis_df)}")
        print(f"Mean return: {analysis_df['total_return'].mean()*100:.2f}%")
        print(f"Median return: {analysis_df['total_return'].median()*100:.2f}%")
        print(f"Std return: {analysis_df['total_return'].std()*100:.2f}%")
        print(f"Min return: {analysis_df['total_return'].min()*100:.2f}%")
        print(f"Max return: {analysis_df['total_return'].max()*100:.2f}%")
        
        # Top predictors by correlation
        if len(corr_df) > 0:
            print(f"\nTop 5 predictors by |correlation|:")
            for _, row in corr_df.head(5).iterrows():
                print(f"  {row['metric']}: r={row['correlation']:.4f}")
    
    return {
        'analysis_df': analysis_df,
        'correlations': corr_df,
        'ols_results': ols_results,
        'buy_year': buy_year,
        'eval_years': eval_years,
        'n_stocks': len(analysis_df),
        'mean_return': analysis_df['total_return'].mean(),
        'postRank': postRank
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest OLS Analysis')
    parser.add_argument('--loadfname', type=str, default=None,
                        help='BoMetric pickle file to load')
    parser.add_argument('--buy_year', type=int, default=2020,
                        help='Year to buy stocks (default: 2020)')
    parser.add_argument('--eval_years', type=int, default=3,
                        help='Years to hold before evaluating (default: 3)')
    parser.add_argument('--topn', type=int, default=100,
                        help='Number of top stocks to analyze (default: 100)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    results = run_ols_analysis(
        loadfname=args.loadfname,
        buy_year=args.buy_year,
        eval_years=args.eval_years,
        topn=args.topn,
        verbose=not args.quiet
    )
    
    if results:
        # Save analysis data for further inspection
        output_file = f"backtest_ols_{args.buy_year}_{args.eval_years}yr.csv"
        results['analysis_df'].to_csv(output_file, index=False)
        print(f"\nAnalysis data saved to: {output_file}")

