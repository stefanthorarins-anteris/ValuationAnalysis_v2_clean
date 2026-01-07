"""
Unified Backtesting Module (No API Calls)

Uses ONLY data from the pickle file - no external API calls needed.
Calculates returns from quarterly price data already in cdx_df.

Usage:
  Standalone: python backtest_unified.py --loadfname <pickle> --buy_years 2019,2020,2021
  From pipeline: import backtest_unified; backtest_unified.run_all(dmdic)
"""

import pandas as pd
import numpy as np
import argparse
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import configuration as cf
import utils as utils
import calcScore as csf


def _get_price_from_cdx(cdx_df, symbol, target_date):
    """
    Get price for a symbol nearest to target_date from cdx_df.
    Uses quarterly data - finds the nearest quarter on or before target_date.
    """
    df = cdx_df[cdx_df['source'] == symbol].copy()
    if df.empty:
        return np.nan
    
    df = df.sort_values('date')
    
    # Find nearest date on or before target
    earlier = df[df['date'] <= target_date]
    if not earlier.empty:
        price = earlier.iloc[-1]['price']
        if price is not None and price > 0:
            return float(price)
    
    # Fallback: earliest after target
    later = df[df['date'] > target_date]
    if not later.empty:
        price = later.iloc[0]['price']
        if price is not None and price > 0:
            return float(price)
    
    return np.nan


def _get_dividend_yield_from_cdx(cdx_df, symbol, start_date, end_date):
    """
    Get average annual dividend yield for a symbol over a period.
    Returns cumulative dividend return estimate.
    """
    df = cdx_df[cdx_df['source'] == symbol].copy()
    if df.empty or 'dividendYield' not in df.columns:
        return 0.0
    
    df = df.sort_values('date')
    
    # Filter to period
    period_data = df[(df['date'] > start_date) & (df['date'] <= end_date)]
    if period_data.empty:
        return 0.0
    
    # Get average quarterly dividend yield
    avg_quarterly_yield = period_data['dividendYield'].mean()
    if pd.isna(avg_quarterly_yield):
        return 0.0
    
    # Number of quarters in period
    n_quarters = len(period_data)
    
    # Cumulative dividend return (simple approximation)
    cumulative_div_return = avg_quarterly_yield * n_quarters
    
    return float(cumulative_div_return)


def _get_top_symbols_fast(bm_filtered, cdx_filtered, dmdic, topn, verbose=False):
    """
    Get top N symbols using BoScore only (NO API calls).
    
    This uses calcScore.simpleScore_fromDict to compute BoScore from local data only,
    avoiding the postBoScoreRanking step which makes API calls.
    """
    import io
    
    # Calculate averages for filtered data
    meandic = csf.getAves2(bm_filtered)
    bmav = meandic['BoMetric_ave']
    bmda = meandic['BoMetric_dateAve']
    
    n = dmdic.get('nrScorePeriods', 8)
    
    # Suppress tqdm output during score calculation
    if not verbose:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
    
    try:
        # Calculate BoScore (local computation, NO API calls)
        BoScore_df = csf.simpleScore_fromDict(bm_filtered, bmav, bmda, n)
    finally:
        if not verbose:
            sys.stdout = old_stdout
    
    if BoScore_df.empty:
        return pd.DataFrame()
    
    # Sort by score and return top N
    BoScore_df = BoScore_df.sort_values('score', ascending=False).head(topn)
    
    return BoScore_df


def run_scenario(dmdic, buy_year, eval_years, topn, verbose=True):
    """
    Run a single backtest scenario using only pickle data.
    
    Returns dict with metrics, returns, and analysis.
    """
    BoMetric_df = dmdic['BoMetric_df']
    cdx_df = dmdic['cdx_df']
    
    buy_date = pd.Timestamp(f"{buy_year}-12-31")
    eval_date = pd.Timestamp(f"{buy_year + eval_years}-12-31")
    
    # Filter to point-in-time for ranking
    bm_filtered = BoMetric_df[BoMetric_df['date'] <= buy_date].copy()
    cdx_filtered = cdx_df[cdx_df['date'] <= buy_date].copy()
    
    if bm_filtered.empty:
        return None
    
    # Get rankings using only point-in-time data
    postRank = _get_top_symbols_fast(bm_filtered, cdx_filtered, dmdic, topn)
    if postRank.empty:
        return None
    
    symbols = postRank['source'].tolist()
    
    # Calculate returns using price data from cdx_df (NO API CALLS)
    returns_data = []
    has_dividend_data = 'dividendYield' in cdx_df.columns
    
    for symbol in symbols:
        buy_price = _get_price_from_cdx(cdx_df, symbol, buy_date)
        if pd.isna(buy_price) or buy_price <= 0:
            continue
        
        eval_price = _get_price_from_cdx(cdx_df, symbol, eval_date)
        
        row = {
            'symbol': symbol,
            'buy_price': buy_price,
            'eval_price': eval_price if not pd.isna(eval_price) else np.nan,
        }
        
        if not pd.isna(eval_price) and eval_price > 0:
            # Price return
            price_return = (eval_price - buy_price) / buy_price
            
            # Add dividend return if available
            if has_dividend_data:
                div_return = _get_dividend_yield_from_cdx(cdx_df, symbol, buy_date, eval_date)
                row['price_return'] = price_return
                row['div_return'] = div_return
                row['total_return'] = price_return + div_return
            else:
                row['total_return'] = price_return
        else:
            row['total_return'] = np.nan
        
        # Also get most recent price for current return
        latest_date = cdx_df['date'].max()
        current_price = _get_price_from_cdx(cdx_df, symbol, latest_date)
        if not pd.isna(current_price) and current_price > 0:
            row['current_price'] = current_price
            current_price_return = (current_price - buy_price) / buy_price
            if has_dividend_data:
                current_div_return = _get_dividend_yield_from_cdx(cdx_df, symbol, buy_date, latest_date)
                row['current_return'] = current_price_return + current_div_return
            else:
                row['current_return'] = current_price_return
        
        returns_data.append(row)
    
    if not returns_data:
        return None
    
    returns_df = pd.DataFrame(returns_data)
    
    # Merge with postRank metrics
    analysis_df = postRank.merge(returns_df, left_on='source', right_on='symbol', how='inner')
    
    # Calculate correlations between metrics and returns
    id_cols = ['source', 'date', 'symbol', 'name', 'Unnamed', 'index']
    metric_cols = [c for c in postRank.columns 
                   if postRank[c].dtype in ['float64', 'int64', 'float32', 'int32']
                   and not any(id_str in c for id_str in id_cols)]
    
    correlations = []
    for col in metric_cols:
        if col in analysis_df.columns:
            valid = analysis_df[[col, 'total_return']].dropna()
            if len(valid) >= 5:
                corr = valid.corr().iloc[0, 1]
                correlations.append({'metric': col, 'correlation': corr})
    
    corr_df = pd.DataFrame(correlations)
    if not corr_df.empty:
        corr_df['abs_corr'] = corr_df['correlation'].abs()
        corr_df = corr_df.sort_values('abs_corr', ascending=False)
    
    # Summary stats
    valid_returns = analysis_df['total_return'].dropna()
    
    return {
        'buy_year': buy_year,
        'eval_years': eval_years,
        'n_stocks': len(analysis_df),
        'n_with_returns': len(valid_returns),
        'mean_return': valid_returns.mean() if len(valid_returns) > 0 else np.nan,
        'median_return': valid_returns.median() if len(valid_returns) > 0 else np.nan,
        'std_return': valid_returns.std() if len(valid_returns) > 0 else np.nan,
        'min_return': valid_returns.min() if len(valid_returns) > 0 else np.nan,
        'max_return': valid_returns.max() if len(valid_returns) > 0 else np.nan,
        'positive_pct': (valid_returns > 0).mean() * 100 if len(valid_returns) > 0 else np.nan,
        'analysis_df': analysis_df,
        'correlations': corr_df,
        'symbols': symbols
    }


def run_ols_metrics_vs_returns(dmdic, topn=100, min_stocks=5000, verbose=True):
    """
    Run OLS regression: BoMetric values → total returns
    
    Uses the oldest buy year with sufficient data (>= min_stocks tickers)
    and evaluates returns to the latest available data.
    Uses ALL stocks with valid returns (not just top-ranked).
    
    Parameters:
    -----------
    dmdic : dict
        Data dictionary with BoMetric_df, cdx_df
    topn : int
        Not used (kept for API compatibility)
    min_stocks : int
        Minimum number of stocks required for a year to be "viable"
    verbose : bool
        Print progress
        
    Returns:
    --------
    dict with OLS results, coefficients, and significant predictors
    """
    BoMetric_df = dmdic['BoMetric_df'].copy()
    cdx_df = dmdic['cdx_df'].copy()
    
    # Find oldest year with sufficient data
    yearly_counts = cdx_df.groupby(cdx_df['date'].dt.year)['source'].nunique()
    viable_years = yearly_counts[yearly_counts >= min_stocks].index.tolist()
    
    if not viable_years:
        if verbose:
            print(f"No years with >= {min_stocks} stocks found")
        return None
    
    # Use oldest viable year as buy year
    buy_year = min(viable_years)
    max_year = cdx_df['date'].max().year
    eval_years = max_year - buy_year  # Hold until latest data
    
    if eval_years < 1:
        if verbose:
            print(f"Insufficient time span for evaluation (buy={buy_year}, max={max_year})")
        return None
    
    buy_date = pd.Timestamp(f"{buy_year}-12-31")
    eval_date = cdx_df['date'].max()
    
    if verbose:
        print(f"Buy date: {buy_date.date()} ({yearly_counts[buy_year]} stocks)")
        print(f"Eval date: {eval_date.date()} (~{eval_years} years)")
    
    # Filter to point-in-time
    bm_filtered = BoMetric_df[BoMetric_df['date'] <= buy_date].copy()
    
    # Get ALL unique symbols available at buy date
    symbols = bm_filtered['source'].unique().tolist()
    
    if verbose:
        print(f"Total stocks available at buy date: {len(symbols)}")
    
    # Get BoMetric values for ALL stocks at buy time
    # Use the most recent data point for each stock before buy_date
    metric_rows = []
    for symbol in symbols:
        sym_data = bm_filtered[bm_filtered['source'] == symbol].sort_values('date')
        if not sym_data.empty:
            latest = sym_data.iloc[-1].to_dict()
            metric_rows.append(latest)
    
    if not metric_rows:
        if verbose:
            print("No metric data found")
        return None
    
    metrics_df = pd.DataFrame(metric_rows)
    
    # Calculate returns for ALL stocks
    has_dividend_data = 'dividendYield' in cdx_df.columns
    returns_data = []
    
    for symbol in symbols:
        buy_price = _get_price_from_cdx(cdx_df, symbol, buy_date)
        if pd.isna(buy_price) or buy_price <= 0:
            continue
        
        eval_price = _get_price_from_cdx(cdx_df, symbol, eval_date)
        if pd.isna(eval_price) or eval_price <= 0:
            continue
        
        price_return = (eval_price - buy_price) / buy_price
        
        if has_dividend_data:
            div_return = _get_dividend_yield_from_cdx(cdx_df, symbol, buy_date, eval_date)
            total_return = price_return + div_return
        else:
            total_return = price_return
        
        returns_data.append({'source': symbol, 'total_return': total_return})
    
    if not returns_data:
        if verbose:
            print("No return data calculated")
        return None
    
    returns_df = pd.DataFrame(returns_data)
    
    # Merge metrics with returns
    analysis_df = metrics_df.merge(returns_df, on='source', how='inner')
    
    if verbose:
        print(f"Stocks with valid returns: {len(analysis_df)}")
        print(f"Mean return: {analysis_df['total_return'].mean()*100:.1f}%")
        print(f"Median return: {analysis_df['total_return'].median()*100:.1f}%")
    
    # Identify numeric metric columns (exclude identifiers)
    exclude_cols = ['source', 'date', 'total_return', 'price_return', 'div_return']
    metric_cols = [c for c in analysis_df.columns 
                   if c not in exclude_cols 
                   and analysis_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    # Filter to columns with enough non-NaN values (>= 50%)
    valid_metrics = []
    for col in metric_cols:
        non_nan_pct = analysis_df[col].notna().mean()
        if non_nan_pct >= 0.5:
            valid_metrics.append(col)
    
    if verbose:
        print(f"Valid metrics (>=50% non-NaN): {len(valid_metrics)}")
    
    if len(valid_metrics) == 0:
        if verbose:
            print("No valid metrics for regression")
        return None
    
    # Simple correlations
    correlations = []
    for col in valid_metrics:
        valid = analysis_df[[col, 'total_return']].dropna()
        if len(valid) >= 5:
            corr = valid.corr().iloc[0, 1]
            correlations.append({'metric': col, 'correlation': corr, 'n': len(valid)})
    
    corr_df = pd.DataFrame(correlations)
    if not corr_df.empty:
        corr_df['abs_corr'] = corr_df['correlation'].abs()
        corr_df = corr_df.sort_values('abs_corr', ascending=False)
    
    # Try OLS regression with proper missing data handling
    ols_results = None
    
    try:
        from sklearn.linear_model import Ridge  # Use Ridge for regularization
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data
        X = analysis_df[valid_metrics].copy()
        y = analysis_df['total_return'].copy()
        
        # Add time trend (days since earliest observation in dataset)
        if 'date' in analysis_df.columns:
            dates = pd.to_datetime(analysis_df['date'])
            min_date = dates.min()
            X['time_trend'] = (dates - min_date).dt.days
            valid_metrics_with_time = valid_metrics + ['time_trend']
        else:
            valid_metrics_with_time = valid_metrics
        
        # Only require valid y (returns)
        valid_y_mask = y.notna()
        X = X[valid_y_mask]
        y = y[valid_y_mask]
        
        # Winsorize extreme returns (cap at +/- 500%)
        y = y.clip(-5, 5)
        
        if len(y) < 20:
            if verbose:
                print(f"\nInsufficient samples for OLS ({len(y)} < 20)")
        else:
            # Impute missing X values with median
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            
            # Use Ridge regression (regularized) to handle multicollinearity
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y)
            r_squared = model.score(X_scaled, y)
            
            # Get coefficients (use the metrics list that includes time_trend if added)
            coef_metrics = valid_metrics_with_time if 'time_trend' in X.columns else valid_metrics
            coef_df = pd.DataFrame({
                'metric': coef_metrics,
                'coefficient': model.coef_,
                'abs_coef': np.abs(model.coef_)
            }).sort_values('abs_coef', ascending=False)
            
            ols_results = {
                'r_squared': r_squared,
                'coefficients': coef_df,
                'n_samples': len(y),
                'intercept': model.intercept_
            }
            
            if verbose:
                print(f"\nRidge Regression (n={len(y)}, returns winsorized to +/-500%):")
                print(f"R-squared: {r_squared:.4f}")
                print(f"\nTop 10 coefficients (standardized):")
                print(coef_df.head(10).to_string(index=False))
                
                # Highlight strongest predictors
                top_positive = coef_df[coef_df['coefficient'] > 0].head(3)
                top_negative = coef_df[coef_df['coefficient'] < 0].head(3)
                
                print(f"\nInterpretation:")
                if not top_positive.empty:
                    print(f"  Higher values HELPED returns:")
                    for _, row in top_positive.iterrows():
                        print(f"    {row['metric']}: +{row['coefficient']:.4f}")
                if not top_negative.empty:
                    print(f"  Higher values HURT returns:")
                    for _, row in top_negative.iterrows():
                        print(f"    {row['metric']}: {row['coefficient']:.4f}")
    
    except ImportError:
        if verbose:
            print("\nsklearn not available, showing correlations only:")
            if not corr_df.empty:
                print(corr_df.head(10).to_string(index=False))
    except Exception as e:
        if verbose:
            print(f"\nRegression failed: {e}")
    
    return {
        'buy_year': buy_year,
        'eval_years': eval_years,
        'n_stocks': len(analysis_df),
        'correlations': corr_df,
        'ols_results': ols_results,
        'analysis_df': analysis_df
    }


def run_postrank_ols(dmdic, verbose=True):
    """
    Load the most recent postRank pickle and run OLS on postRank metrics vs returns.
    
    This uses the actual postRank metrics (Altman-Z, Piotroski, CycleHeat, moatScore, etc.)
    that were calculated during the ranking process.
    """
    import glob
    
    # Find most recent postRank pickle
    postrank_files = glob.glob('postRank_*.pickle')
    if not postrank_files:
        if verbose:
            print("No postRank pickle files found. Run the main pipeline first.")
        return None
    
    # Sort by filename (date) and get most recent
    postrank_files.sort(reverse=True)
    latest_file = postrank_files[0]
    
    if verbose:
        print(f"Loading: {latest_file}")
    
    try:
        postrank_data = pd.read_pickle(latest_file)
    except Exception as e:
        if verbose:
            print(f"Error loading {latest_file}: {e}")
        return None
    
    postRank = postrank_data.get('postRank', pd.DataFrame())
    cdx_df = postrank_data.get('cdx_df', dmdic.get('cdx_df', pd.DataFrame()))
    date_created = postrank_data.get('date_created', 'unknown')
    
    if postRank.empty:
        if verbose:
            print("PostRank is empty")
        return None
    
    if verbose:
        print(f"PostRank created: {date_created}")
        print(f"Stocks in postRank: {len(postRank)}")
    
    # Get current prices to calculate returns since postRank was created
    # Use the latest price data available
    cdx_df['date'] = pd.to_datetime(cdx_df['date'])
    eval_date = cdx_df['date'].max()
    
    # Estimate buy date from postRank creation date
    buy_date = pd.Timestamp(date_created) if date_created != 'unknown' else eval_date - pd.DateOffset(months=1)
    
    if verbose:
        print(f"Buy date (approx): {buy_date.date()}")
        print(f"Eval date: {eval_date.date()}")
    
    # Calculate returns for postRank stocks
    symbols = postRank['source'].tolist()
    has_dividend_data = 'dividendYield' in cdx_df.columns
    
    returns_data = []
    for symbol in symbols:
        buy_price = _get_price_from_cdx(cdx_df, symbol, buy_date)
        eval_price = _get_price_from_cdx(cdx_df, symbol, eval_date)
        
        if pd.isna(buy_price) or buy_price <= 0 or pd.isna(eval_price) or eval_price <= 0:
            continue
        
        price_return = (eval_price - buy_price) / buy_price
        
        if has_dividend_data:
            div_return = _get_dividend_yield_from_cdx(cdx_df, symbol, buy_date, eval_date)
            total_return = price_return + div_return
        else:
            total_return = price_return
        
        returns_data.append({'source': symbol, 'total_return': total_return})
    
    if not returns_data:
        if verbose:
            print("No return data available")
        return None
    
    returns_df = pd.DataFrame(returns_data)
    
    # Merge postRank metrics with returns
    analysis_df = postRank.merge(returns_df, on='source', how='inner')
    
    if verbose:
        print(f"Stocks with returns: {len(analysis_df)}")
        print(f"Mean return: {analysis_df['total_return'].mean()*100:.1f}%")
        print(f"Median return: {analysis_df['total_return'].median()*100:.1f}%")
    
    # Get postRank metric columns (exclude identifiers)
    exclude_cols = ['source', 'date', 'total_return', 'symbol', 'Unnamed', 'index', 'rankOfRanks', 'AggScore']
    metric_cols = [c for c in analysis_df.columns 
                   if c not in exclude_cols 
                   and analysis_df[c].dtype in ['float64', 'int64', 'float32', 'int32']
                   and analysis_df[c].notna().mean() >= 0.3]  # At least 30% non-NaN
    
    if verbose:
        print(f"PostRank metrics for OLS: {len(metric_cols)}")
        print(f"  {metric_cols}")
    
    if len(metric_cols) == 0:
        if verbose:
            print("No valid metrics for regression")
        return None
    
    # Run Ridge regression
    try:
        from sklearn.linear_model import Ridge
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
        X = analysis_df[metric_cols].copy()
        y = analysis_df['total_return'].clip(-5, 5)  # Winsorize
        
        mask = y.notna()
        X, y = X[mask], y[mask]
        
        if len(y) < 10:
            if verbose:
                print(f"Insufficient samples ({len(y)} < 10)")
            return None
        
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(imputer.fit_transform(X))
        
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)
        r_squared = model.score(X_scaled, y)
        
        coef_df = pd.DataFrame({
            'metric': metric_cols,
            'coefficient': model.coef_,
            'abs_coef': np.abs(model.coef_)
        }).sort_values('abs_coef', ascending=False)
        
        if verbose:
            print(f"\nRidge Regression (n={len(y)}):")
            print(f"R-squared: {r_squared:.4f}")
            print(f"\nTop 10 coefficients:")
            print(coef_df.head(10).to_string(index=False))
            
            print(f"\nInterpretation:")
            top_pos = coef_df[coef_df['coefficient'] > 0].head(3)
            top_neg = coef_df[coef_df['coefficient'] < 0].head(3)
            if not top_pos.empty:
                print("  HELPED returns:")
                for _, r in top_pos.iterrows():
                    print(f"    {r['metric']}: +{r['coefficient']:.4f}")
            if not top_neg.empty:
                print("  HURT returns:")
                for _, r in top_neg.iterrows():
                    print(f"    {r['metric']}: {r['coefficient']:.4f}")
        
        return {
            'r_squared': r_squared,
            'coefficients': coef_df,
            'n_samples': len(y),
            'postrank_file': latest_file,
            'analysis_df': analysis_df
        }
        
    except Exception as e:
        if verbose:
            print(f"Regression failed: {e}")
        return None


def run_all(dmdic=None, loadfname=None, buy_years=None, eval_years_list=None, 
            topn=100, verbose=True, save_results=True):
    """
    Run all backtest scenarios using ONLY pickle data (no API calls).
    
    Parameters:
    -----------
    dmdic : dict, optional
        Data dictionary from pipeline. If None, loads from loadfname.
    loadfname : str, optional
        Pickle file to load if dmdic not provided.
    buy_years : list of int
        Years to test as buy dates. Default: based on data range
    eval_years_list : list of int
        Evaluation periods to test. Default: [1, 2, 3]
    topn : int
        Number of top stocks to analyze
    verbose : bool
        Print progress
    save_results : bool
        Save results to CSV
        
    Returns:
    --------
    dict with all scenario results and summary
    """
    
    # Load data if needed
    if dmdic is None:
        cfg = cf.getDataFetchConfiguration([])
        if loadfname is None:
            loadfname = cfg.get('loadBoMetricfname')
        
        if verbose:
            print(f"Loading data from: {loadfname}")
        
        load_dic = {'loadBoMetric': 1, 'loadBoMetricfname': loadfname}
        dmdic = utils.loadWrapper('metric', load_dic)
    
    # Ensure dates are datetime
    dmdic['BoMetric_df']['date'] = pd.to_datetime(dmdic['BoMetric_df']['date'])
    dmdic['cdx_df']['date'] = pd.to_datetime(dmdic['cdx_df']['date'])
    
    # Determine data range
    min_date = dmdic['cdx_df']['date'].min()
    max_date = dmdic['cdx_df']['date'].max()
    
    # Defaults based on data range
    if buy_years is None:
        max_year = max_date.year
        # Default: test 3 years before max, with room for evaluation
        buy_years = [max_year - 4, max_year - 3, max_year - 2]
        # Filter out years without enough data
        buy_years = [y for y in buy_years if y >= min_date.year + 1]
    
    if eval_years_list is None:
        eval_years_list = [1, 2, 3]
    
    # Check for dividend data
    has_dividend_data = 'dividendYield' in dmdic['cdx_df'].columns
    
    if verbose:
        print(f"\n{'='*70}")
        print("UNIFIED BACKTESTING (No API Calls)")
        print('='*70)
        print(f"Data range: {min_date.date()} to {max_date.date()}")
        print(f"Buy years: {buy_years}")
        print(f"Eval periods: {eval_years_list} years")
        print(f"Top N: {topn}")
        if has_dividend_data:
            print(f"\nNote: Total returns include price + dividend yield")
        else:
            print(f"\nNote: Using price returns only (no dividend data in pickle)")
    
    # Run all scenarios
    if verbose:
        print(f"\nRunning scenarios...")
    
    all_results = []
    scenario_details = {}
    
    for buy_year in buy_years:
        for eval_years in eval_years_list:
            scenario_key = f"{buy_year}_{eval_years}yr"
            
            # Check if evaluation date is within data range
            eval_date = pd.Timestamp(f"{buy_year + eval_years}-12-31")
            if eval_date > max_date:
                if verbose:
                    print(f"  Skipping {scenario_key}: eval date {eval_date.date()} > max data {max_date.date()}")
                continue
            
            if verbose:
                print(f"  Running: Buy {buy_year}, Eval {eval_years}yr...", end=" ")
            
            result = run_scenario(dmdic, buy_year, eval_years, topn, verbose=False)
            
            if result:
                all_results.append({
                    'buy_year': buy_year,
                    'eval_years': eval_years,
                    'n_stocks': result['n_with_returns'],
                    'mean_return': result['mean_return'],
                    'median_return': result['median_return'],
                    'std_return': result['std_return'],
                    'positive_pct': result['positive_pct'],
                    'min_return': result['min_return'],
                    'max_return': result['max_return'],
                })
                scenario_details[scenario_key] = result
                
                if verbose:
                    print(f"Mean: {result['mean_return']*100:.1f}%, Pos: {result['positive_pct']:.0f}%")
            else:
                if verbose:
                    print("No data")
    
    # Summary table
    summary_df = pd.DataFrame(all_results)
    
    if verbose:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print('='*70)
        
        if not summary_df.empty:
            # Format for display
            display_df = summary_df.copy()
            for col in ['mean_return', 'median_return', 'std_return', 'min_return', 'max_return']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x*100:.1f}%" if not pd.isna(x) else "N/A")
            display_df['positive_pct'] = display_df['positive_pct'].apply(lambda x: f"{x:.0f}%" if not pd.isna(x) else "N/A")
            
            print(display_df.to_string(index=False))
            
            # Best/worst scenarios
            if len(summary_df) > 1:
                valid = summary_df.dropna(subset=['mean_return'])
                if len(valid) > 0:
                    best = valid.loc[valid['mean_return'].idxmax()]
                    worst = valid.loc[valid['mean_return'].idxmin()]
                    print(f"\nBest scenario: Buy {int(best['buy_year'])}, {int(best['eval_years'])}yr hold ({best['mean_return']*100:.1f}% mean)")
                    print(f"Worst scenario: Buy {int(worst['buy_year'])}, {int(worst['eval_years'])}yr hold ({worst['mean_return']*100:.1f}% mean)")
        
        # Top predictors across all scenarios
        print(f"\n{'='*70}")
        print("TOP PREDICTORS (averaged across scenarios)")
        print('='*70)
        
        all_corrs = []
        for key, result in scenario_details.items():
            if 'correlations' in result and not result['correlations'].empty:
                corr_df = result['correlations'].copy()
                corr_df['scenario'] = key
                all_corrs.append(corr_df)
        
        if all_corrs:
            combined = pd.concat(all_corrs, ignore_index=True)
            avg_corrs = combined.groupby('metric')['correlation'].agg(['mean', 'std', 'count']).reset_index()
            avg_corrs = avg_corrs.sort_values('mean', key=abs, ascending=False)
            print(avg_corrs.head(10).to_string(index=False))
        else:
            print("No correlation data available")
    
    # Run OLS analysis on oldest viable data
    if verbose:
        print(f"\n{'='*70}")
        print("OLS: METRICS vs TOTAL RETURNS (oldest viable data)")
        print('='*70)
    
    ols_result = run_ols_metrics_vs_returns(dmdic, topn, min_stocks=500, verbose=verbose)
    
    # Save results
    if save_results and not summary_df.empty:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Summary
        summary_file = f"backtest_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Detailed results per scenario
        for key, result in scenario_details.items():
            if 'analysis_df' in result:
                result['analysis_df'].to_csv(f"backtest_detail_{key}_{timestamp}.csv", index=False)
        
        if verbose:
            print(f"\nResults saved to: backtest_summary_{timestamp}.csv")
    
    # Run OLS on postRank metrics if a postRank pickle exists
    if verbose:
        print(f"\n{'='*70}")
        print("OLS: POSTRANK METRICS vs RETURNS (from saved postRank)")
        print('='*70)
    
    postrank_ols = run_postrank_ols(dmdic, verbose=verbose)
    
    return {
        'summary': summary_df,
        'scenarios': scenario_details,
        'ols_analysis': ols_result,
        'postrank_ols': postrank_ols,
    }


# Command-line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Backtesting (No API Calls)')
    parser.add_argument('--loadfname', type=str, default=None,
                        help='BoMetric pickle file to load')
    parser.add_argument('--buy_years', type=str, default='2020,2021,2022',
                        help='Comma-separated buy years (default: 2020,2021,2022)')
    parser.add_argument('--eval_years', type=str, default='1,2,3',
                        help='Comma-separated eval periods in years (default: 1,2,3)')
    parser.add_argument('--topn', type=int, default=100,
                        help='Number of top stocks (default: 100)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to CSV')
    
    args = parser.parse_args()
    
    buy_years = [int(y.strip()) for y in args.buy_years.split(',')]
    eval_years_list = [int(y.strip()) for y in args.eval_years.split(',')]
    
    results = run_all(
        loadfname=args.loadfname,
        buy_years=buy_years,
        eval_years_list=eval_years_list,
        topn=args.topn,
        verbose=not args.quiet,
        save_results=not args.no_save
    )
