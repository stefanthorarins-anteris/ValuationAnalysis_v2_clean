import pandas as pd
import numpy as np
import sys
import configuration as cf
import utils as utils
import getData_gen as gdg
import calcScore as csf
import postBo as pb
import gains as gains
from datetime import datetime, timedelta
from tqdm import tqdm

"""
Portfolio Backtesting Module

Supports two modes:
1. One-and-done: Buy top N stocks once at a point in time, hold until evaluation
2. Periodic rebalance: Rebalance portfolio at specified intervals (annual/quarterly)
"""


def runBacktest(loadfname=None, buy_year=None, topn=10, mode='one_and_done', 
                rebalance_freq='A', eval_years=[1, 2, 3, 5], verbose=True):
    """
    Main backtest function.
    
    Parameters:
    -----------
    loadfname : str
        Filename of pickled BoMetric data to load
    buy_year : int
        Year to start buying (e.g., 2020). If None, uses 2 years before latest data.
    topn : int
        Number of top stocks to buy
    mode : str
        'one_and_done' - buy once, hold forever
        'rebalance' - periodically rebalance portfolio
    rebalance_freq : str
        'A' for annual, 'Q' for quarterly (only used if mode='rebalance')
    eval_years : list
        Years after buy date to evaluate returns (e.g., [1, 2, 3, 5])
    verbose : bool
        Print progress and diagnostics
        
    Returns:
    --------
    dict with:
        'summary': DataFrame with returns by evaluation period
        'holdings': DataFrame with individual stock performance
        'buy_date': datetime of initial buy
        'symbols': list of symbols bought
    """
    # Load configuration and data
    cfg = cf.getDataFetchConfiguration([])
    if loadfname is None:
        loadfname = cfg.get('loadBoMetricfname')
    
    baseurl = cfg.get('baseurl', 'https://financialmodelingprep.com/api/')
    api_key = cfg.get('api_key')
    
    if verbose:
        print(f"Loading data from: {loadfname}")
    
    load_dic = {'loadBoMetric': 1, 'loadBoMetricfname': loadfname}
    dmdic = utils.loadWrapper('metric', load_dic)
    
    # Get data
    BoMetric_df = dmdic['BoMetric_df'].copy()
    cdx_df = dmdic['cdx_df'].copy()
    
    # Ensure dates are datetime
    BoMetric_df['date'] = pd.to_datetime(BoMetric_df['date'])
    cdx_df['date'] = pd.to_datetime(cdx_df['date'])
    
    # Determine buy date
    latest_date = BoMetric_df['date'].max()
    if buy_year is None:
        buy_year = latest_date.year - 2
    
    buy_date = pd.Timestamp(f"{buy_year}-12-31")
    
    if verbose:
        print(f"Data range: {BoMetric_df['date'].min()} to {latest_date}")
        print(f"Buy date: {buy_date}")
        print(f"Mode: {mode}, Top N: {topn}")
    
    if mode == 'one_and_done':
        return _run_one_and_done(BoMetric_df, cdx_df, dmdic, buy_date, topn, 
                                  eval_years, baseurl, api_key, verbose)
    elif mode == 'rebalance':
        return _run_rebalance(BoMetric_df, cdx_df, dmdic, buy_year, topn,
                               rebalance_freq, eval_years, baseurl, api_key, verbose)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'one_and_done' or 'rebalance'")


def _run_one_and_done(BoMetric_df, cdx_df, dmdic, buy_date, topn, 
                       eval_years, baseurl, api_key, verbose):
    """Execute one-and-done backtest: buy once, hold until evaluation dates."""
    
    if verbose:
        print("\n" + "="*60)
        print("ONE-AND-DONE BACKTEST")
        print("="*60)
    
    # Filter data to only include data available at buy date
    bm_filtered = BoMetric_df[BoMetric_df['date'] <= buy_date].copy()
    cdx_filtered = cdx_df[cdx_df['date'] <= buy_date].copy()
    
    if bm_filtered.empty:
        print(f"ERROR: No data available before {buy_date}")
        return None
    
    if verbose:
        print(f"Data filtered to {buy_date}: {bm_filtered['source'].nunique()} tickers")
    
    # Get rankings using the actual algorithm
    symbols = _get_top_symbols(bm_filtered, cdx_filtered, dmdic, topn, verbose)
    
    if not symbols:
        print("ERROR: No symbols returned from ranking")
        return None
    
    if verbose:
        print(f"\nTop {topn} symbols selected: {symbols}")
    
    # Get buy prices and current/evaluation prices
    holdings = _build_holdings(symbols, buy_date, eval_years, baseurl, api_key, verbose)
    
    if holdings.empty:
        print("ERROR: Could not build holdings (no prices found)")
        return None
    
    # Calculate summary returns
    summary = _calculate_summary(holdings, eval_years)
    
    if verbose:
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"\nPortfolio of {len(holdings)} stocks bought on {buy_date.date()}")
        print(f"\nIndividual Holdings:")
        print(holdings.to_string())
        print(f"\nPortfolio Summary (equal-weighted):")
        print(summary.to_string())
    
    return {
        'summary': summary,
        'holdings': holdings,
        'buy_date': buy_date,
        'symbols': symbols
    }


def _run_rebalance(BoMetric_df, cdx_df, dmdic, start_year, topn,
                    rebalance_freq, eval_years, baseurl, api_key, verbose):
    """Execute rebalancing backtest: periodic portfolio updates."""
    
    if verbose:
        print("\n" + "="*60)
        print(f"REBALANCING BACKTEST ({rebalance_freq})")
        print("="*60)
    
    # Determine rebalance dates
    latest_date = BoMetric_df['date'].max()
    end_year = latest_date.year
    
    if rebalance_freq == 'A':
        rebalance_dates = [pd.Timestamp(f"{y}-12-31") for y in range(start_year, end_year)]
    elif rebalance_freq == 'Q':
        rebalance_dates = []
        for y in range(start_year, end_year + 1):
            for m in [3, 6, 9, 12]:
                dt = pd.Timestamp(f"{y}-{m:02d}-01") + pd.offsets.MonthEnd(0)
                if dt <= latest_date and dt >= pd.Timestamp(f"{start_year}-01-01"):
                    rebalance_dates.append(dt)
    else:
        raise ValueError(f"Unknown rebalance_freq: {rebalance_freq}")
    
    if len(rebalance_dates) < 2:
        print("ERROR: Not enough rebalance dates")
        return None
    
    if verbose:
        print(f"Rebalance dates: {[d.date() for d in rebalance_dates]}")
    
    # Track portfolio value over time
    portfolio_value = 1.0  # Start with $1
    period_returns = []
    all_holdings = []
    
    for i in range(len(rebalance_dates) - 1):
        current_date = rebalance_dates[i]
        next_date = rebalance_dates[i + 1]
        
        if verbose:
            print(f"\n--- Period: {current_date.date()} to {next_date.date()} ---")
        
        # Filter data to current date
        bm_filtered = BoMetric_df[BoMetric_df['date'] <= current_date].copy()
        cdx_filtered = cdx_df[cdx_df['date'] <= current_date].copy()
        
        if bm_filtered.empty:
            if verbose:
                print(f"  No data available, skipping period")
            continue
        
        # Get rankings
        symbols = _get_top_symbols(bm_filtered, cdx_filtered, dmdic, topn, verbose=False)
        
        if not symbols:
            if verbose:
                print(f"  No symbols ranked, skipping period")
            continue
        
        # Calculate period returns
        period_ret = _calculate_period_return(symbols, current_date, next_date, 
                                               baseurl, api_key, verbose)
        
        if not np.isnan(period_ret):
            portfolio_value *= (1 + period_ret)
            period_returns.append({
                'start_date': current_date,
                'end_date': next_date,
                'return': period_ret,
                'cumulative_value': portfolio_value,
                'symbols': symbols
            })
            
            if verbose:
                print(f"  Period return: {period_ret*100:.2f}%, Cumulative: {portfolio_value:.4f}")
    
    # Build results
    returns_df = pd.DataFrame(period_returns)
    
    # Calculate annualized return
    if not returns_df.empty:
        total_years = (returns_df['end_date'].max() - returns_df['start_date'].min()).days / 365.25
        total_return = portfolio_value - 1
        annualized_return = (portfolio_value ** (1/total_years)) - 1 if total_years > 0 else 0
    else:
        total_return = 0
        annualized_return = 0
    
    summary = pd.DataFrame({
        'Total Return': [f"{total_return*100:.2f}%"],
        'Annualized Return': [f"{annualized_return*100:.2f}%"],
        'Periods': [len(period_returns)],
        'Final Value ($1 invested)': [f"${portfolio_value:.4f}"]
    })
    
    if verbose:
        print("\n" + "="*60)
        print("REBALANCING RESULTS")
        print("="*60)
        print(f"\nPeriod Returns:")
        if not returns_df.empty:
            print(returns_df[['start_date', 'end_date', 'return', 'cumulative_value']].to_string())
        print(f"\nSummary:")
        print(summary.to_string())
    
    return {
        'summary': summary,
        'period_returns': returns_df,
        'final_value': portfolio_value,
        'start_year': start_year
    }


def _get_top_symbols(bm_filtered, cdx_filtered, dmdic, topn, verbose=True):
    """Run the ranking algorithm and return top N symbols."""
    
    # Recalculate averages for filtered data
    meandic = csf.getAves2(bm_filtered)
    
    # Build dmdic for postBoWrapper
    temp_dmdic = {
        'BoMetric_df': bm_filtered,
        'BoMetric_ave': meandic['BoMetric_ave'],
        'BoMetric_dateAve': meandic['BoMetric_dateAve'],
        'cdx_df': cdx_filtered,
        'nrScorePeriods': dmdic.get('nrScorePeriods', 8),
        'baseurl': dmdic.get('baseurl', 'https://financialmodelingprep.com/api/'),
        'api_key': dmdic.get('api_key'),
        'period': dmdic.get('period', 'quarter')
    }
    
    # Suppress verbose output during ranking
    import io
    import sys
    
    if not verbose:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
    
    try:
        resdic = pb.postBoWrapper(temp_dmdic)
    finally:
        if not verbose:
            sys.stdout = old_stdout
    
    # Extract top symbols from postRank
    if 'postRank' in resdic:
        postRank = resdic['postRank']
        if 'source' in postRank.columns:
            symbols = postRank['source'].head(topn).tolist()
            return symbols
    
    # Fallback to BoScore if postRank not available
    if 'BoScore_df' in resdic:
        return resdic['BoScore_df']['source'].head(topn).tolist()
    
    return []


def _build_holdings(symbols, buy_date, eval_years, baseurl, api_key, verbose=True):
    """Build holdings DataFrame with buy prices and returns at evaluation dates."""
    
    rows = []
    today = pd.Timestamp.now()
    
    # Cache for price/dividend history
    price_cache = {}
    div_cache = {}
    
    if verbose:
        print(f"\nFetching price data for {len(symbols)} symbols...")
    
    for symbol in tqdm(symbols, disable=not verbose):
        try:
            # Get historical prices
            hist_prices = gains.getHistPrices(symbol, api_key, baseurl)
            if hist_prices.empty:
                if verbose:
                    print(f"  {symbol}: No price history, skipping")
                continue
            
            price_cache[symbol] = hist_prices
            
            # Get buy price
            buy_price = gains.getPrice(symbol, hist_prices, buy_date)
            if pd.isna(buy_price) or buy_price == 0:
                if verbose:
                    print(f"  {symbol}: No buy price at {buy_date.date()}, skipping")
                continue
            
            # Get dividend history
            hist_divs = gains.getHistDivs(symbol, api_key, baseurl)
            div_cache[symbol] = hist_divs
            
            row = {
                'symbol': symbol,
                'buy_date': buy_date,
                'buy_price': buy_price
            }
            
            # Calculate returns for each evaluation period
            for years in eval_years:
                eval_date = buy_date + pd.DateOffset(years=years)
                
                # Don't evaluate future dates
                if eval_date > today:
                    row[f'{years}yr_return'] = np.nan
                    row[f'{years}yr_price'] = np.nan
                    continue
                
                eval_price = gains.getPrice(symbol, hist_prices, eval_date)
                if pd.isna(eval_price):
                    row[f'{years}yr_return'] = np.nan
                    row[f'{years}yr_price'] = np.nan
                    continue
                
                # Get dividends in period
                divs = gains.getDividends(symbol, hist_divs, buy_date, eval_date)
                
                # Total return = (price change + dividends) / buy price
                total_return = (eval_price - buy_price + divs) / buy_price
                row[f'{years}yr_return'] = total_return
                row[f'{years}yr_price'] = eval_price
                row[f'{years}yr_divs'] = divs
            
            # Current value (latest available)
            current_price = gains.getPrice(symbol, hist_prices, today)
            if not pd.isna(current_price):
                divs_to_now = gains.getDividends(symbol, hist_divs, buy_date, today)
                row['current_price'] = current_price
                row['current_return'] = (current_price - buy_price + divs_to_now) / buy_price
            
            rows.append(row)
            
        except Exception as e:
            if verbose:
                print(f"  {symbol}: Error - {e}")
            continue
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows)


def _calculate_summary(holdings, eval_years):
    """Calculate portfolio summary statistics."""
    
    summary_data = []
    
    for years in eval_years:
        col = f'{years}yr_return'
        if col in holdings.columns:
            returns = holdings[col].dropna()
            if len(returns) > 0:
                avg_return = returns.mean()
                median_return = returns.median()
                min_return = returns.min()
                max_return = returns.max()
                positive_pct = (returns > 0).sum() / len(returns) * 100
                
                summary_data.append({
                    'Period': f'{years} Year',
                    'Avg Return': f"{avg_return*100:.2f}%",
                    'Median Return': f"{median_return*100:.2f}%",
                    'Min': f"{min_return*100:.2f}%",
                    'Max': f"{max_return*100:.2f}%",
                    'Positive %': f"{positive_pct:.1f}%",
                    'N Stocks': len(returns)
                })
    
    # Add current returns if available
    if 'current_return' in holdings.columns:
        returns = holdings['current_return'].dropna()
        if len(returns) > 0:
            summary_data.append({
                'Period': 'Current',
                'Avg Return': f"{returns.mean()*100:.2f}%",
                'Median Return': f"{returns.median()*100:.2f}%",
                'Min': f"{returns.min()*100:.2f}%",
                'Max': f"{returns.max()*100:.2f}%",
                'Positive %': f"{(returns > 0).sum() / len(returns) * 100:.1f}%",
                'N Stocks': len(returns)
            })
    
    return pd.DataFrame(summary_data)


def _calculate_period_return(symbols, start_date, end_date, baseurl, api_key, verbose=False):
    """Calculate equal-weighted portfolio return for a period."""
    
    returns = []
    
    for symbol in symbols:
        try:
            hist_prices = gains.getHistPrices(symbol, api_key, baseurl)
            if hist_prices.empty:
                continue
            
            start_price = gains.getPrice(symbol, hist_prices, start_date)
            end_price = gains.getPrice(symbol, hist_prices, end_date)
            
            if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
                continue
            
            # Get dividends
            hist_divs = gains.getHistDivs(symbol, api_key, baseurl)
            divs = gains.getDividends(symbol, hist_divs, start_date, end_date)
            
            ret = (end_price - start_price + divs) / start_price
            returns.append(ret)
            
        except Exception:
            continue
    
    if not returns:
        return np.nan
    
    # Equal-weighted average
    return np.mean(returns)


# Command-line interface
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Portfolio Backtesting')
    parser.add_argument('--loadfname', type=str, default=None, 
                        help='BoMetric pickle file to load')
    parser.add_argument('--buy_year', type=int, default=None,
                        help='Year to start buying (default: 2 years before latest data)')
    parser.add_argument('--topn', type=int, default=10,
                        help='Number of top stocks to buy (default: 10)')
    parser.add_argument('--mode', type=str, default='one_and_done',
                        choices=['one_and_done', 'rebalance'],
                        help='Backtest mode (default: one_and_done)')
    parser.add_argument('--rebalance_freq', type=str, default='A',
                        choices=['A', 'Q'],
                        help='Rebalance frequency: A=annual, Q=quarterly (default: A)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    results = runBacktest(
        loadfname=args.loadfname,
        buy_year=args.buy_year,
        topn=args.topn,
        mode=args.mode,
        rebalance_freq=args.rebalance_freq,
        verbose=not args.quiet
    )
    
    if results:
        print("\nBacktest completed successfully.")
    else:
        print("\nBacktest failed.")
