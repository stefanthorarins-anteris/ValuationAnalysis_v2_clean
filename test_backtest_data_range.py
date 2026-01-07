"""
Check data availability for backtesting
- How far back does the data go?
- How many stocks have sufficient history?
- What's the minimum viable backtest horizon?
"""

import pandas as pd
import numpy as np
import glob

# Find pickle files
pickle_files = glob.glob('Bometric_dic*.pickle')
print(f"Found {len(pickle_files)} pickle file(s)")

for pf in pickle_files:
    print(f"\n{'='*70}")
    print(f"File: {pf}")
    print('='*70)
    
    bm = pd.read_pickle(pf)
    cdx_df = bm.get('cdx_df', pd.DataFrame())
    bometric_df = bm.get('BoMetric_df', pd.DataFrame())
    
    if 'date' not in cdx_df.columns:
        print("  No 'date' column in cdx_df")
        continue
    
    cdx_df['date'] = pd.to_datetime(cdx_df['date'])
    
    # Overall date range
    min_date = cdx_df['date'].min()
    max_date = cdx_df['date'].max()
    print(f"\nData range: {min_date.date()} to {max_date.date()}")
    print(f"Span: {(max_date - min_date).days / 365.25:.1f} years")
    
    # Check data density by year
    print(f"\nData points by year:")
    cdx_df['year'] = cdx_df['date'].dt.year
    yearly = cdx_df.groupby('year').agg({
        'source': 'nunique',
        'date': 'count'
    }).rename(columns={'source': 'unique_tickers', 'date': 'total_rows'})
    print(yearly.to_string())
    
    # Check how many stocks have N years of history
    print(f"\nStocks with sufficient history:")
    ticker_spans = cdx_df.groupby('source')['date'].agg(['min', 'max'])
    ticker_spans['years'] = (ticker_spans['max'] - ticker_spans['min']).dt.days / 365.25
    
    for min_years in [1, 2, 3, 4, 5]:
        n_stocks = (ticker_spans['years'] >= min_years).sum()
        pct = n_stocks / len(ticker_spans) * 100
        print(f"  >= {min_years} years: {n_stocks:,} stocks ({pct:.1f}%)")
    
    # Check data points per stock (need 4+ quarters for meaningful analysis)
    print(f"\nData points per stock (quartely data):")
    points_per_ticker = cdx_df.groupby('source').size()
    for min_points in [4, 8, 12, 16, 20]:
        n_stocks = (points_per_ticker >= min_points).sum()
        pct = n_stocks / len(points_per_ticker) * 100
        print(f"  >= {min_points} quarters: {n_stocks:,} stocks ({pct:.1f}%)")
    
    # Recommend backtest horizon
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print('='*70)
    
    years_available = (max_date - min_date).days / 365.25
    
    if years_available >= 5:
        print(f"  5-year backtest possible (data from {min_date.year})")
        print(f"  Buy date: ~{max_date.year - 5} or earlier")
    elif years_available >= 3:
        print(f"  3-year backtest possible")
        print(f"  Buy date: ~{max_date.year - 3}")
    else:
        print(f"  Limited backtest possible ({years_available:.1f} years of data)")
    
    # Sample of oldest data
    print(f"\nSample of oldest data points:")
    oldest = cdx_df.nsmallest(5, 'date')[['source', 'date', 'marketCap', 'netIncome']].copy()
    print(oldest.to_string())

