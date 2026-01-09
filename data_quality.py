"""
Data Quality Module

Identifies and removes clearly invalid/corrupted data before any analysis.
Logs removed data to CSV for transparency.

Conservative filters that only catch obvious corruption:
1. Negative prices (impossible)
2. Zero prices (invalid for return calculations)
3. Price-to-MarketCap sanity check (catches API garbage)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


def check_price_sanity(row, price_col='price', mcap_col='marketCap', 
                       prev_price=None, prev_mcap=None):
    """
    Check if a price data point is sane.
    
    Returns:
    --------
    tuple: (is_valid, reason) - reason is None if valid
    """
    price = row.get(price_col, np.nan)
    mcap = row.get(mcap_col, np.nan)
    
    # Check 1: Negative price (impossible)
    if pd.notna(price) and price < 0:
        return False, f"negative_price ({price:.4f})"
    
    # Check 2: Zero price (invalid for returns)
    if pd.notna(price) and price == 0:
        return False, "zero_price"
    
    # Check 3: Price-MarketCap sanity
    # If mcap is reasonable but price is absurd, it's likely API corruption
    # A $30B company shouldn't have a $4M stock price
    if pd.notna(price) and pd.notna(mcap) and mcap > 0 and price > 0:
        # Implied shares outstanding
        implied_shares = mcap / price
        
        # If implied shares < 1000, price is likely way too high
        # (Almost no public company has < 1000 shares)
        if implied_shares < 1000:
            return False, f"impossible_price_vs_mcap (price=${price:.2f}, mcap=${mcap/1e9:.2f}B, implied_shares={implied_shares:.0f})"
        
        # If implied shares > 1 trillion, price is likely way too low (or mcap wrong)
        # We're conservative here - some companies have trillions of shares
        if implied_shares > 1e12:
            return False, f"suspicious_price_vs_mcap (price=${price:.6f}, mcap=${mcap/1e9:.2f}B, implied_shares={implied_shares:.0e})"
    
    # Check 4: Extreme quarter-over-quarter jump (if we have previous data)
    # A 1000x jump in one quarter while mcap stays similar is API corruption
    if prev_price is not None and prev_mcap is not None:
        if prev_price > 0 and price > 0:
            price_change_ratio = price / prev_price
            
            # Check if price jumped >100x but mcap stayed within 5x
            if pd.notna(mcap) and pd.notna(prev_mcap) and prev_mcap > 0:
                mcap_change_ratio = mcap / prev_mcap
                
                # Price up 100x but mcap within 5x = corruption
                if price_change_ratio > 100 and 0.2 < mcap_change_ratio < 5:
                    return False, f"price_mcap_mismatch (price {price_change_ratio:.0f}x but mcap {mcap_change_ratio:.2f}x)"
                
                # Price down 100x but mcap within 5x = also suspicious
                if price_change_ratio < 0.01 and 0.2 < mcap_change_ratio < 5:
                    return False, f"price_mcap_mismatch (price {price_change_ratio:.4f}x but mcap {mcap_change_ratio:.2f}x)"
    
    return True, None


def filter_invalid_data(cdx_df, price_col='price', mcap_col='marketCap', 
                        min_periods_required=8, verbose=True):
    """
    Filter out rows with clearly invalid/corrupted price data.
    
    Logic:
    1. Identify all corrupt data points
    2. For each ticker with corruption, find the MOST RECENT corruption date
    3. Remove ALL data at or before that corruption date (keep only newer data)
    4. If remaining data < min_periods_required, remove ticker entirely
    
    This approach: we keep the NEWEST reliable data, since older data before
    corruption may have hidden issues.
    
    Parameters:
    -----------
    cdx_df : DataFrame
        The cdx (fundamentals + price) dataframe
    price_col : str
        Name of price column
    mcap_col : str
        Name of market cap column
    min_periods_required : int
        Minimum data points needed after filtering. If less, remove ticker entirely.
    verbose : bool
        Print summary statistics
    
    Returns:
    --------
    tuple: (clean_df, removed_df)
        - clean_df: DataFrame with invalid rows removed
        - removed_df: DataFrame of removed rows with 'removal_reason' column
    """
    if cdx_df is None or cdx_df.empty:
        return cdx_df, pd.DataFrame()
    
    df = cdx_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by source and date for sequential checking
    df = df.sort_values(['source', 'date']).reset_index(drop=True)
    
    # =========================================================================
    # PASS 1: Identify all corrupt data points
    # =========================================================================
    corrupt_records = []  # (source, date, reason)
    prev_data = {}  # source -> (prev_price, prev_mcap, prev_date)
    
    for idx, row in df.iterrows():
        source = row.get('source', 'unknown')
        price = row.get(price_col, np.nan)
        mcap = row.get(mcap_col, np.nan)
        date = row.get('date', None)
        
        prev_price, prev_mcap, prev_date = prev_data.get(source, (None, None, None))
        
        is_valid, reason = check_price_sanity(
            row, price_col, mcap_col, prev_price, prev_mcap
        )
        
        if not is_valid:
            corrupt_records.append({
                'source': source,
                'date': date,
                'price': price,
                'marketCap': mcap,
                'reason': reason,
            })
        else:
            if pd.notna(price) and price > 0:
                prev_data[source] = (price, mcap, date)
    
    if not corrupt_records:
        if verbose:
            print("No corrupt data found.")
        return df, pd.DataFrame()
    
    corrupt_df = pd.DataFrame(corrupt_records)
    
    # =========================================================================
    # PASS 2: For each ticker, find most recent corruption date
    # =========================================================================
    # Get the most recent (max) corruption date per ticker
    most_recent_corruption = corrupt_df.groupby('source')['date'].max().to_dict()
    
    if verbose:
        print(f"\nTickers with corruption: {len(most_recent_corruption):,}")
    
    # =========================================================================
    # PASS 3: Remove all data at or before the most recent corruption date
    # =========================================================================
    removal_records = []
    rows_to_remove = set()
    
    for idx, row in df.iterrows():
        source = row.get('source', 'unknown')
        date = row.get('date', None)
        
        if source in most_recent_corruption:
            corruption_date = most_recent_corruption[source]
            
            # Remove all data at or before the corruption date
            if date <= corruption_date:
                rows_to_remove.add(idx)
                removal_records.append({
                    'source': source,
                    'date': date,
                    'price': row.get(price_col, np.nan),
                    'marketCap': row.get(mcap_col, np.nan),
                    'removal_reason': f"data_before_corruption (corruption at {corruption_date.date()})",
                })
    
    # Create intermediate filtered df
    valid_mask = ~df.index.isin(rows_to_remove)
    filtered_df = df[valid_mask].copy()
    
    # =========================================================================
    # PASS 4: Remove tickers with insufficient remaining data
    # =========================================================================
    ticker_counts = filtered_df.groupby('source').size()
    insufficient_tickers = ticker_counts[ticker_counts < min_periods_required].index.tolist()
    
    if insufficient_tickers:
        # Remove these tickers entirely
        for source in insufficient_tickers:
            source_rows = df[df['source'] == source]
            for idx, row in source_rows.iterrows():
                if idx not in rows_to_remove:  # Don't double-count
                    removal_records.append({
                        'source': source,
                        'date': row.get('date', None),
                        'price': row.get(price_col, np.nan),
                        'marketCap': row.get(mcap_col, np.nan),
                        'removal_reason': f"insufficient_data_after_corruption (<{min_periods_required} periods)",
                    })
        
        # Update filtered_df
        filtered_df = filtered_df[~filtered_df['source'].isin(insufficient_tickers)].copy()
    
    removed_df = pd.DataFrame(removal_records)
    
    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        n_total = len(df)
        n_removed = len(removal_records)
        n_kept = len(filtered_df)
        
        print(f"\n{'='*60}")
        print("DATA QUALITY FILTER APPLIED")
        print('='*60)
        print(f"Total rows: {n_total:,}")
        print(f"Rows removed: {n_removed:,} ({n_removed/n_total*100:.2f}%)")
        print(f"Rows kept: {n_kept:,}")
        
        # Count original corrupt points
        n_corrupt_points = len(corrupt_df)
        print(f"\nCorrupt data points detected: {n_corrupt_points:,}")
        
        # Breakdown by reason
        if not corrupt_df.empty:
            reason_counts = corrupt_df['reason'].str.split('(').str[0].value_counts()
            print(f"\nCorruption types:")
            for reason, count in reason_counts.items():
                print(f"  {reason.strip()}: {count:,}")
        
        # Tickers removed entirely
        original_tickers = df['source'].nunique()
        remaining_tickers = filtered_df['source'].nunique()
        removed_tickers = original_tickers - remaining_tickers
        
        print(f"\nTickers:")
        print(f"  Original: {original_tickers:,}")
        print(f"  Removed entirely: {removed_tickers:,}")
        print(f"  Remaining: {remaining_tickers:,}")
        
        if insufficient_tickers:
            print(f"\nTickers removed (insufficient data after corruption): {len(insufficient_tickers):,}")
            if len(insufficient_tickers) <= 20:
                print(f"  {insufficient_tickers}")
        
        print('='*60 + '\n')
    
    return filtered_df, removed_df


def save_removed_data(removed_df, filename=None):
    """
    Save removed data to CSV for transparency.
    
    Parameters:
    -----------
    removed_df : DataFrame
        DataFrame of removed rows from filter_invalid_data()
    filename : str, optional
        Custom filename. Default: removed_data_quality_YYYYMMDD.csv
    """
    if removed_df is None or removed_df.empty:
        return None
    
    if filename is None:
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"removed_data_quality_{date_str}.csv"
    
    # Ensure output directory exists
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    removed_df.to_csv(filepath, index=False)
    
    print(f"Removed data logged to: {filepath}")
    
    return filepath


def apply_data_quality_filter(dmdic, verbose=True, save_log=True):
    """
    Apply data quality filter to the data dictionary.
    
    This should be called early in the pipeline, before any scoring.
    
    Parameters:
    -----------
    dmdic : dict
        Data dictionary with 'cdx_df' and 'BoMetric_df'
    verbose : bool
        Print summary
    save_log : bool
        Save removed data to CSV
    
    Returns:
    --------
    dict : Updated dmdic with filtered data
    """
    if 'cdx_df' not in dmdic:
        if verbose:
            print("Warning: No cdx_df in data dictionary, skipping quality filter")
        return dmdic
    
    # Filter cdx_df
    clean_cdx, removed_cdx = filter_invalid_data(
        dmdic['cdx_df'], 
        price_col='price', 
        mcap_col='marketCap',
        verbose=verbose
    )
    
    # Get list of affected sources (for filtering BoMetric_df consistently)
    if not removed_cdx.empty:
        # Get sources that had ALL their data removed (completely invalid)
        sources_in_clean = set(clean_cdx['source'].unique())
        sources_original = set(dmdic['cdx_df']['source'].unique())
        completely_removed = sources_original - sources_in_clean
        
        if verbose and len(completely_removed) > 0:
            print(f"Tickers completely removed (all data invalid): {len(completely_removed)}")
            if len(completely_removed) <= 20:
                print(f"  {list(completely_removed)}")
        
        # Save log
        if save_log:
            save_removed_data(removed_cdx)
    
    # Update dictionary
    dmdic['cdx_df'] = clean_cdx
    dmdic['removed_data_quality'] = removed_cdx
    
    # Also filter BoMetric_df to only include sources that have valid price data
    if 'BoMetric_df' in dmdic and not clean_cdx.empty:
        valid_sources = set(clean_cdx['source'].unique())
        original_bm_len = len(dmdic['BoMetric_df'])
        dmdic['BoMetric_df'] = dmdic['BoMetric_df'][
            dmdic['BoMetric_df']['source'].isin(valid_sources)
        ].copy()
        
        if verbose:
            new_bm_len = len(dmdic['BoMetric_df'])
            if original_bm_len != new_bm_len:
                print(f"BoMetric_df filtered: {original_bm_len:,} -> {new_bm_len:,} rows")
    
    return dmdic


# Convenience function for standalone use
def filter_pickle_data(pickle_path, output_pickle_path=None, verbose=True):
    """
    Load a pickle, apply quality filter, optionally save cleaned version.
    
    Parameters:
    -----------
    pickle_path : str
        Path to input pickle
    output_pickle_path : str, optional
        Path for cleaned pickle. If None, doesn't save.
    verbose : bool
        Print details
    
    Returns:
    --------
    dict : Cleaned data dictionary
    """
    import pickle
    
    with open(pickle_path, 'rb') as f:
        dmdic = pickle.load(f)
    
    print(f"Loaded: {pickle_path}")
    
    dmdic = apply_data_quality_filter(dmdic, verbose=verbose, save_log=True)
    
    if output_pickle_path:
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(dmdic, f)
        print(f"Saved cleaned data to: {output_pickle_path}")
    
    return dmdic


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply data quality filter to pickle file")
    parser.add_argument('--input', '-i', required=True, help='Input pickle path')
    parser.add_argument('--output', '-o', help='Output pickle path (optional)')
    
    args = parser.parse_args()
    
    filter_pickle_data(args.input, args.output)
