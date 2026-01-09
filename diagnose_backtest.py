"""
Diagnostic script to investigate backtest data quality issues.

Issues to investigate:
1. Absurd return outliers (131,786,956% max)
2. OLS all stocks showing 0 samples
3. What's causing extreme returns
"""

import pandas as pd
import numpy as np
import pickle
import glob
from datetime import datetime

def load_data():
    """Load the most recent BoMetric pickle."""
    files = glob.glob('Bometric_dic*.pickle') + glob.glob('BoMetric_dic*.pickle')
    if not files:
        print("No BoMetric pickle found!")
        return None
    
    files.sort(reverse=True)
    fname = files[0]
    print(f"Loading: {fname}")
    
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    
    return data


def investigate_extreme_returns(dmdic, n_extreme=50):
    """Find and investigate the most extreme returns."""
    
    cdx_df = dmdic['cdx_df'].copy()
    cdx_df['date'] = pd.to_datetime(cdx_df['date'])
    
    min_date = cdx_df['date'].min()
    max_date = cdx_df['date'].max()
    
    print(f"\n{'='*70}")
    print("DATA RANGE")
    print('='*70)
    print(f"Date range: {min_date.date()} to {max_date.date()}")
    print(f"Total rows in cdx_df: {len(cdx_df):,}")
    print(f"Unique symbols: {cdx_df['source'].nunique():,}")
    
    # Use a historical buy date (3 years back)
    buy_year = max_date.year - 3
    buy_date = pd.Timestamp(f"{buy_year}-12-31")
    eval_date = max_date
    
    print(f"\nAnalyzing returns from {buy_date.date()} to {eval_date.date()}")
    
    # Get price column
    price_col = 'price' if 'price' in cdx_df.columns else 'stockPrice'
    print(f"Using price column: '{price_col}'")
    
    # Calculate returns for all stocks
    returns = []
    symbols = cdx_df['source'].unique()
    
    for symbol in symbols:
        sym_data = cdx_df[cdx_df['source'] == symbol].sort_values('date')
        
        # Get buy price (closest to buy_date, before or on)
        buy_data = sym_data[sym_data['date'] <= buy_date]
        if buy_data.empty:
            continue
        buy_row = buy_data.iloc[-1]
        buy_price = buy_row.get(price_col, np.nan)
        buy_actual_date = buy_row['date']
        
        # Get eval price (closest to eval_date)
        eval_data = sym_data[sym_data['date'] <= eval_date]
        if eval_data.empty:
            continue
        eval_row = eval_data.iloc[-1]
        eval_price = eval_row.get(price_col, np.nan)
        eval_actual_date = eval_row['date']
        
        if pd.isna(buy_price) or pd.isna(eval_price) or buy_price <= 0:
            continue
        
        pct_return = (eval_price - buy_price) / buy_price
        
        returns.append({
            'symbol': symbol,
            'buy_date_target': buy_date,
            'buy_date_actual': buy_actual_date,
            'buy_price': buy_price,
            'eval_date_target': eval_date,
            'eval_date_actual': eval_actual_date,
            'eval_price': eval_price,
            'return_pct': pct_return * 100,
            'return_decimal': pct_return,
        })
    
    returns_df = pd.DataFrame(returns)
    
    print(f"\nStocks with valid price data: {len(returns_df):,}")
    
    # Summary stats
    print(f"\n{'='*70}")
    print("RETURN DISTRIBUTION SUMMARY")
    print('='*70)
    print(f"Mean return: {returns_df['return_pct'].mean():.1f}%")
    print(f"Median return: {returns_df['return_pct'].median():.1f}%")
    print(f"Std dev: {returns_df['return_pct'].std():.1f}%")
    print(f"Min return: {returns_df['return_pct'].min():.1f}%")
    print(f"Max return: {returns_df['return_pct'].max():.1f}%")
    
    # Percentiles
    print(f"\nPercentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]:
        val = returns_df['return_pct'].quantile(p/100)
        print(f"  {p}th: {val:.1f}%")
    
    # Investigate extremes
    print(f"\n{'='*70}")
    print(f"TOP {n_extreme} HIGHEST RETURNS (potential data issues)")
    print('='*70)
    
    top_returns = returns_df.nlargest(n_extreme, 'return_pct')
    
    for idx, row in top_returns.iterrows():
        symbol = row['symbol']
        print(f"\n{symbol}:")
        print(f"  Return: {row['return_pct']:,.1f}%")
        print(f"  Buy: ${row['buy_price']:.4f} on {row['buy_date_actual'].date()}")
        print(f"  Eval: ${row['eval_price']:.4f} on {row['eval_date_actual'].date()}")
        
        # Get full price history for this symbol
        sym_data = cdx_df[cdx_df['source'] == symbol][[price_col, 'date', 'marketCap']].sort_values('date')
        
        if len(sym_data) > 0:
            print(f"  Price history ({len(sym_data)} data points):")
            # Show first few and last few
            if len(sym_data) <= 6:
                for _, r in sym_data.iterrows():
                    mcap = r.get('marketCap', np.nan)
                    mcap_str = f"${mcap/1e9:.2f}B" if pd.notna(mcap) and mcap > 1e9 else f"${mcap/1e6:.1f}M" if pd.notna(mcap) else "N/A"
                    print(f"    {r['date'].date()}: ${r[price_col]:.4f} (MCap: {mcap_str})")
            else:
                for _, r in sym_data.head(3).iterrows():
                    mcap = r.get('marketCap', np.nan)
                    mcap_str = f"${mcap/1e9:.2f}B" if pd.notna(mcap) and mcap > 1e9 else f"${mcap/1e6:.1f}M" if pd.notna(mcap) else "N/A"
                    print(f"    {r['date'].date()}: ${r[price_col]:.4f} (MCap: {mcap_str})")
                print(f"    ... ({len(sym_data) - 6} more points) ...")
                for _, r in sym_data.tail(3).iterrows():
                    mcap = r.get('marketCap', np.nan)
                    mcap_str = f"${mcap/1e9:.2f}B" if pd.notna(mcap) and mcap > 1e9 else f"${mcap/1e6:.1f}M" if pd.notna(mcap) else "N/A"
                    print(f"    {r['date'].date()}: ${r[price_col]:.4f} (MCap: {mcap_str})")
    
    # Look for suspicious patterns
    print(f"\n{'='*70}")
    print("SUSPICIOUS PATTERNS")
    print('='*70)
    
    # Near-zero buy prices
    near_zero = returns_df[returns_df['buy_price'] < 0.10]
    print(f"\nStocks with buy price < $0.10: {len(near_zero)}")
    if len(near_zero) > 0:
        print("  Examples:", near_zero['symbol'].head(10).tolist())
    
    # Huge price jumps
    huge_returns = returns_df[returns_df['return_pct'] > 1000]
    print(f"\nStocks with return > 1000%: {len(huge_returns)}")
    if len(huge_returns) > 0:
        print("  Examples:", huge_returns['symbol'].head(20).tolist())
    
    # Preferred shares / odd classes
    preferreds = returns_df[returns_df['symbol'].str.contains(r'-[A-Z]$|\.PR|\.PF', regex=True, na=False)]
    print(f"\nPotential preferred shares/odd classes: {len(preferreds)}")
    if len(preferreds) > 0:
        print("  Examples:", preferreds['symbol'].head(10).tolist())
    
    # Check for currency issues (very low prices that might be in different currency)
    very_low = returns_df[(returns_df['buy_price'] < 1) & (returns_df['eval_price'] > 100)]
    print(f"\nPossible currency/unit mismatch (buy < $1, eval > $100): {len(very_low)}")
    if len(very_low) > 0:
        print("  Examples:", very_low['symbol'].head(10).tolist())
    
    return returns_df


def investigate_ols_failure(dmdic):
    """Check why OLS might be failing with 0 samples."""
    
    print(f"\n{'='*70}")
    print("OLS PIPELINE INVESTIGATION")
    print('='*70)
    
    BoMetric_df = dmdic['BoMetric_df'].copy()
    cdx_df = dmdic['cdx_df'].copy()
    
    BoMetric_df['date'] = pd.to_datetime(BoMetric_df['date'])
    cdx_df['date'] = pd.to_datetime(cdx_df['date'])
    
    print(f"\nBoMetric_df:")
    print(f"  Rows: {len(BoMetric_df):,}")
    print(f"  Columns: {len(BoMetric_df.columns)}")
    print(f"  Unique symbols: {BoMetric_df['source'].nunique():,}")
    print(f"  Date range: {BoMetric_df['date'].min().date()} to {BoMetric_df['date'].max().date()}")
    
    print(f"\ncdx_df:")
    print(f"  Rows: {len(cdx_df):,}")
    print(f"  Columns: {len(cdx_df.columns)}")
    print(f"  Unique symbols: {cdx_df['source'].nunique():,}")
    print(f"  Date range: {cdx_df['date'].min().date()} to {cdx_df['date'].max().date()}")
    
    # Check overlap
    bm_symbols = set(BoMetric_df['source'].unique())
    cdx_symbols = set(cdx_df['source'].unique())
    overlap = bm_symbols & cdx_symbols
    
    print(f"\nSymbol overlap:")
    print(f"  BoMetric symbols: {len(bm_symbols):,}")
    print(f"  cdx_df symbols: {len(cdx_symbols):,}")
    print(f"  Overlap: {len(overlap):,}")
    
    # Check price column
    price_col = 'price' if 'price' in cdx_df.columns else 'stockPrice'
    print(f"\nPrice column '{price_col}':")
    print(f"  Non-null: {cdx_df[price_col].notna().sum():,}")
    print(f"  Null: {cdx_df[price_col].isna().sum():,}")
    print(f"  Zero or negative: {(cdx_df[price_col] <= 0).sum():,}")
    
    # Check what happens at buy date
    max_date = cdx_df['date'].max()
    buy_year = max_date.year - 3
    buy_date = pd.Timestamp(f"{buy_year}-12-31")
    
    bm_at_buy = BoMetric_df[BoMetric_df['date'] <= buy_date]
    cdx_at_buy = cdx_df[cdx_df['date'] <= buy_date]
    
    print(f"\nAt buy date ({buy_date.date()}):")
    print(f"  BoMetric rows: {len(bm_at_buy):,}")
    print(f"  BoMetric symbols: {bm_at_buy['source'].nunique():,}")
    print(f"  cdx_df rows: {len(cdx_at_buy):,}")
    print(f"  cdx_df symbols: {cdx_at_buy['source'].nunique():,}")
    
    # Check if BoScore calculation would work
    print(f"\nBoMetric columns (first 15):")
    print(f"  {list(BoMetric_df.columns)[:15]}")
    
    # Check for NaN percentage in key columns
    print(f"\nNaN percentage in BoMetric numeric columns:")
    numeric_cols = BoMetric_df.select_dtypes(include=[np.number]).columns[:10]
    for col in numeric_cols:
        nan_pct = BoMetric_df[col].isna().mean() * 100
        print(f"  {col}: {nan_pct:.1f}%")


def check_ticker_details(dmdic, tickers):
    """Deep dive into specific tickers."""
    
    cdx_df = dmdic['cdx_df'].copy()
    cdx_df['date'] = pd.to_datetime(cdx_df['date'])
    
    price_col = 'price' if 'price' in cdx_df.columns else 'stockPrice'
    
    print(f"\n{'='*70}")
    print("TICKER DEEP DIVE")
    print('='*70)
    
    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        
        sym_data = cdx_df[cdx_df['source'] == ticker].sort_values('date')
        
        if sym_data.empty:
            print("  No data found!")
            continue
        
        print(f"  Data points: {len(sym_data)}")
        print(f"  Date range: {sym_data['date'].min().date()} to {sym_data['date'].max().date()}")
        
        # Show all available columns for first row
        print(f"  Available columns: {list(sym_data.columns)}")
        
        # Price history
        print(f"\n  Full price history:")
        cols_to_show = [c for c in ['date', price_col, 'marketCap', 'revenue', 'netIncome'] if c in sym_data.columns]
        display_df = sym_data[cols_to_show].copy()
        display_df['date'] = display_df['date'].dt.date
        
        for _, row in display_df.iterrows():
            parts = [f"{row['date']}"]
            if price_col in row:
                parts.append(f"price=${row[price_col]:.4f}")
            if 'marketCap' in row and pd.notna(row['marketCap']):
                mcap = row['marketCap']
                if mcap > 1e9:
                    parts.append(f"mcap=${mcap/1e9:.2f}B")
                else:
                    parts.append(f"mcap=${mcap/1e6:.1f}M")
            print(f"    {' | '.join(parts)}")


def main():
    print("="*70)
    print("BACKTEST DIAGNOSTIC REPORT")
    print("="*70)
    print(f"Generated: {datetime.now()}")
    
    dmdic = load_data()
    if dmdic is None:
        return
    
    # Investigate extreme returns
    returns_df = investigate_extreme_returns(dmdic, n_extreme=30)
    
    # Investigate OLS failure
    investigate_ols_failure(dmdic)
    
    # Deep dive into specific problematic tickers (if any identified)
    if returns_df is not None and len(returns_df) > 0:
        # Get top 5 extreme return tickers for deep dive
        extreme_tickers = returns_df.nlargest(5, 'return_pct')['symbol'].tolist()
        check_ticker_details(dmdic, extreme_tickers)
    
    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print('='*70)
    print("\nNext steps based on findings:")
    print("1. Review extreme return tickers for data quality issues")
    print("2. Check if price series are split/dividend adjusted")
    print("3. Verify currency consistency across tickers")
    print("4. Consider filtering out penny stocks / illiquid names")
    print("5. Check for look-ahead bias in fundamentals")


if __name__ == "__main__":
    main()
