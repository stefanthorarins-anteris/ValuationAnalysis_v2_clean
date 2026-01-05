"""
Test script for CycleHeat calculation
Run this BEFORE the full pipeline to verify:
1. Required columns exist in cdx_df
2. EPS calculation works
3. Profile API returns beta
4. CycleHeat formula produces sensible values
"""

import pandas as pd
import numpy as np
import requests

# Load config
api_key = open('fmpAPIkey.txt', 'r').read().strip()
baseurl = "https://financialmodelingprep.com/api/"

# Load existing pickle file
PICKLE_FILE = 'Bometric_dic-fmp_stock_NA1_EU1_all_2025-12-09_len9155_manelim3692_fails1601.pickle'

print("="*60)
print("CycleHeat Test Script")
print("="*60)

# Test 1: Load pickle and check columns
print("\n[TEST 1] Loading pickle and checking required columns...")
try:
    bm = pd.read_pickle(PICKLE_FILE)
    cdx_df = bm['cdx_df']
    print(f"  [OK] Loaded pickle: {len(cdx_df)} rows")
    
    required_cols = ['netIncome', 'weightedAverageShsOut', 'source', 'marketCap', 'freeCashFlow']
    missing = [c for c in required_cols if c not in cdx_df.columns]
    
    if missing:
        print(f"  [FAIL] MISSING columns: {missing}")
    else:
        print(f"  [OK] All required columns present: {required_cols}")
except Exception as e:
    print(f"  [FAIL] FAILED to load pickle: {e}")
    exit(1)

# Test 2: EPS calculation on sample tickers
print("\n[TEST 2] EPS calculation on sample tickers...")
sample_tickers = list(cdx_df['source'].unique()[:5])
print(f"  Testing tickers: {sample_tickers}")

for ticker in sample_tickers:
    tempcdx = cdx_df.loc[cdx_df['source'] == ticker]
    try:
        eps = tempcdx['netIncome'] / tempcdx['weightedAverageShsOut']
        eps_clean = eps.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(eps_clean) >= 2:
            eps_current = eps_clean.iloc[0]
            eps_mean = eps_clean.mean()
            
            if eps_mean > 0:
                R = eps_current / eps_mean
                R_excess = max(0.0, R - 1.0)
                print(f"  [OK] {ticker}: EPS_current={eps_current:.4f}, EPS_mean={eps_mean:.4f}, R={R:.2f}, R_excess={R_excess:.2f}")
            else:
                print(f"  [WARN] {ticker}: eps_mean <= 0 ({eps_mean:.4f}), would use R=1.0")
        else:
            print(f"  [WARN] {ticker}: Only {len(eps_clean)} valid EPS values (need >=2)")
    except Exception as e:
        print(f"  [FAIL] {ticker}: FAILED - {e}")

# Test 3: Profile API for beta (limited calls)
print("\n[TEST 3] Profile API for beta (3 test calls)...")
test_tickers = ['AAPL', 'MSFT', 'GOOGL']

for ticker in test_tickers:
    try:
        resp = requests.get(f'{baseurl}v3/profile/{ticker}?apikey={api_key}')
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                beta = data[0].get('beta')
                sector = data[0].get('sector', 'N/A')
                industry = data[0].get('industry', 'N/A')
                print(f"  [OK] {ticker}: beta={beta}, sector={sector}, industry={industry}")
            else:
                print(f"  [FAIL] {ticker}: Empty response")
        else:
            print(f"  [FAIL] {ticker}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"  [FAIL] {ticker}: FAILED - {e}")

# Test 4: Full CycleHeat calculation
print("\n[TEST 4] Full CycleHeat calculation on sample tickers...")

def calculate_cycle_heat(tempcdx, beta_stock):
    """Replicate the CycleHeat logic from postBoRank.py"""
    try:
        eps = tempcdx['netIncome'] / tempcdx['weightedAverageShsOut']
        eps_clean = eps.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(eps_clean) >= 2:
            eps_current = eps_clean.iloc[0]
            eps_mean = eps_clean.mean()
            
            if eps_mean > 0:
                R = eps_current / eps_mean
            else:
                R = 1.0
            
            R_excess = max(0.0, R - 1.0)
            cycle_heat = R_excess * (1.0 + max(0.0, beta_stock))
            cycle_heat = min(cycle_heat, 3.0)  # Cap
            return cycle_heat, R, R_excess
        return 0.0, None, None
    except:
        return 0.0, None, None

# Use tickers from pickle that also have profile data
for ticker in sample_tickers[:3]:
    tempcdx = cdx_df.loc[cdx_df['source'] == ticker]
    
    # Fetch beta
    try:
        resp = requests.get(f'{baseurl}v3/profile/{ticker}?apikey={api_key}')
        beta = 1.0  # default
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0 and data[0].get('beta') is not None:
                beta = float(data[0].get('beta'))
    except:
        beta = 1.0
    
    cycle_heat, R, R_excess = calculate_cycle_heat(tempcdx, beta)
    
    if R is not None:
        print(f"  [OK] {ticker}: beta={beta:.2f}, R={R:.2f}, R_excess={R_excess:.2f} -> CycleHeat={cycle_heat:.3f}")
    else:
        print(f"  [WARN] {ticker}: CycleHeat={cycle_heat:.3f} (insufficient EPS data)")

# Test 5: V4 endpoint check (optional features)
print("\n[TEST 5] V4 endpoints (optional - may require higher subscription)...")
v4_endpoints = [
    ('stock_peers', f'{baseurl}v4/stock_peers?symbol=AAPL&apikey={api_key}'),
    ('sector_price_earning_ratio', f'https://financialmodelingprep.com/api/v4/sector_price_earning_ratio?date=2025-01-05&exchange=NYSE&apikey={api_key}'),
]

for name, url in v4_endpoints:
    try:
        resp = requests.get(url)
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            print(f"  [OK] {name}: Working (got {len(data)} items)")
        elif isinstance(data, dict) and 'Error Message' in str(data):
            print(f"  [WARN] {name}: Requires higher subscription tier")
        else:
            print(f"  [WARN] {name}: Empty response (may need higher tier)")
    except Exception as e:
        print(f"  [FAIL] {name}: FAILED - {e}")

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("""
If Tests 1-4 pass: CycleHeat should work in the pipeline.
If Test 5 shows warnings: Excel presentation will have missing data but won't crash.

To run: python test_cycleheat.py
""")

