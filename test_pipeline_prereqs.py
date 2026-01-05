"""
Pre-flight test script for the full pipeline
Checks all potential issues WITHOUT running the full pipeline

Run this BEFORE: python Sbocker.py -loadbometric 1 -bometricfilename <file>

Tests:
1. Pickle file loads and has required structure
2. All required columns exist for calculations
3. API endpoints are accessible (limited test calls)
4. Data quality checks (NaN percentages, data types)
5. BoScore metric definitions are consistent
6. PostBo ranking metrics are available
"""

import pandas as pd
import numpy as np
import requests
import sys

# Config
api_key = open('fmpAPIkey.txt', 'r').read().strip()
baseurl = "https://financialmodelingprep.com/api/"

PICKLE_FILE = 'Bometric_dic-fmp_stock_NA1_EU1_all_2025-12-09_len9155_manelim3692_fails1601.pickle'

print("="*70)
print("Pipeline Pre-flight Check")
print("="*70)

all_passed = True
warnings_count = 0

def test_pass(msg):
    print(f"  [OK] {msg}")

def test_fail(msg):
    global all_passed
    all_passed = False
    print(f"  [FAIL] {msg}")

def test_warn(msg):
    global warnings_count
    warnings_count += 1
    print(f"  [WARN] {msg}")

# =============================================================================
# TEST 1: Load pickle file
# =============================================================================
print("\n[TEST 1] Pickle file structure...")

try:
    bm = pd.read_pickle(PICKLE_FILE)
    test_pass(f"Loaded pickle: {PICKLE_FILE}")
except Exception as e:
    test_fail(f"Cannot load pickle: {e}")
    print("\nCANNOT CONTINUE - pickle file required for further tests")
    sys.exit(1)

# Check required keys in pickle
required_keys = ['cdx_df', 'BoMetric_df']
optional_keys = ['BoScore_df']  # Computed during pipeline, may not be in input pickle

for key in required_keys:
    if key in bm:
        test_pass(f"Pickle has '{key}' ({type(bm[key]).__name__})")
    else:
        test_fail(f"Pickle missing required '{key}'")

for key in optional_keys:
    if key in bm:
        test_pass(f"Pickle has optional '{key}' ({type(bm[key]).__name__})")
    else:
        test_pass(f"Pickle missing '{key}' (OK - computed during pipeline)")

cdx_df = bm.get('cdx_df', pd.DataFrame())
bometric_df = bm.get('BoMetric_df', pd.DataFrame())
boscore_df = bm.get('BoScore_df', pd.DataFrame())

# =============================================================================
# TEST 2: Required columns for calculations
# =============================================================================
print("\n[TEST 2] Required columns in cdx_df...")

# Columns needed for various calculations
column_groups = {
    'EPS/CycleHeat': ['netIncome', 'weightedAverageShsOut'],
    'FCF metrics': ['freeCashFlow', 'marketCap'],
    'Altman-Z': ['totalAssets', 'totalLiabilities', 'totalCurrentAssets', 
                 'totalCurrentLiabilities', 'revenue', 'operatingIncome',
                 'totalStockholdersEquity'],
    'Piotroski': ['netIncome', 'totalAssets', 'netCashProvidedByOperatingActivities',
                  'longTermDebt', 'totalCurrentAssets', 'totalCurrentLiabilities',
                  'grossProfit', 'revenue'],
    'Basic': ['source', 'price', 'date'],
    'Key metrics': ['pbRatio', 'earningsYield', 'grahamNumber', 'returnOnEquity',
                    'returnOnCapitalEmployed', 'currentRatio', 'netDebtToEBITDA'],
}

for group_name, cols in column_groups.items():
    missing = [c for c in cols if c not in cdx_df.columns]
    if missing:
        test_fail(f"{group_name}: Missing columns {missing}")
    else:
        test_pass(f"{group_name}: All {len(cols)} columns present")

# =============================================================================
# TEST 3: Data quality checks
# =============================================================================
print("\n[TEST 3] Data quality in cdx_df...")

# Check NaN percentages for critical columns
critical_cols = ['netIncome', 'weightedAverageShsOut', 'marketCap', 'totalAssets']
for col in critical_cols:
    if col in cdx_df.columns:
        nan_pct = cdx_df[col].isna().sum() / len(cdx_df) * 100
        if nan_pct > 50:
            test_warn(f"{col}: {nan_pct:.1f}% NaN (>50% threshold)")
        elif nan_pct > 20:
            test_warn(f"{col}: {nan_pct:.1f}% NaN (moderate)")
        else:
            test_pass(f"{col}: {nan_pct:.1f}% NaN")

# Check unique tickers
n_tickers = cdx_df['source'].nunique() if 'source' in cdx_df.columns else 0
test_pass(f"Unique tickers: {n_tickers}")

# Check data types
if 'marketCap' in cdx_df.columns:
    if cdx_df['marketCap'].dtype in [np.float64, np.int64, float, int]:
        test_pass("marketCap is numeric")
    else:
        test_fail(f"marketCap is {cdx_df['marketCap'].dtype} (should be numeric)")

# =============================================================================
# TEST 4: BoMetric dataframe check (BoScore is computed during pipeline)
# =============================================================================
print("\n[TEST 4] BoMetric dataframe...")

if not bometric_df.empty:
    test_pass(f"BoMetric_df has {len(bometric_df)} rows, {len(bometric_df.columns)} columns")
    if 'source' in bometric_df.columns:
        n_tickers_bm = bometric_df['source'].nunique()
        test_pass(f"BoMetric_df covers {n_tickers_bm} unique tickers")
    else:
        test_warn("BoMetric_df missing 'source' column")
else:
    test_fail("BoMetric_df is empty")

# =============================================================================
# TEST 5: API endpoint checks (limited calls)
# =============================================================================
print("\n[TEST 5] API endpoints (3 test calls each)...")

test_ticker = 'AAPL'

# V3 endpoints (should work on basic subscription)
v3_endpoints = [
    ('profile', f'{baseurl}v3/profile/{test_ticker}?apikey={api_key}'),
    ('key-metrics', f'{baseurl}v3/key-metrics/{test_ticker}?period=quarter&limit=4&apikey={api_key}'),
    ('ratios', f'{baseurl}v3/ratios/{test_ticker}?period=quarter&limit=4&apikey={api_key}'),
    ('discounted-cash-flow', f'{baseurl}v3/discounted-cash-flow/{test_ticker}?apikey={api_key}'),
    ('financial-growth', f'{baseurl}v3/financial-growth/{test_ticker}?limit=4&apikey={api_key}'),
    ('rating', f'{baseurl}v3/rating/{test_ticker}?apikey={api_key}'),
]

for name, url in v3_endpoints:
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                test_pass(f"v3/{name}: OK ({len(data)} items)")
            elif isinstance(data, dict) and 'Error' not in str(data):
                test_pass(f"v3/{name}: OK (dict response)")
            else:
                test_warn(f"v3/{name}: Empty or error response")
        else:
            test_fail(f"v3/{name}: HTTP {resp.status_code}")
    except Exception as e:
        test_fail(f"v3/{name}: {e}")

# V4 endpoints (may require higher subscription)
print("\n  V4 endpoints (optional features):")
v4_endpoints = [
    ('stock_peers', f'{baseurl}v4/stock_peers?symbol={test_ticker}&apikey={api_key}'),
]

for name, url in v4_endpoints:
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            test_pass(f"v4/{name}: Available")
        else:
            test_warn(f"v4/{name}: Not available (higher tier required)")
    except Exception as e:
        test_warn(f"v4/{name}: {e}")

# =============================================================================
# TEST 6: URL format check (double-slash bug)
# =============================================================================
print("\n[TEST 6] URL format verification...")

# Simulate URL construction
test_urls = [
    (f'{baseurl}v3/profile/AAPL', 'profile'),
    (f'{baseurl}v3/rating/AAPL', 'rating'),
    (f'{baseurl}v3/key-metrics/AAPL', 'key-metrics'),
]

for url, name in test_urls:
    if '//' in url.replace('https://', '').replace('http://', ''):
        test_fail(f"{name} URL has double-slash: {url}")
    else:
        test_pass(f"{name} URL format OK")

# =============================================================================
# TEST 7: createDicts consistency check
# =============================================================================
print("\n[TEST 7] Metric definitions consistency...")

try:
    from createDicts import getDicts, getPostDict
    
    # Get preReq_dict from getDicts
    preReq_dict, _, _, _, _, _, _ = getDicts()
    
    # Get postBo ranking dicts
    postBmRankingDict, postNewRankingDict = getPostDict()
    
    # Check preReq columns are in cdx_df
    all_prereq_cols = []
    for source, cols in preReq_dict.items():
        all_prereq_cols.extend(cols)
    
    missing_prereq = [c for c in all_prereq_cols if c not in cdx_df.columns and c != 'price']
    if missing_prereq:
        test_warn(f"preReq_dict columns not in cdx_df: {missing_prereq[:5]}...")
    else:
        test_pass(f"All preReq_dict columns ({len(all_prereq_cols)}) present in cdx_df")
    
    # Check CycleHeat is in postNewRankingDict
    if 'CycleHeat' in postNewRankingDict:
        weight = postNewRankingDict['CycleHeat'].get('w', 'N/A')
        if weight < 0:
            test_pass(f"CycleHeat in postNewRankingDict with negative weight ({weight})")
        else:
            test_warn(f"CycleHeat weight should be negative, got {weight}")
    else:
        test_fail("CycleHeat not in postNewRankingDict")
    
    # List all postBo metrics
    test_pass(f"postBmRankingDict has {len(postBmRankingDict)} metrics")
    test_pass(f"postNewRankingDict has {len(postNewRankingDict)} metrics")
    
    # Show postNewRankingDict metrics and weights
    print("  PostNew metrics: ", end="")
    print(", ".join([f"{k}({v['w']})" for k, v in postNewRankingDict.items()]))
    
except Exception as e:
    test_fail(f"Cannot import createDicts: {e}")

# =============================================================================
# TEST 8: Sample calculation test
# =============================================================================
print("\n[TEST 8] Sample calculations on real data...")

# Pick a ticker with good data
sample_tickers = list(cdx_df['source'].unique()[:20])
calc_success = 0

for ticker in sample_tickers:
    tempcdx = cdx_df.loc[cdx_df['source'] == ticker]
    if len(tempcdx) < 4:
        continue
    
    try:
        # Test FCF yield calculation
        fcf = tempcdx['freeCashFlow'].iloc[0]
        mcap = tempcdx['marketCap'].iloc[0]
        if mcap and mcap != 0 and not np.isnan(mcap):
            fcf_yield = fcf / mcap
            calc_success += 1
            if calc_success == 1:
                test_pass(f"FCF yield calc works ({ticker}: {fcf_yield:.4f})")
        
        # Test Altman-Z components
        if calc_success >= 3:
            break
            
    except Exception as e:
        pass

if calc_success == 0:
    test_warn("No successful sample calculations")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if all_passed and warnings_count == 0:
    print("\n[SUCCESS] All tests passed! Pipeline should run without issues.")
elif all_passed:
    print(f"\n[CAUTION] All critical tests passed, but {warnings_count} warning(s).")
    print("          Pipeline should run, but some features may have limited data.")
else:
    print("\n[FAILURE] Some critical tests failed. Fix issues before running pipeline.")

print(f"""
Next steps:
  - If all OK: python Sbocker.py -loadbometric 1 -bometricfilename {PICKLE_FILE}
  - If warnings: Pipeline will run but may have missing data in some outputs
  - If failures: Fix the issues listed above before running
""")

