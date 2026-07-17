#!/usr/bin/env python3
r"""
generate_presentation.py -- Standalone HTML presentation generator for investment-filter runs.

Turns a saved pipeline run into one self-contained HTML presentation for manual post-filter review.
Reads saved artifacts only (read-only on all pipeline inputs). Re-runnable in minutes on any run dir.

CLI:
    python generate_presentation.py --run-dir <dir> [--out <path>] [--augment on|off]

Defaults:
    --run-dir: HomeGDrive (where pulled runs land)
    --augment: off (offline-safe verify; 'on' fetches from FMP gracefully with gap-fallback)

Output:
    presentation_<run_date>.html (fully self-contained: inline CSS, inline SVG, no external refs)
"""

import os
import sys
import glob
import json
import time
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from html import escape
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================
VALUATION_REPO = Path(r"C:\Users\stefanthorarinsson\Documents\Projects\ValuationAnalysis_v2_clean")
DEFAULT_RUN_DIR = Path(r"C:\Users\stefanthorarinsson\Documents\HomeGDrive")

# Persistent, TRACKED (committed) per-ticker Yahoo info store. No time expiry: an entry,
# once fetched, is reused forever unless --refresh-yahoo forces a re-fetch. Travels with
# the repo so it accumulates across runs/machines and a fresh clone never re-fetches all.
YAHOO_STORE_PATH = VALUATION_REPO / "yahoo_info.json"
YAHOO_FIELDS = ['longName', 'longBusinessSummary', 'sector', 'industry', 'website',
                'fullTimeEmployees', 'city', 'country', 'marketCap']
YAHOO_FETCH_SPACING_S = 0.6

PLAYBOOK_METRICS = [
    'returnOnCapitalEmployed', 'returnOnEquity', 'RoA', 'grossProfitMargin',
    'freeCashFlowYield', 'currentRatio', 'earnYield', 'revenueGrowth',
    'incomeQuality', 'Altman-Z', 'Piotroski', 'bVpRatio', 'tbVpRatio',
    'freeCashFlowPerShareGrowth', 'moatScore', 'CycleHeat',
]

COHORTS = ['REIT', 'Mining', 'InvestmentVehicle', 'FinManager', 'BalanceSheetFin']

# Section-G / distribution-bar metric mapping.
#   label       : display label
#   cdx_col     : column in cdx_df supplying the RAW latest marker value; None means the
#                 metric has no native cdx column (compute it from the reviewReference raw
#                 pool instead -- e.g. freeCashFlowYield, which is a TTM ratio, not a
#                 per-quarter cdx column).
#   pool_metric : the PLAYBOOK_METRICS column name used for the cohort distribution/percentile
#                 (note the cdx column and the pool column can differ: cdx 'returnOnAssets'
#                 vs pool 'RoA').
#   fmt         : 'ratio' or 'pct' for the raw-value display.
# All 16 PLAYBOOK_METRICS, each rendered as raw value + dot-on-p10-p90 bar + percentile
# against the name's OWN cohort pool. Marker rule (RAW, never a z-score): use the cdx_df
# latest value when the metric has a native cdx column; otherwise the reviewReference raw
# pool value (freeCashFlowYield / revenueGrowth / Altman-Z / Piotroski / bVpRatio /
# tbVpRatio / freeCashFlowPerShareGrowth / moatScore / CycleHeat have no clean cdx column).
# The first six are kept first and unchanged so their confirmed values do not move.
SECTION_G_METRICS = [
    ('ROIC',                'returnOnCapitalEmployed', 'returnOnCapitalEmployed',    'ratio'),
    ('ROE',                 'returnOnEquity',          'returnOnEquity',             'ratio'),
    ('ROA',                 'returnOnAssets',          'RoA',                        'ratio'),
    ('Gross Margin',        'grossProfitMargin',       'grossProfitMargin',          'pct'),
    ('FCF Yield',           None,                      'freeCashFlowYield',          'pct'),
    ('Current Ratio',       'currentRatio',            'currentRatio',               'ratio'),
    ('Earnings Yield',      'earningsYield',           'earnYield',                  'pct'),
    ('Revenue Growth',      None,                      'revenueGrowth',              'pct'),
    ('Income Quality',      'incomeQuality',           'incomeQuality',              'ratio'),
    ('Altman-Z',            None,                      'Altman-Z',                   'ratio'),
    ('Piotroski',           None,                      'Piotroski',                  'ratio'),
    ('Book/Price (B/P)',    None,                      'bVpRatio',                   'ratio'),
    ('Tangible Book/Price', None,                      'tbVpRatio',                  'ratio'),
    ('FCF/Share Growth',    None,                      'freeCashFlowPerShareGrowth', 'pct'),
    ('Moat Score',          None,                      'moatScore',                  'ratio'),
    ('CycleHeat',           None,                      'CycleHeat',                  'ratio'),
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def find_latest_pickle(run_dir, pattern):
    """Find the latest pickle matching the glob pattern in run_dir."""
    files = glob.glob(os.path.join(run_dir, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_pickle(filepath):
    """Load and return a pickle file."""
    if not os.path.exists(filepath):
        log.warning(f"Pickle not found: {filepath}")
        return None
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        log.error(f"Failed to load pickle {filepath}: {e}")
        return None


def safe_get(d, key, default=None):
    """Safely get a nested key from a dict."""
    if d is None:
        return default
    return d.get(key, default)


def safe_float(v):
    """Convert to float, returning NaN for None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


def pct_format(v):
    """Format a fraction as a percentage string (multiply by 100). For 0-1 ratios only."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:.1f}%"


def pctile_format(v):
    """Format a percentile that is ALREADY on a 0-100 scale (NO x100). Used for cohort
    percentiles, which must never pass through the pct_format ratio formatter."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.1f}"


def ratio_format(v, decimals=2):
    """Format a float as a ratio."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def ttm_sum(df, col):
    """Compute trailing-12-month sum from quarterly data (NEWEST 4 rows = head(4) after newest-first sort)."""
    if df is None or df.empty or col not in df.columns:
        return np.nan
    values = df[col].dropna().head(4).values
    if len(values) == 0:
        return np.nan
    return float(np.sum(values))


def latest_row_value(df, col):
    """Get value from latest (first) row in sorted quarterly data."""
    if df is None or df.empty or col not in df.columns:
        return np.nan
    vals = df[col].dropna()
    if vals.empty:
        return np.nan
    return float(vals.iloc[0])


def compute_operating_margin(df):
    """Compute operating margin from quarterly data (latest row)."""
    if df is None or df.empty:
        return np.nan
    rev = latest_row_value(df, 'revenue')
    op_inc = latest_row_value(df, 'operatingIncome')
    if np.isnan(rev) or np.isnan(op_inc) or rev == 0:
        return np.nan
    return op_inc / rev


def compute_fcf_margin_ttm(df):
    """Compute TTM FCF margin (TTM FCF / TTM revenue)."""
    if df is None or df.empty:
        return np.nan
    fcf_ttm = ttm_sum(df, 'freeCashFlow')
    rev_ttm = ttm_sum(df, 'revenue')
    if np.isnan(fcf_ttm) or np.isnan(rev_ttm) or rev_ttm == 0:
        return np.nan
    return fcf_ttm / rev_ttm


def compute_cash_conversion(df):
    """Compute cash conversion (TTM FCF / TTM netIncome)."""
    if df is None or df.empty:
        return np.nan
    fcf_ttm = ttm_sum(df, 'freeCashFlow')
    ni_ttm = ttm_sum(df, 'netIncome')
    if np.isnan(fcf_ttm) or np.isnan(ni_ttm) or ni_ttm == 0:
        return np.nan
    return fcf_ttm / ni_ttm


def compute_interest_coverage(df):
    """Compute interest coverage (operatingIncome / interestExpense)."""
    if df is None or df.empty:
        return np.nan
    op_inc = latest_row_value(df, 'operatingIncome')
    int_exp = latest_row_value(df, 'interestExpense')
    if np.isnan(op_inc) or np.isnan(int_exp) or int_exp == 0:
        return np.nan
    return op_inc / int_exp


def compute_ffo_per_share(df):
    """REIT FFO per share (proxy): (netIncome + D&A) / weightedAverageShsOut (TTM)."""
    if df is None or df.empty:
        return np.nan
    ni_ttm = ttm_sum(df, 'netIncome')
    da_ttm = ttm_sum(df, 'depreciationAndAmortization')
    shares = latest_row_value(df, 'weightedAverageShsOut')
    if np.isnan(ni_ttm) or np.isnan(da_ttm) or np.isnan(shares) or shares == 0:
        return np.nan
    return (ni_ttm + da_ttm) / shares


def compute_ffo_yield(mktcap, ffo_per_share):
    """Compute FFO yield (marketCap / TTM FFO)."""
    if mktcap is None or ffo_per_share is None:
        return np.nan
    mktcap = safe_float(mktcap)
    ffo = safe_float(ffo_per_share)
    if np.isnan(mktcap) or np.isnan(ffo) or ffo == 0 or mktcap == 0:
        return np.nan
    # FFO total = ffo_per_share * shares, but we'll use marketCap / FFO directly
    # This is approximate; ideally would need share count
    return mktcap / (ffo_per_share * 1e6)  # Rough approximation


def compute_ltv_proxy(df):
    """REIT LTV proxy (longTermDebt / totalAssets)."""
    if df is None or df.empty:
        return np.nan
    debt = latest_row_value(df, 'longTermDebt')
    assets = latest_row_value(df, 'totalAssets')
    if np.isnan(debt) or np.isnan(assets) or assets == 0:
        return np.nan
    return debt / assets


def get_percentile_marker(value, dist):
    """Get percentile of value within distribution."""
    if value is None or np.isnan(value):
        return np.nan
    if isinstance(dist, (list, np.ndarray)):
        dist = np.asarray(dist, dtype='float64')
        dist = dist[~np.isnan(dist)]
        if len(dist) == 0:
            return np.nan
        strict = np.count_nonzero(dist < value)
        weak = np.count_nonzero(dist <= value)
        return 100.0 * (strict + weak) / (2.0 * len(dist))
    return np.nan


# ============================================================================
# YAHOO FINANCE AUGMENT (persistent per-ticker store; no time expiry)
# ============================================================================
def _fetch_one_yahoo(yf, ticker):
    """Fetch one ticker's Yahoo info. NEVER raises -- on any failure or an unresolved
    ticker (empty info) it returns an entry carrying an 'error' string so the caller can
    gap-tag. yfinance needs no API key."""
    entry = {'fetched_date': datetime.now().strftime('%Y-%m-%d')}
    try:
        info = yf.Ticker(ticker).info or {}
        got = {k: info.get(k) for k in YAHOO_FIELDS}
        if not any(v not in (None, '', []) for v in got.values()):
            entry['error'] = 'no data (ticker unresolved on Yahoo)'
        else:
            entry.update(got)
            entry['error'] = None
    except Exception as e:
        entry['error'] = f"{type(e).__name__}: {str(e)[:180]}"
    return entry


def load_yahoo_data(tickers, augment, store_path=None, refresh=None):
    """Return {ticker: {yahoo fields + fetched_date + error}} from the persistent store,
    fetching ONLY the tickers that need it.

    tickers   : the page names (fetch order).
    augment   : if False, never hit the network -- use whatever is already stored, gap the
                rest (offline-safe).
    refresh   : None            -> fetch only tickers absent from the store (default);
                True            -> force re-fetch ALL requested tickers;
                set/list of tk  -> force re-fetch just those (plus any missing).
    store_path: JSON store location (default YAHOO_STORE_PATH, tracked/committed).
    """
    store_path = str(store_path or YAHOO_STORE_PATH)
    store = {}
    if os.path.exists(store_path):
        try:
            with open(store_path, 'r', encoding='utf-8') as f:
                store = json.load(f)
        except Exception as e:
            log.warning(f"Yahoo store unreadable ({e}); starting a fresh store.")
            store = {}

    # Decide what to fetch.
    if refresh is True:
        to_fetch = list(tickers)
    else:
        forced = set(refresh) if isinstance(refresh, (set, list, tuple)) else set()
        to_fetch = [t for t in tickers if t not in store or t in forced]

    if not augment:
        if to_fetch:
            log.info(f"Yahoo augment OFF: skipping {len(to_fetch)} fetch(es); "
                     f"using {len(tickers) - len(to_fetch)} stored, gap-tagging the rest.")
        return store

    if not to_fetch:
        log.info(f"Yahoo: 0 fetched (all {len(tickers)} page names already in store).")
        return store

    yf = None
    try:
        import yfinance as yf_mod
        yf = yf_mod
    except Exception as e:
        log.warning(f"yfinance import failed ({e}); Yahoo augment unavailable -> gap tags.")
        return store

    fetched = 0
    for t in to_fetch:
        if fetched > 0:
            time.sleep(YAHOO_FETCH_SPACING_S)
        store[t] = _fetch_one_yahoo(yf, t)
        fetched += 1
        tag = 'ok' if not store[t].get('error') else f"gap ({store[t]['error']})"
        log.info(f"  Yahoo fetch {fetched}/{len(to_fetch)}: {t} -> {tag}")

    try:
        os.makedirs(os.path.dirname(store_path) or '.', exist_ok=True)
        with open(store_path, 'w', encoding='utf-8') as f:
            json.dump(store, f, indent=2, ensure_ascii=False, sort_keys=True)
        log.info(f"Yahoo store written: {store_path} ({len(store)} tickers, "
                 f"{fetched} newly fetched this run).")
    except Exception as e:
        log.warning(f"Failed to write Yahoo store: {e}")

    return store


# ============================================================================
# SVG CHART GENERATION
# ============================================================================
def create_sparkline_svg(values, width=200, height=30, label=""):
    """Create a simple SVG sparkline from a list of values."""
    if not values or all(np.isnan(v) if isinstance(v, float) else v is None for v in values):
        return f'<span style="color:#999;">[no data]</span>'

    valid = [v for v in values if not (v is None or (isinstance(v, float) and np.isnan(v)))]
    if not valid:
        return f'<span style="color:#999;">[no data]</span>'

    vmin = min(valid)
    vmax = max(valid)
    if vmin == vmax:
        vmax = vmin + 1

    # Normalize to 0-1
    norm_vals = [(v - vmin) / (vmax - vmin) for v in values]

    # Generate points
    margin = 2
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    points = []
    for i, nv in enumerate(norm_vals):
        x = margin + i * plot_width / (len(norm_vals) - 1) if len(norm_vals) > 1 else margin + plot_width / 2
        y = height - margin - nv * plot_height
        points.append(f"{x},{y}")

    polyline = " ".join(points)
    svg = f'''<svg width="{width}" height="{height}" style="border:1px solid #ddd;display:inline-block;">
      <polyline points="{polyline}" fill="none" stroke="#0066cc" stroke-width="1.5"/>
    </svg>'''
    return svg


def create_distribution_bar(value, p10, p50, p90, width=150, height=15):
    """Create a distribution bar showing value against p10-p50-p90 spread."""
    if np.isnan(p10) or np.isnan(p50) or np.isnan(p90) or p10 == p90:
        return '<span style="color:#999;">—</span>'

    # Normalize positions
    val_norm = (value - p10) / (p90 - p10) if p90 > p10 else 0.5
    val_norm = max(0, min(1, val_norm))

    p50_norm = (p50 - p10) / (p90 - p10) if p90 > p10 else 0.5

    marker_x = width * val_norm
    p50_x = width * p50_norm

    svg = f'''<svg width="{width + 20}" height="{height + 4}" style="display:inline-block;">
      <rect x="2" y="2" width="{width}" height="{height}" fill="#f0f0f0" stroke="#ccc" stroke-width="0.5"/>
      <line x1="{2 + p50_x}" y1="1" x2="{2 + p50_x}" y2="{height + 3}" stroke="#999" stroke-width="1" stroke-dasharray="2,2"/>
      <circle cx="{2 + marker_x}" cy="{2 + height/2}" r="3" fill="#0066cc"/>
    </svg>'''
    return svg


# ============================================================================
# DATA LOADING & PROCESSING
# ============================================================================
def load_run_data(run_dir, valuation_repo):
    """Load all run data from pickles and CSVs."""
    run_dir = Path(run_dir)

    log.info(f"Loading run artifacts from {run_dir}...")

    # Find latest files by glob
    postrank_file = find_latest_pickle(str(run_dir), "postRank_*.pickle")
    boresults_file = find_latest_pickle(str(run_dir), "Boresults_dic-*.pickle")
    aggscore_file = find_latest_pickle(str(run_dir), "AggScoreTop100-*.csv")
    forensic_file = find_latest_pickle(str(run_dir), "ForensicFlagsTop100-*.csv")

    if not postrank_file:
        raise FileNotFoundError("No postRank_*.pickle found")
    if not boresults_file:
        raise FileNotFoundError("No Boresults_dic-*.pickle found")

    # Extract date from filename
    run_date = postrank_file.split('_')[1] if '_' in postrank_file else datetime.now().strftime('%Y-%m-%d')

    log.info(f"Run date: {run_date}")

    # Load pickles
    postrank_dic = load_pickle(postrank_file)
    boresults_dic = load_pickle(boresults_file)

    if not postrank_dic or not boresults_dic:
        raise ValueError("Failed to load core pickles")

    postrank_df = postrank_dic.get('postRank')
    cdx_df = postrank_dic.get('cdx_df')
    moatdf = postrank_dic.get('moatdf')
    tickers_df = boresults_dic.get('Tickers_df')
    carveout_sidelists = boresults_dic.get('carveout_sidelists')

    # Load CSVs
    aggscore_df = None
    forensic_df = None
    if aggscore_file and os.path.exists(aggscore_file):
        aggscore_df = pd.read_csv(aggscore_file)
    if forensic_file and os.path.exists(forensic_file):
        forensic_df = pd.read_csv(forensic_file)

    # Load industry dict
    industrydic_file = find_latest_pickle(str(valuation_repo), "industrydic_*.pickle")
    industrydic = load_pickle(industrydic_file) if industrydic_file else {}

    # Prepare industry lookup
    def get_industry(ticker):
        if isinstance(industrydic, dict):
            return industrydic.get(ticker, "Unknown")
        return "Unknown"

    return {
        'run_date': run_date,
        'postrank_df': postrank_df,
        'cdx_df': cdx_df,
        'moatdf': moatdf,
        'tickers_df': tickers_df,
        'carveout_sidelists': carveout_sidelists,
        'aggscore_df': aggscore_df,
        'forensic_df': forensic_df,
        'get_industry': get_industry,
    }


# ============================================================================
# COHORT PERCENTILE COMPUTATION (from reviewReference)
# ============================================================================
def _empty_bundle():
    return {'percentiles': {}, 'markers': {}, 'stats': {}, 'raw_all': None}


def compute_cohort_percentiles(data):
    """Build the cohort-percentile bundle used by Sections C/G/H.

    Returns a dict:
      'percentiles' : {(ticker, cohort): {pool_metric: percentile_0_100}}
      'markers'     : {ticker: {pool_metric: raw_marker_value}}   (cohort-independent)
      'stats'       : {(cohort, pool_metric): (p10, p50, p90)}
      'raw_all'     : DataFrame (source-indexed) of reviewReference raw playbook metrics
                      for every page name -- source of raw Altman-Z, FCF yield, etc.

    Contract (verified against confirmed numbers RAVE ROE pct=55.6, MRD.TO ROE pct=52.4):
      The MARKER is the RAW latest cdx_df value (or, for metrics with no native cdx column
      such as freeCashFlowYield, the reviewReference raw-pool value) -- NOT the postRank
      z-score. The percentile is that raw marker's position within its OWN cohort's raw
      distribution (the same pool reviewReference builds). General names -> general pool;
      carve names -> their carve cohort's pool.
    """
    try:
        sys.path.insert(0, str(VALUATION_REPO))
        import reviewReference as rr

        postrank_df = data['postrank_df']
        cdx_df = data['cdx_df']
        moatdf = data.get('moatdf')
        carveout = data.get('carveout_sidelists', {})

        # Full cohort membership (every ranked peer -> the true benchmark distribution).
        membership = {'general': list(postrank_df['source'].unique())}
        for cohort_label in COHORTS:
            cohort_dic = carveout.get(cohort_label, {})
            postrank_cohort = cohort_dic.get('postRank', pd.DataFrame())
            if postrank_cohort is not None and not postrank_cohort.empty:
                membership[cohort_label] = list(postrank_cohort['source'].unique())

        # Raw playbook-metric pools per cohort (from cdx_df, moatScore merged).
        dist_pools = rr.full_membership_pools(membership, cdx_df, moatdf=moatdf)

        # Cohort p10/p50/p90 lookup for the distribution bars.
        stats_long = rr.cohort_stats_long(dist_pools)
        stats = {}
        for _, r in stats_long.iterrows():
            stats[(r['cohort'], r['metric'])] = (
                float(r['p10']), float(r['p50']), float(r['p90']))

        # Per-ticker raw metrics for EVERY page name (Altman-Z, FCF yield, RoA, ...).
        # These reductions are per-ticker (pool-independent), so one frame covers all.
        all_names = set(membership['general'])
        for cohort_label in COHORTS:
            all_names |= set(membership.get(cohort_label, []))
        raw_all = rr._pool_raw_fast(list(all_names), cdx_df, nq=16)
        raw_all = rr._merge_moat({'_': raw_all}, moatdf)['_']
        if 'source' in raw_all.columns:
            raw_all = raw_all.set_index('source')

        # Newest-first cdx per ticker, for RAW latest markers.
        cdx_sorted = cdx_df.copy()
        cdx_sorted['date'] = pd.to_datetime(cdx_sorted['date'], errors='coerce')
        cdx_sorted = cdx_sorted.sort_values('date', ascending=False)
        cdx_by_ticker = {t: g for t, g in cdx_sorted.groupby('source', sort=False)}

        def latest_marker(ticker, cdx_col, pool_metric):
            """Raw marker: cdx latest if the metric has a native cdx column, else the
            reviewReference raw-pool value (e.g. freeCashFlowYield)."""
            if cdx_col is not None:
                g = cdx_by_ticker.get(ticker)
                if g is not None and cdx_col in g.columns:
                    v = g[cdx_col].dropna()
                    if len(v):
                        return float(v.iloc[0])
                return np.nan
            if ticker in raw_all.index and pool_metric in raw_all.columns:
                return safe_float(raw_all.loc[ticker, pool_metric])
            return np.nan

        markers = {}
        percentiles = {}
        # Compute markers once per ticker; percentiles per (ticker, cohort context).
        for cohort_label, members in membership.items():
            pool = dist_pools.get(cohort_label)
            for ticker in members:
                tm = markers.setdefault(ticker, {})
                pk = (ticker, cohort_label)
                percentiles[pk] = {}
                for _label, cdx_col, pool_metric, _fmt in SECTION_G_METRICS:
                    if pool_metric not in tm:
                        tm[pool_metric] = latest_marker(ticker, cdx_col, pool_metric)
                    marker = tm[pool_metric]
                    if pool is not None and pool_metric in pool.columns:
                        dist = rr._finite_series(pool[pool_metric]).dropna().values
                        if len(dist) > 0:
                            percentiles[pk][pool_metric] = rr._percentile_of(dist, marker)

        return {'percentiles': percentiles, 'markers': markers,
                'stats': stats, 'raw_all': raw_all}
    except Exception as e:
        log.warning(f"Failed to compute cohort percentiles: {e}")
        import traceback
        traceback.print_exc()
        return _empty_bundle()


# ============================================================================
# HTML GENERATION
# ============================================================================
class PresentationBuilder:
    """Builds self-contained HTML presentation."""

    def __init__(self, data, augment=False, refresh_yahoo=None):
        self.data = data
        self.augment = augment
        bundle = compute_cohort_percentiles(data)
        self.percentiles = bundle.get('percentiles', {})   # {(ticker,cohort):{metric:pct}}
        self.markers = bundle.get('markers', {})            # {ticker:{metric:raw_marker}}
        self.cohort_stats = bundle.get('stats', {})         # {(cohort,metric):(p10,p50,p90)}
        self.raw_all = bundle.get('raw_all')                # source-indexed raw playbook df
        # Yahoo augment (Section A only): persistent per-ticker store, fetch-missing-only.
        self.yahoo = load_yahoo_data(self._page_tickers(), augment, refresh=refresh_yahoo)
        self.html_parts = []

    def _page_tickers(self):
        """Ordered, de-duplicated list of every ticker that gets a page (general top-20 +
        top-5 per cohort). Single source of truth for both the Yahoo fetch set and build."""
        postrank_df = self.data['postrank_df']
        names = list(postrank_df.head(20)['source'])
        carveout = self.data.get('carveout_sidelists', {})
        for cohort_label in COHORTS:
            cp = carveout.get(cohort_label, {}).get('postRank', pd.DataFrame())
            if cp is not None and not cp.empty:
                names += list(cp.head(5)['source'])
        seen, out = set(), []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def _yahoo_block(self, ticker):
        """Section-A Yahoo block: business summary + company basics, every field gap-tagged
        on absence/failure and clearly labeled as Yahoo-sourced (not pipeline data)."""
        gap = '<span class="gap-inline">not available from Yahoo</span>'
        y = self.yahoo.get(ticker) or {}

        def f(key):
            v = y.get(key)
            if v is None or (isinstance(v, str) and not v.strip()) or v == []:
                return None
            return v

        summary = f('longBusinessSummary')
        if summary:
            summary_e = escape(summary)
            if len(summary) > 400:
                head = summary[:300].rsplit('.', 1)[0].strip()
                head = (head + '.') if head else (summary[:300].strip() + '…')
                desc = (f'<details><summary>{escape(head)}</summary>'
                        f'<p>{summary_e}</p></details>')
            else:
                desc = f'<p>{summary_e}</p>'
        else:
            desc = f'<p>business summary {gap}</p>'

        sector_industry = ' / '.join([x for x in [f('sector'), f('industry')] if x]) or gap
        hq = ', '.join([str(x) for x in [f('city'), f('country')] if x]) or gap
        emp = f('fullTimeEmployees')
        try:
            emp_str = f"{int(emp):,}" if emp is not None else gap
        except (ValueError, TypeError):
            emp_str = gap
        web = f('website')
        web_str = (f'<a href="{escape(str(web))}" target="_blank" rel="noopener">'
                   f'{escape(str(web))}</a>') if web else gap

        basics = (
            '<div class="yahoo-basics">'
            f'<span><strong>Yahoo sector/industry:</strong> {sector_industry}</span>'
            f'<span><strong>HQ:</strong> {hq}</span>'
            f'<span><strong>Employees:</strong> {emp_str}</span>'
            f'<span><strong>Website:</strong> {web_str}</span>'
            '</div>'
        )
        return (
            '<div class="description yahoo">'
            '<div class="yahoo-tag">source: Yahoo Finance (not pipeline data)</div>'
            f'{desc}{basics}'
            '</div>'
        )

    def raw_metric(self, ticker, metric):
        """Raw reviewReference playbook value for a ticker (Altman-Z, FCF yield, ...)."""
        ra = self.raw_all
        if ra is None or ticker not in ra.index or metric not in ra.columns:
            return np.nan
        return safe_float(ra.loc[ticker, metric])

    def dist_bar(self, ticker, cohort_label, pool_metric, marker=None):
        """Render the p10-p50-p90 distribution bar with the raw-value marker, or '' if the
        cohort distribution for this metric is unavailable."""
        stats = self.cohort_stats.get((cohort_label, pool_metric))
        if stats is None:
            return ""
        if marker is None:
            marker = self.markers.get(ticker, {}).get(pool_metric, np.nan)
        if marker is None or np.isnan(safe_float(marker)):
            return ""
        p10, p50, p90 = stats
        return create_distribution_bar(safe_float(marker), p10, p50, p90)

    def get_ticker_info(self, ticker):
        """Get name, exchange, sector for a ticker."""
        tickers_df = self.data.get('tickers_df')
        aggscore_df = self.data.get('aggscore_df')

        name = "Unknown"
        exchange = "—"
        sector = "—"

        if tickers_df is not None and not tickers_df.empty:
            # Try both 'symbol' and 'source' columns
            row = tickers_df[tickers_df['symbol'] == ticker]
            if row.empty:
                row = tickers_df[tickers_df.get('source', '') == ticker]
            if not row.empty:
                name = row.iloc[0].get('name', 'Unknown')
                exchange = row.iloc[0].get('exchange', '—')

        if aggscore_df is not None and not aggscore_df.empty:
            row = aggscore_df[aggscore_df['source'] == ticker]
            if not row.empty:
                sector = row.iloc[0].get('sector', self.data['get_industry'](ticker))
            else:
                sector = self.data['get_industry'](ticker)
        else:
            sector = self.data['get_industry'](ticker)

        return name, exchange, sector

    def get_rank_row(self, ticker):
        """Get the ranking row for a ticker (general or cohort)."""
        postrank_df = self.data['postrank_df']
        row = postrank_df[postrank_df['source'] == ticker]
        if not row.empty:
            return row.iloc[0]

        # Try cohort-specific postrank
        carveout_sidelists = self.data.get('carveout_sidelists', {})
        for cohort_label, cohort_dic in carveout_sidelists.items():
            cohort_postrank = cohort_dic.get('postRank', pd.DataFrame())
            if not cohort_postrank.empty:
                row = cohort_postrank[cohort_postrank['source'] == ticker]
                if not row.empty:
                    return row.iloc[0]

        return None

    def get_cdx_for_ticker(self, ticker):
        """Get the cdx_df rows for a ticker, sorted newest-first."""
        cdx_df = self.data.get('cdx_df')
        if cdx_df is None or cdx_df.empty:
            return pd.DataFrame()
        rows = cdx_df[cdx_df['source'] == ticker].copy()
        if 'date' in rows.columns:
            rows['date'] = pd.to_datetime(rows['date'], errors='coerce')
            rows = rows.sort_values('date', ascending=False)
        return rows

    def get_moat_components(self, ticker):
        """Get moat components for a ticker."""
        moatdf = self.data.get('moatdf')
        if moatdf is None or moatdf.empty:
            return {}
        row = moatdf[moatdf['source'] == ticker]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    def section_a_identity(self, ticker, rank):
        """Section A: Identity & orientation."""
        name, exchange, sector = self.get_ticker_info(ticker)
        aggscore_df = self.data['aggscore_df']

        row = self.get_rank_row(ticker)
        if row is None:
            return ""

        agg_score = row.get('AggScore', '—')
        price = row.get('price')
        price_str = f"${price:.2f}" if price and not np.isnan(safe_float(price)) else "—"

        # Market cap from latest cdx
        cdx = self.get_cdx_for_ticker(ticker)
        mktcap = latest_row_value(cdx, 'marketCap')
        mktcap_str = f"${mktcap/1e9:.2f}B" if mktcap and not np.isnan(mktcap) else "—"

        rating = "—"
        if aggscore_df is not None:
            ag_row = aggscore_df[aggscore_df['source'] == ticker]
            if not ag_row.empty:
                rating = ag_row.iloc[0].get('rating_fmp', '—')

        # Determine nav bucket
        nav_bucket = "top-5" if rank <= 5 else ("top-10" if rank <= 10 else "top-20")

        html = f"""
        <div class="section-a">
            <h2>{ticker} <span class="subtitle">{name}</span></h2>
            <div class="meta-row">
                <span><strong>Exchange:</strong> {exchange}</span>
                <span><strong>Sector:</strong> {sector}</span>
                <span><strong>Price:</strong> {price_str}</span>
                <span><strong>Market Cap:</strong> {mktcap_str}</span>
            </div>
            <div class="meta-row">
                <span><strong>AggScore:</strong> {ratio_format(agg_score, 1)}</span>
                <span><strong>Rank:</strong> {rank}</span>
                <span><strong>Nav Bucket:</strong> {nav_bucket}</span>
                <span><strong>FMP Rating:</strong> {rating}</span>
            </div>
            {self._yahoo_block(ticker)}
            <div class="editable">
                <label>Why is this cheap? ____________________________________________</label>
            </div>
        </div>
        """
        return html

    def section_b_flags(self, ticker):
        """Section B: Score & flags banner."""
        aggscore_df = self.data['aggscore_df']

        row = self.get_rank_row(ticker)
        if row is None:
            return ""

        agg_score = row.get('AggScore', np.nan)
        moat_score = row.get('moatScore', np.nan)

        forensic_tag = "—"
        if aggscore_df is not None:
            ag_row = aggscore_df[aggscore_df['source'] == ticker]
            if not ag_row.empty:
                forensic_tag = ag_row.iloc[0].get('forensicTag', '—')

        html = f"""
        <div class="section-b banner">
            <div class="score-banner">
                <span class="score-item"><strong>AggScore:</strong> {ratio_format(agg_score, 1)}</span>
                <span class="score-item"><strong>MoatScore:</strong> {ratio_format(moat_score, 1)}</span>
                <span class="score-item forensic"><strong>Forensic:</strong> {forensic_tag}</span>
            </div>
            <div class="flags">
                [computed flags will appear here]
            </div>
        </div>
        """
        return html

    def section_c_valuation(self, ticker, cohort_label):
        """Section C: Valuation ratios by industry (cohort-specific)."""
        aggscore_df = self.data.get('aggscore_df')
        cdx_df = self.get_cdx_for_ticker(ticker)

        row = self.get_rank_row(ticker)
        if row is None:
            return ""

        mktcap = latest_row_value(cdx_df, 'marketCap')

        if cohort_label == 'REIT':
            # REIT-specific metrics (all from cdx_df raw values, not z-scores)
            ffo_per_share = compute_ffo_per_share(cdx_df)
            # P/FFO = marketCap / TTM FFO (where FFO = ffo_per_share * shares)
            shares = latest_row_value(cdx_df, 'weightedAverageShsOut')
            ffo_total = ffo_per_share * shares if ffo_per_share and shares and not np.isnan(ffo_per_share) and not np.isnan(shares) else np.nan
            p_ffo = mktcap / ffo_total if mktcap and ffo_total and not np.isnan(mktcap) and not np.isnan(ffo_total) else np.nan
            ltv = compute_ltv_proxy(cdx_df)
            pb_ratio = latest_row_value(cdx_df, 'pbRatio')

            html = f"""
            <div class="section-c valuation">
                <h3>Valuation Ratios (REIT)</h3>
                <table class="metrics-table">
                    <tr><td><strong>FFO/Share (proxy)</strong></td><td>{ratio_format(ffo_per_share)}</td><td>TTM</td></tr>
                    <tr><td><strong>P/FFO</strong></td><td>{ratio_format(p_ffo)}</td><td>(proxy)</td></tr>
                    <tr><td><strong>LTV (proxy)</strong></td><td>{pct_format(ltv)}</td><td>debt/assets</td></tr>
                    <tr><td><strong>NAV Disc/Prem (proxy)</strong></td><td>{ratio_format(pb_ratio)}</td><td>P/B proxy</td></tr>
                </table>
                <div class="gap-note">[cap-rate, occupancy, WALE, AFFO not obtainable from filter data]</div>
            </div>
            """
            return html

        elif cohort_label == 'Mining':
            # Mining-specific metrics (raw values from cdx_df, not z-scores)
            net_debt_ebitda = latest_row_value(cdx_df, 'netDebtToEBITDA')
            fcf = latest_row_value(cdx_df, 'freeCashFlow')
            # CycleHeat IS a poolable playbook metric -> use its RAW reviewReference value
            # (the rank row carries a z-score, not the raw value) so the number matches
            # Section G and the dot-on-bar is a fair peer comparison.
            cycleheat = self.raw_metric(ticker, 'CycleHeat')
            ch_bar = self.dist_bar(ticker, cohort_label, 'CycleHeat', marker=cycleheat)

            html = f"""
            <div class="section-c valuation">
                <h3>Valuation Ratios (Mining)</h3>
                <table class="metrics-table">
                    <tr><td><strong>Net Debt/EBITDA</strong></td><td>{ratio_format(net_debt_ebitda)}</td><td>latest</td></tr>
                    <tr><td><strong>Free Cash Flow</strong></td><td>{ratio_format(fcf)}</td><td>latest Q</td></tr>
                    <tr><td><strong>CycleHeat</strong></td><td>{ratio_format(cycleheat)} {ch_bar}</td><td>strong signal</td></tr>
                </table>
                <div class="gap-note">[AISC, cost-curve, reserve-life not obtainable from filter data]</div>
            </div>
            """
            return html

        elif cohort_label in ['BalanceSheetFin', 'FinManager']:
            # Bank/FinManager metrics (raw values from cdx_df, not z-scores)
            pb = latest_row_value(cdx_df, 'pbRatio')
            roe = latest_row_value(cdx_df, 'returnOnEquity')
            # cdx_df's ROA column is 'returnOnAssets' (the pool uses 'RoA').
            roa = latest_row_value(cdx_df, 'returnOnAssets')
            op_margin = compute_operating_margin(cdx_df)

            roe_bar = self.dist_bar(ticker, cohort_label, 'returnOnEquity', marker=roe)
            roa_bar = self.dist_bar(ticker, cohort_label, 'RoA', marker=roa)

            html = f"""
            <div class="section-c valuation">
                <h3>Valuation Ratios ({'Bank' if cohort_label == 'BalanceSheetFin' else 'FinManager'})</h3>
                <table class="metrics-table">
                    <tr><td><strong>P/B</strong></td><td>{ratio_format(pb)}</td><td>latest</td></tr>
                    <tr><td><strong>ROE</strong></td><td>{pct_format(roe)} {roe_bar}</td><td>latest</td></tr>
                    <tr><td><strong>ROA</strong></td><td>{pct_format(roa)} {roa_bar}</td><td>latest</td></tr>
                    <tr><td><strong>Op Margin</strong></td><td>{pct_format(op_margin)}</td><td>TTM</td></tr>
                </table>
                <div class="gap-note">[NIM, efficiency ratio, NPL, CET1, AUM, fee-margin not obtainable from filter data]</div>
            </div>
            """
            return html

        elif cohort_label == 'InvestmentVehicle':
            # Investment vehicle metrics (raw values from cdx_df, not z-scores)
            pb = latest_row_value(cdx_df, 'pbRatio')
            roe = latest_row_value(cdx_df, 'returnOnEquity')
            roic = latest_row_value(cdx_df, 'returnOnCapitalEmployed')

            roe_bar = self.dist_bar(ticker, cohort_label, 'returnOnEquity', marker=roe)
            roic_bar = self.dist_bar(ticker, cohort_label, 'returnOnCapitalEmployed', marker=roic)

            html = f"""
            <div class="section-c valuation">
                <h3>Valuation Ratios (InvestmentVehicle)</h3>
                <table class="metrics-table">
                    <tr><td><strong>P/B (NAV proxy)</strong></td><td>{ratio_format(pb)}</td><td>disc to NAV</td></tr>
                    <tr><td><strong>ROE</strong></td><td>{pct_format(roe)} {roe_bar}</td><td>latest</td></tr>
                    <tr><td><strong>ROIC</strong></td><td>{ratio_format(roic)} {roic_bar}</td><td>latest</td></tr>
                </table>
                <div class="gap-note">[NAV/holdings composition not obtainable from filter data]</div>
            </div>
            """
            return html

        else:
            # General block (default) - all raw values from cdx_df, not z-scores from postrank
            roic = latest_row_value(cdx_df, 'returnOnCapitalEmployed')
            gm = latest_row_value(cdx_df, 'grossProfitMargin')
            op_margin = compute_operating_margin(cdx_df)
            fcf_margin = compute_fcf_margin_ttm(cdx_df)
            cash_conv = compute_cash_conversion(cdx_df)
            income_qual = latest_row_value(cdx_df, 'incomeQuality')
            net_debt_ebitda = latest_row_value(cdx_df, 'netDebtToEBITDA')
            int_cov = compute_interest_coverage(cdx_df)
            # P/E: general names -> AggScoreTop100['PE-ratio']; fallback / carve names not
            # in the top-100 CSV -> 1 / earningsYield (cdx latest), guarding non-positive
            # or NaN earnings (gap-tag rather than emit a garbage negative/huge P/E).
            pe_ratio = np.nan
            if aggscore_df is not None and not aggscore_df.empty:
                ag_row = aggscore_df[aggscore_df['source'] == ticker]
                if not ag_row.empty:
                    pe_ratio = safe_float(ag_row.iloc[0].get('PE-ratio'))
            if np.isnan(pe_ratio):
                ey = latest_row_value(cdx_df, 'earningsYield')
                if not np.isnan(ey) and ey > 0:
                    pe_ratio = 1.0 / ey
            # FCF yield has no native cdx_df column -- source the raw reviewReference pool
            # value (same metric the cohort distribution is built on).
            fcf_yield = self.raw_metric(ticker, 'freeCashFlowYield')

            roic_bar = self.dist_bar(ticker, cohort_label, 'returnOnCapitalEmployed', marker=roic)
            gm_bar = self.dist_bar(ticker, cohort_label, 'grossProfitMargin', marker=gm)
            iq_bar = self.dist_bar(ticker, cohort_label, 'incomeQuality', marker=income_qual)
            fcfy_bar = self.dist_bar(ticker, cohort_label, 'freeCashFlowYield', marker=fcf_yield)

            html = f"""
            <div class="section-c valuation">
                <h3>Valuation Ratios</h3>
                <table class="metrics-table">
                    <tr><td><strong>ROIC/ROCE</strong></td><td>{ratio_format(roic)} {roic_bar}</td><td>proxy</td></tr>
                    <tr><td><strong>Gross Margin</strong></td><td>{pct_format(gm)} {gm_bar}</td><td>latest</td></tr>
                    <tr><td><strong>Op Margin</strong></td><td>{pct_format(op_margin)}</td><td>TTM</td></tr>
                    <tr><td><strong>FCF Margin</strong></td><td>{pct_format(fcf_margin)}</td><td>TTM</td></tr>
                    <tr><td><strong>Cash Conversion</strong></td><td>{ratio_format(cash_conv)}</td><td>TTM FCF / NI</td></tr>
                    <tr><td><strong>Income Quality</strong></td><td>{ratio_format(income_qual)} {iq_bar}</td><td>audit</td></tr>
                    <tr><td><strong>Net Debt/EBITDA</strong></td><td>{ratio_format(net_debt_ebitda)}</td><td>latest</td></tr>
                    <tr><td><strong>Interest Coverage</strong></td><td>{ratio_format(int_cov)}</td><td>op inc / int exp</td></tr>
                    <tr><td><strong>P/E</strong></td><td>{ratio_format(pe_ratio)}</td><td>traded or yield inv</td></tr>
                    <tr><td><strong>FCF Yield</strong></td><td>{pct_format(fcf_yield)} {fcfy_bar}</td><td>reviewRef raw (TTM)</td></tr>
                </table>
                <div class="gap-note">[WACC, EV/EBIT not obtainable from filter data]</div>
            </div>
            """
            return html

    def section_d_trends(self, ticker):
        """Section D: Multi-year trend charts (all 12 metrics per spec)."""
        cdx = self.get_cdx_for_ticker(ticker)
        if cdx.empty:
            return '<div class="section-d"><p>[No quarterly data available]</p></div>'

        # Get time-series data (newest-first, so reverse for display)
        cdx = cdx.sort_values('date') if 'date' in cdx.columns else cdx

        # Helper to get trend series (oldest to newest for charting)
        def get_trend(col):
            vals = cdx[col].dropna().values.tolist()
            return vals if vals else None

        # Compute operating margin trend
        op_margin_trend = []
        for _, row in cdx.iterrows():
            if pd.notna(row.get('revenue')) and pd.notna(row.get('operatingIncome')) and row.get('revenue') != 0:
                op_margin_trend.append(row['operatingIncome'] / row['revenue'])
        op_margin_trend = op_margin_trend if op_margin_trend else None

        sparklines = []
        metrics = [
            ('Revenue', get_trend('revenue')),
            ('Net Income', get_trend('netIncome')),
            ('Gross Margin', get_trend('grossProfitMargin')),
            ('Operating Margin', op_margin_trend),
            ('ROIC', get_trend('returnOnCapitalEmployed')),
            ('ROE', get_trend('returnOnEquity')),
            ('Free Cash Flow', get_trend('freeCashFlow')),
            ('Shares Outstanding', get_trend('weightedAverageShsOut')),
            ('Book Value/Share', get_trend('bookValuePerShare')),
            ('Net Debt/EBITDA', get_trend('netDebtToEBITDA')),
            ('Days Sales Outstanding', get_trend('daysSalesOutstanding')),
            ('Days Inventory Outstanding', get_trend('daysOfInventoryOutstanding')),
        ]

        for name, values in metrics:
            if values:
                sparklines.append(f"<div><strong>{name}:</strong> {create_sparkline_svg(values)}</div>")

        html = f"""
        <div class="section-d trends">
            <h3>Multi-Year Trends (Quarterly)</h3>
            {''.join(sparklines) if sparklines else '<p>[No trend data]</p>'}
        </div>
        """
        return html

    def section_e_moat(self, ticker):
        """Section E: Moat checklist."""
        moat_comp = self.get_moat_components(ticker)
        postrank_df = self.data['postrank_df']

        row = postrank_df[postrank_df['source'] == ticker]
        if row.empty:
            moat_score = np.nan
        else:
            moat_score = row.iloc[0].get('moatScore', np.nan)

        html = f"""
        <div class="section-e moat">
            <h3>Moat Checklist</h3>
            <p><strong>Moat Score:</strong> {ratio_format(moat_score)}</p>
            <table class="moat-components">
        """

        for metric in ['FCFyield', 'GrossMargin', 'RevtoASS', 'RoE', 'RoA', 'ROIC',
                       'SGAtoGP', 'DeptoGP', 'NetMargin', 'CapExtoEarnings', 'TLtoEquity']:
            val = moat_comp.get(metric, np.nan)
            html += f"<tr><td>{metric}:</td><td>{ratio_format(val)}</td></tr>"

        html += f"""
            </table>
            <div class="editable">
                <label>Moat verdict: ____________________________________________</label>
            </div>
        </div>
        """
        return html

    def section_f_forensic(self, ticker):
        """Section F: Forensic / accounting quality."""
        aggscore_df = self.data.get('aggscore_df')
        forensic_df = self.data.get('forensic_df')
        cdx = self.get_cdx_for_ticker(ticker)

        row = self.get_rank_row(ticker)
        if row is None:
            return '<div class="section-f"></div>'

        # Raw incomeQuality from cdx (the rank row's incomeQuality is a z-score, not a
        # ratio) -- must match Section C's value for the same name.
        income_qual = latest_row_value(cdx, 'incomeQuality')
        fcf_ttm = ttm_sum(cdx, 'freeCashFlow')
        ni_ttm = ttm_sum(cdx, 'netIncome')

        m_score = "—"
        c_score = "—"
        sloan = "—"
        forensic_tag = "—"

        if aggscore_df is not None:
            ag_row = aggscore_df[aggscore_df['source'] == ticker]
            if not ag_row.empty:
                m_score = ag_row.iloc[0].get('M-Score', '—')
                c_score = ag_row.iloc[0].get('C-Score', '—')
                sloan = ag_row.iloc[0].get('sloanAccruals', '—')
                forensic_tag = ag_row.iloc[0].get('forensicTag', '—')

        html = f"""
        <div class="section-f forensic">
            <h3>Forensic / Accounting Quality</h3>
            <table class="forensic-table">
                <tr><td><strong>M-Score</strong></td><td>{m_score}</td></tr>
                <tr><td><strong>C-Score</strong></td><td>{c_score}</td></tr>
                <tr><td><strong>Sloan Accruals</strong></td><td>{sloan}</td></tr>
                <tr><td><strong>Income Quality</strong></td><td>{ratio_format(income_qual)}</td></tr>
                <tr><td><strong>FCF vs Net Income (TTM)</strong></td><td>FCF: {ratio_format(fcf_ttm)} / NI: {ratio_format(ni_ttm)}</td></tr>
                <tr><td><strong>Forensic Tag</strong></td><td>{forensic_tag}</td></tr>
            </table>
        </div>
        """
        return html

    def section_g_cohort(self, ticker, cohort_label):
        """Section G: Cohort/peer distribution + vs-MSCI (using RAW values, not z-scores)."""
        cdx = self.get_cdx_for_ticker(ticker)
        if cdx.empty:
            return '<div class="section-g cohort"><p>[No peer data]</p></div>'

        # Percentiles are keyed by (ticker, cohort context): general names -> general pool,
        # carve names -> their carve cohort's pool.
        ticker_percentiles = self.percentiles.get((ticker, cohort_label), {})
        ticker_markers = self.markers.get(ticker, {})

        html = f"""
        <div class="section-g cohort">
            <h3>Peer Distribution (Cohort: {cohort_label})</h3>
            <div class="cohort-metrics">
        """

        # Raw marker (cdx-latest, or reviewReference raw pool for FCF yield) + percentile
        # within the cohort's raw distribution + p10-p50-p90 spread bar.
        for label, _cdx_col, pool_metric, fmt in SECTION_G_METRICS:
            marker = ticker_markers.get(pool_metric, np.nan)
            pct = ticker_percentiles.get(pool_metric, np.nan)
            val_str = pct_format(marker) if fmt == 'pct' else ratio_format(marker)
            bar = self.dist_bar(ticker, cohort_label, pool_metric, marker=marker)
            html += (f'<div><strong>{label}:</strong> {val_str} '
                     f'<span class="pctile">({pctile_format(pct)} pct)</span> {bar}</div>')

        html += f"""
            </div>
            <div class="gap-note">[vs-MSCI trailing return not obtainable from saved artifacts]</div>
        </div>
        """
        return html

    def section_h_flags(self, ticker):
        """Section H: Computed auto-highlight flags (using RAW cdx_df values, not z-scores)."""
        cdx = self.get_cdx_for_ticker(ticker)

        row = self.get_rank_row(ticker)
        if row is None:
            return ""

        flags = []

        # Solvency check - Altman-Z is NOT a cdx_df column (it lives in postRank as a
        # z-score, and raw in the reviewReference pool). Source the RAW computed value so
        # a genuinely distressed name actually trips the flag.
        altman_z = self.raw_metric(ticker, 'Altman-Z')
        if not np.isnan(safe_float(altman_z)) and safe_float(altman_z) < 1.8:
            flags.append(('RED', 'Solvency Risk (Z<1.8)'))

        # Interest coverage check
        int_cov = compute_interest_coverage(cdx)
        if not np.isnan(safe_float(int_cov)) and safe_float(int_cov) < 2:
            flags.append(('RED', 'Low Interest Coverage (<2)'))

        # Leverage check - use raw netDebtToEBITDA from cdx
        net_debt_ebitda = latest_row_value(cdx, 'netDebtToEBITDA')
        if not np.isnan(safe_float(net_debt_ebitda)) and safe_float(net_debt_ebitda) > 4:
            flags.append(('AMBER', 'High Leverage (ND/EBITDA>4)'))

        # Dilution check (shares rising 3y)
        shares = cdx['weightedAverageShsOut'].dropna().values
        if len(shares) >= 12:
            oldest = shares[-1]
            newest = shares[0]
            if oldest > 0 and (newest / oldest - 1) > 0.1:
                flags.append(('RED', 'Dilution (shares +10% in 3y)'))

        # Cash vs earnings - use raw incomeQuality from cdx
        income_qual = latest_row_value(cdx, 'incomeQuality')
        if not np.isnan(safe_float(income_qual)) and safe_float(income_qual) < 0.7:
            flags.append(('AMBER', 'Cash vs Earnings (IQ<0.7)'))

        # Margin erosion (3y decline)
        gm = cdx['grossProfitMargin'].dropna().values
        if len(gm) >= 12:
            old_gm = gm[-1]
            new_gm = gm[0]
            if not np.isnan(old_gm) and not np.isnan(new_gm) and new_gm < old_gm - 0.05:
                flags.append(('AMBER', 'Margin Erosion (GM down 3y)'))

        flag_html = " ".join([f'<span class="flag {color}">{label}</span>'
                              for color, label in flags])

        html = f"""
        <div class="section-h flags">
            <div class="flag-strip">
                {flag_html if flag_html else '<span style="color:#999;">No flags</span>'}
            </div>
        </div>
        """
        return html

    def build_name_page(self, ticker, rank, cohort_label='general'):
        """Build complete page for one name."""
        html = f"""
        <div class="name-page" id="{ticker}">
            <div class="page-header">
                <h1>{ticker}</h1>
                <p class="rank-info">Rank #{rank} | Cohort: {cohort_label}</p>
            </div>
        """

        html += self.section_a_identity(ticker, rank)
        html += self.section_b_flags(ticker)
        html += self.section_c_valuation(ticker, cohort_label)
        html += self.section_d_trends(ticker)
        html += self.section_e_moat(ticker)
        html += self.section_f_forensic(ticker)
        html += self.section_g_cohort(ticker, cohort_label)
        html += self.section_h_flags(ticker)

        html += """
        </div>
        <hr class="page-break">
        """
        return html

    def build_html(self):
        """Build complete HTML document."""
        run_date = self.data['run_date']
        postrank_df = self.data['postrank_df']

        # Get top 20 general
        top_20 = postrank_df.head(20)['source'].tolist()

        # Get top 5 per cohort
        cohort_names = {}
        carveout_sidelists = self.data.get('carveout_sidelists', {})
        for cohort_label in COHORTS:
            cohort_dic = carveout_sidelists.get(cohort_label, {})
            cohort_postrank = cohort_dic.get('postRank', pd.DataFrame())
            if not cohort_postrank.empty:
                top_5 = cohort_postrank.head(5)['source'].tolist()
                cohort_names[cohort_label] = top_5

        # Build nav
        nav_html = self._build_nav(top_20, cohort_names)

        # Build content
        content = """<div class="content">"""

        # General top-20
        for i, ticker in enumerate(top_20, 1):
            content += self.build_name_page(ticker, i, 'general')

        # Cohort sections
        for cohort_label, tickers in cohort_names.items():
            content += f'<div class="cohort-section"><h1>Cohort: {cohort_label}</h1></div>\n'
            for i, ticker in enumerate(tickers, 1):
                content += self.build_name_page(ticker, i, cohort_label)

        content += "</div>"

        # Assemble full HTML
        css = self._get_css()

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Investment Filter Presentation - {run_date}</title>
    <style>
{css}
    </style>
</head>
<body>
    <div class="container">
        {nav_html}
        {content}
    </div>
</body>
</html>
"""
        return html

    def _build_nav(self, general_tickers, cohort_tickers):
        """Build navigation HTML."""
        nav = """<nav class="sidebar">
            <div class="nav-header">Investment Filter</div>
            <div class="nav-section">
                <h4>General Top-20</h4>
                <div class="nav-group">
                    <h5>Top-5</h5>
        """

        for ticker in general_tickers[:5]:
            nav += f'<a href="#{ticker}">{ticker}</a>\n'

        nav += """
                </div>
                <div class="nav-group">
                    <h5>Top-10</h5>
        """

        for ticker in general_tickers[5:10]:
            nav += f'<a href="#{ticker}">{ticker}</a>\n'

        nav += """
                </div>
                <div class="nav-group">
                    <h5>Top-20</h5>
        """

        for ticker in general_tickers[10:20]:
            nav += f'<a href="#{ticker}">{ticker}</a>\n'

        nav += """
                </div>
            </div>
        """

        for cohort_label, tickers in cohort_tickers.items():
            nav += f"""
            <div class="nav-section">
                <h4>{cohort_label}</h4>
            """
            for ticker in tickers:
                nav += f'<a href="#{ticker}">{ticker}</a>\n'
            nav += """
            </div>
            """

        nav += """</nav>"""
        return nav

    def _get_css(self):
        """Return inline CSS."""
        return """
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f5f5;
    color: #333;
    line-height: 1.6;
}

.container {
    display: flex;
    max-width: 1600px;
    margin: 0 auto;
}

nav.sidebar {
    width: 220px;
    background: #fff;
    padding: 20px;
    border-right: 1px solid #ddd;
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
    font-size: 0.9em;
}

.nav-header {
    font-weight: bold;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid #0066cc;
}

.nav-section {
    margin-bottom: 20px;
}

.nav-section h4 {
    font-size: 0.95em;
    margin-bottom: 8px;
    color: #0066cc;
}

.nav-group {
    margin-left: 10px;
    margin-bottom: 12px;
}

.nav-group h5 {
    font-size: 0.85em;
    color: #666;
    margin-bottom: 4px;
}

.nav-group a {
    display: block;
    color: #0066cc;
    text-decoration: none;
    padding: 2px 4px;
    margin: 2px 0;
    font-size: 0.85em;
}

.nav-group a:hover {
    background: #e8f0ff;
    border-radius: 3px;
}

.content {
    flex: 1;
    padding: 30px;
}

.name-page {
    background: #fff;
    margin-bottom: 30px;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.page-header {
    border-bottom: 2px solid #0066cc;
    padding-bottom: 15px;
    margin-bottom: 20px;
}

.page-header h1 {
    font-size: 2em;
    margin-bottom: 5px;
}

.page-header .rank-info {
    color: #666;
    font-size: 0.95em;
}

.section-a .subtitle {
    font-weight: normal;
    color: #666;
    font-size: 0.8em;
}

.meta-row {
    display: flex;
    gap: 30px;
    margin: 10px 0;
    flex-wrap: wrap;
    font-size: 0.95em;
}

.meta-row span {
    flex: 0 1 auto;
}

.description {
    margin: 15px 0;
    padding: 15px;
    background: #f9f9f9;
    border-left: 3px solid #0066cc;
}

.description.yahoo {
    border-left-color: #6f42c1;
}

.yahoo-tag {
    font-size: 0.75em;
    color: #6f42c1;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-bottom: 8px;
}

.description.yahoo details summary {
    cursor: pointer;
    color: #333;
}

.description.yahoo details p {
    margin-top: 8px;
    color: #444;
}

.yahoo-basics {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 24px;
    margin-top: 12px;
    font-size: 0.9em;
}

.gap-inline {
    color: #b0b0b0;
    font-style: italic;
}

.editable label {
    display: block;
    margin-top: 15px;
    padding: 10px;
    background: #fffef0;
    border: 1px dashed #ccc;
    border-radius: 3px;
    font-size: 0.9em;
}

.section-b.banner {
    background: #f0f7ff;
    padding: 15px;
    border-radius: 5px;
    margin: 20px 0;
}

.score-banner {
    display: flex;
    gap: 30px;
    margin-bottom: 10px;
}

.score-item {
    font-size: 0.95em;
}

.score-item.forensic {
    color: #d9534f;
}

.section-c h3,
.section-d h3,
.section-e h3,
.section-f h3,
.section-g h3 {
    margin-top: 25px;
    margin-bottom: 15px;
    font-size: 1.3em;
    color: #0066cc;
}

.metrics-table,
.moat-components,
.forensic-table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    font-size: 0.95em;
}

.metrics-table td,
.moat-components td,
.forensic-table td {
    padding: 8px;
    border-bottom: 1px solid #eee;
}

.metrics-table td:first-child,
.moat-components td:first-child,
.forensic-table td:first-child {
    font-weight: 500;
    width: 40%;
}

.metrics-table td:nth-child(2),
.moat-components td:nth-child(2),
.forensic-table td:nth-child(2) {
    text-align: right;
    font-family: monospace;
}

.gap-note {
    margin-top: 10px;
    padding: 8px;
    background: #f5f5f5;
    color: #999;
    font-size: 0.85em;
    border-left: 3px solid #ddd;
}

.cohort-metrics div {
    margin: 5px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.pctile {
    color: #666;
    font-size: 0.85em;
}

.trends {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.flag-strip {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.flag {
    padding: 4px 10px;
    border-radius: 3px;
    font-size: 0.8em;
    font-weight: bold;
}

.flag.RED {
    background: #f8d7da;
    color: #721c24;
}

.flag.AMBER {
    background: #fff3cd;
    color: #856404;
}

.flag.GREEN {
    background: #d4edda;
    color: #155724;
}

.page-break {
    margin: 40px 0;
    border: none;
    border-top: 2px solid #ddd;
}

.cohort-section {
    margin-top: 40px;
    padding: 20px;
    background: #f0f7ff;
    border-left: 5px solid #0066cc;
    margin-bottom: 30px;
}

.cohort-section h1 {
    color: #0066cc;
}

@media print {
    nav.sidebar {
        display: none;
    }
    .container {
        max-width: 100%;
    }
    .name-page {
        page-break-inside: avoid;
    }
}
"""


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate standalone HTML presentation from investment-filter run")
    parser.add_argument('--run-dir', type=str, default=str(DEFAULT_RUN_DIR),
                       help='Directory containing saved run artifacts')
    parser.add_argument('--out', type=str, default=None,
                       help='Output HTML file path override (default: '
                            '<VALUATION_REPO>/presentations/presentation_<date>.html)')
    parser.add_argument('--augment', type=str, choices=['on', 'off'], default='on',
                       help='Online Yahoo Finance augmentation of Section A (default: ON). '
                            'Use "off" (or --no-augment) for a fully offline run; missing '
                            'Yahoo data degrades to gap tags.')
    parser.add_argument('--no-augment', action='store_true',
                       help='Shortcut for --augment off (offline run).')
    parser.add_argument('--refresh-yahoo', nargs='?', const='__ALL__', default=None,
                       metavar='TICKERS',
                       help='Force re-fetch of Yahoo info (default is fetch-missing-only). '
                            'Bare flag = refresh ALL page names; or pass a comma-separated '
                            'ticker list to refresh just those.')

    args = parser.parse_args()

    augment = (args.augment == 'on') and not args.no_augment
    if args.refresh_yahoo is None:
        refresh_yahoo = None
    elif args.refresh_yahoo == '__ALL__':
        refresh_yahoo = True
    else:
        refresh_yahoo = {t.strip() for t in args.refresh_yahoo.split(',') if t.strip()}

    try:
        # Load data
        data = load_run_data(args.run_dir, VALUATION_REPO)
        log.info(f"Loaded run data for {len(data['postrank_df'])} names")

        # Build presentation
        builder = PresentationBuilder(data, augment=augment, refresh_yahoo=refresh_yahoo)
        html = builder.build_html()

        # Determine output path.
        # Rule: read the saved run from --run-dir, but ALWAYS write the HTML into the
        # repo's presentations/ dir unless --out overrides. This keeps the CEO's live
        # picks inside the repo (gitignored) regardless of where the run was read from.
        if args.out:
            out_path = args.out
        else:
            presentations_dir = os.path.join(str(VALUATION_REPO), 'presentations')
            os.makedirs(presentations_dir, exist_ok=True)
            out_path = os.path.join(presentations_dir, f"presentation_{data['run_date']}.html")

        # Write HTML
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html)

        log.info(f"Presentation written to {out_path}")
        log.info(f"File size: {os.path.getsize(out_path) / 1024:.1f} KB")

        # Verify self-contained: flag external RESOURCE LOADS (src=, stylesheet <link>,
        # @import, CSS url()) -- these break offline opening. Outbound <a href> links (the
        # Yahoo website links) and text URLs inside business summaries are fine: they do not
        # load anything to render, so they don't affect self-containment.
        with open(out_path, 'r', encoding='utf-8') as f:
            content = f.read()
            import re
            has_external_src = bool(re.search(r'src\s*=\s*["\'](?!data:)', content))
            has_ext_css = bool(re.search(r'<link[^>]+rel\s*=\s*["\']?stylesheet', content)) \
                or '@import' in content or bool(re.search(r'url\(\s*https?://', content))
            if has_external_src or has_ext_css:
                log.warning("HTML loads an external resource (may not be fully self-contained)")
            else:
                log.info("HTML is self-contained (no external resource loads; SVG inlined; "
                         "outbound <a> links only)")

        return 0

    except Exception as e:
        log.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
