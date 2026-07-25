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
import re
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
    # ROE/ROA formatted as % here to MATCH Section C's pct_format (reviewer F6: the same
    # metric must not render twice with inconsistent formatting on a cohort page).
    ('ROE',                 'returnOnEquity',          'returnOnEquity',             'pct'),
    ('ROA',                 'returnOnAssets',          'RoA',                        'pct'),
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


def ttm_aligned_sums(df, cols):
    """Trailing-12-month sums for several columns computed over the SAME set of quarters
    (the newest 4 rows where EVERY listed column is non-NaN). Unlike calling ttm_sum per
    column -- which drops NaNs independently and can sum DIFFERENT quarters per column --
    this keeps a paired sum (e.g. R2's netIncome vs operating cash flow) aligned on one
    consistent period set. Returns a tuple of floats (np.nan where unavailable), in `cols`
    order. `df` is assumed newest-first (get_cdx_for_ticker's order)."""
    n = len(cols)
    if df is None or df.empty or any(c not in df.columns for c in cols):
        return tuple(np.nan for _ in cols)
    sub = df[cols].dropna()          # rows where ALL cols present -> aligned quarters
    if sub.empty:
        return tuple(np.nan for _ in cols)
    sub = sub.head(4)
    return tuple(float(sub[c].sum()) for c in cols)


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


def create_distribution_bar(value, p10, p50, p90, width=150, height=15, benchmark=None):
    """Create a distribution bar showing value against p10-p50-p90 spread.

    `benchmark` (optional): a RAW threshold value in the same units as `value`. When given
    (and finite), a thin solid reference tick is drawn on the EXISTING bar at the threshold's
    normalized position (clamped to the bar). This is PURELY ADDITIVE -- when benchmark is
    None the SVG is byte-identical to the pre-benchmark output, so no existing bar moves."""
    if np.isnan(p10) or np.isnan(p50) or np.isnan(p90) or p10 == p90:
        return '<span style="color:#999;">—</span>'

    # Normalize positions
    val_norm = (value - p10) / (p90 - p10) if p90 > p10 else 0.5
    val_norm = max(0, min(1, val_norm))

    p50_norm = (p50 - p10) / (p90 - p10) if p90 > p10 else 0.5

    marker_x = width * val_norm
    p50_x = width * p50_norm

    # Optional benchmark tick (drawn before the marker so the value dot sits on top).
    bench_svg = ''
    bm = safe_float(benchmark)
    if benchmark is not None and not np.isnan(bm) and p90 > p10:
        bm_norm = max(0, min(1, (bm - p10) / (p90 - p10)))
        bm_x = width * bm_norm
        bench_svg = (f'<line x1="{2 + bm_x}" y1="0" x2="{2 + bm_x}" y2="{height + 4}" '
                     f'stroke="#c0392b" stroke-width="1.5"/>')

    svg = f'''<svg width="{width + 20}" height="{height + 4}" style="display:inline-block;">
      <rect x="2" y="2" width="{width}" height="{height}" fill="#f0f0f0" stroke="#ccc" stroke-width="0.5"/>
      <line x1="{2 + p50_x}" y1="1" x2="{2 + p50_x}" y2="{height + 3}" stroke="#999" stroke-width="1" stroke-dasharray="2,2"/>
      {bench_svg}<circle cx="{2 + marker_x}" cy="{2 + height/2}" r="3" fill="#0066cc"/>
    </svg>'''
    return svg


def quarter_label(ts):
    """Compact calendar-quarter label for a Timestamp, e.g. 'Q2 2020'."""
    try:
        q = (ts.month - 1) // 3 + 1
        return f"Q{q} {ts.year}"
    except Exception:
        return "?"


def span_caption(dates):
    """Rough length of a date span for a sparkline: '~6y' (>= ~9 months) or '~5q'
    (shorter). `dates` is oldest->newest. Returns '' when the span can't be computed."""
    if not dates or len(dates) < 2:
        return ""
    try:
        yrs = (dates[-1] - dates[0]).days / 365.25
    except Exception:
        return ""
    if yrs >= 0.75:
        return f"~{yrs:.0f}y"
    return f"~{len(dates)}q"


# ============================================================================
# EXTENDED PEER-BAR POOLS (metrics beyond the 16 playbook set)
# ============================================================================
# Global rules (per the bars-expansion build spec):
#  1. winsorize every extended pool at p1/p99 before spread/percentile;
#  2. draw a bar only if the metric has >= MIN_POOL_N non-NaN cohort members, else
#     "pool too small (n=X)";
#  3. financial-cohort suppression for metrics meaningless to banks/asset managers;
#  4. de-dup vs the canonical cdx/Section-G bars (only non-overlapping moat comps added).
# This is a SEPARATE, additive system from the 16-playbook machinery (self.cohort_stats),
# so no existing displayed number moves. Pools are winsorized here; the original 16 are NOT.
MIN_POOL_N = 30
FIN_COHORTS = {'FinManager', 'BalanceSheetFin'}
# Extended metrics suppressed on financial cohorts (economically meaningless there).
FIN_SUPPRESS = {'op_margin', 'fcf_margin', 'interest_coverage', 'inv_days', 'dso',
                'net_debt_ebitda', 'SGAtoGP', 'DeptoGP'}
# Non-duplicate moat components pooled straight from moatdf (rule 4).
EXT_MOAT_COLS = ['RevtoASS', 'SGAtoGP', 'DeptoGP', 'NetMargin', 'CapExtoEarnings', 'TLtoEquity']
# cdx columns the reducer needs.
_EXT_CDX_COLS = ['date', 'revenue', 'operatingIncome', 'freeCashFlow', 'netIncome',
                 'interestExpense', 'netDebtToEBITDA', 'effectiveTaxRate',
                 'daysSalesOutstanding', 'daysOfInventoryOutstanding',
                 'netCashProvidedByOperatingActivities', 'totalAssets',
                 'weightedAverageShsOut', 'marketCap', 'depreciationAndAmortization',
                 'dividendsPaid', 'longTermDebt']


# ============================================================================
# ORIENTATION + BENCHMARK STATIC TABLES  (additive presentation layer)
# ============================================================================
# Two decoupled, data-driven tables read verbatim from the build spec
# (benchmarks-orientation-spec.md). Rendering is driven ENTIRELY from these dicts so the
# layer stays maintainable and every direction is stated once, in one place. This is PURELY
# ADDITIVE: it renders chips/ticks alongside existing values and never changes a displayed
# number, bar, or chart.
#
# ORIENTATION -- one chip per index/score: scale + which way is good + a one-line meaning
# (shown as a tooltip). Keys match how each metric is referenced at its render site:
#   * raw-level metrics (Sections C/F/G)  -> keyed by the pool_metric / cdx name;
#   * moat components (Section E)          -> namespaced 'moat:<Component>' so the
#                                            threshold-relative orientation never collides
#                                            with a same-named raw metric (e.g. RoA).
# CRITICAL (see spec SIGN-FLIP): the four moat components SGAtoGP/DeptoGP/CapExtoEarnings/
# TLtoEquity are stored as (threshold - ratio), so a HIGHER value is GOOD -> '↑ better'.
ORIENTATION = {
    # -- Composites --
    'AggScore':   ('unbounded (emp −0.47…0.40)', '↑ better',
                   'Stage-2 weighted sum of z-scored metrics'),
    'moatScore':  ('0–11 (emp 2–6 in top-100)', '↑ better',
                   'count of 11 quality boxes ticked'),
    # -- Forensic (Sections F / G) --
    'M-Score':      ('stored Beneish+1.78', '↓ better (>0 flags)',
                     'earnings-manipulation index'),
    'C-Score':      ('0–6', '↓ better (≥4 flag, 0 clean)',
                     'count of 6 accounting red flags'),
    'sloanAccruals':('ratio', '↓ better (low/neg = cash-backed)',
                     'balance-sheet accruals; pipeline flag = worst quintile in-run, not absolute'),
    'incomeQuality':('≈1 neutral (emp −13.8…8.0)', '↑ better (≈>1 good)',
                     'CFO/NI; noisy near NI≈0'),
    # 2026-07-19 soundness audit: the pipeline's computed variant DEVIATES from published
    # Altman-Z (wrong x2 variable, quarter-scale flow terms), so the published 1.8/3.0
    # thresholds do not apply to the quantity we compute -> no hard verdict, relative-only.
    'Altman-Z':     ('relative-only (uncalibrated)', '↑ better',
                     'computed variant deviates from published Altman-Z — thresholds not '
                     'calibrated; treat as relative-only pending fix'),
    'Piotroski':    ('0–9 (≥7 strong, ≤3 weak)', '↑ better',
                     'count of 9 fundamental-improvement tests'),
    'CycleHeat':    ('[−3,3]', '↓ better (penalized, w=−0.080)',
                     'EPS heat vs the stock’s own history; high = late-cycle risk'),
    # -- Stage-2 valuation/quality (raw-level, Sections C / G) --
    'currentRatio': ('≥1.5 healthy', '↑ better',
                     'current assets / current liabilities — short-term liquidity'),
    'returnOnCapitalEmployed': ('ratio', '↑ better', 'return on capital employed (raw level)'),
    'returnOnEquity':          ('ratio', '↑ better', 'return on equity (raw level)'),
    'RoA':                     ('ratio', '↑ better', 'return on assets (raw level)'),
    'grossProfitMargin':       ('ratio', '↑ better', 'gross margin (raw level)'),
    'bVpRatio':   ('Book/Price (emp 0.09–3.75)', '↑ better (cheaper)', 'inverse P/B'),
    'tbVpRatio':  ('Tangible Book/Price (can be neg)', '↑ better (cheaper)', 'tangible inverse P/B'),
    'earnYield':  ('E/P (emp −0.02–0.12)', '↑ better (cheaper)', 'earnings yield'),
    'freeCashFlowYield':          ('TTM FCF/mcap', '↑ better', 'free-cash-flow yield'),
    'revenueGrowth':              ('mean YoY', '↑ better', 'revenue growth'),
    'freeCashFlowPerShareGrowth': ('mean YoY (unstable)', '↑ better', 'FCF/share growth'),
    # -- Moat components (Section E) -- ALL ↑ better; >0 = passes its threshold --
    'moat:FCFyield':    ('mean−thr (>10%)', '↑ better (>0 passes)', 'FCF yield vs 10% threshold'),
    'moat:GrossMargin': ('mean−thr (>30%)', '↑ better (>0 passes)', 'gross margin vs 30%'),
    'moat:RevtoASS':    ('mean−thr (>0.75)', '↑ better (>0 passes)', 'asset turnover vs 0.75'),
    'moat:RoE':         ('mean−thr (>15%)', '↑ better (>0 passes)', 'ROE vs 15%'),
    'moat:RoA':         ('mean−thr (>10%)', '↑ better (>0 passes)', 'ROA vs 10%'),
    'moat:ROIC':        ('mean−thr (>15%)', '↑ better (>0 passes)', 'ROIC vs 15%'),
    'moat:NetMargin':   ('mean−thr (>20%)', '↑ better (>0 passes)', 'net margin vs 20%'),
    'moat:SGAtoGP':     ('thr−ratio (SGA/GP<15%)', '↑ better (threshold−ratio; +=passes)',
                         'SG&A/gross-profit; stored as threshold−ratio, so higher = leaner'),
    'moat:DeptoGP':     ('thr−ratio (D&A/GP<10%)', '↑ better (threshold−ratio; +=passes)',
                         'D&A/gross-profit; stored as threshold−ratio'),
    'moat:CapExtoEarnings': ('thr−ratio (capex/earn<20%)', '↑ better (threshold−ratio; +=passes)',
                             'capex/earnings; stored as threshold−ratio'),
    'moat:TLtoEquity':  ('thr−ratio (TL/eq<0.8)', '↑ better (threshold−ratio; +=passes)',
                         'total-liabilities/equity; stored as threshold−ratio'),
}

# BENCHMARKS -- only metrics the spec marks `universal: yes` get a static tick + verdict chip.
# 'tick' is the reference line drawn on the peer bar (raw units == the bar marker's units);
# 'warn'/'good' are (op, value) predicates for the ⚠ / ✓ verdict (a metric with only 'warn'
# is binary: not-warned => ✓; with both, the middle band is neutral). 'suppress' names the
# cohorts where the rule is invalid (financials, REITs) -> no tick, no verdict there.
_FIN = FIN_COHORTS  # {'FinManager', 'BalanceSheetFin'}
BENCHMARKS = {
    'currentRatio': dict(tick=1.5, warn=('<', 1.5),
                         note='rule-of-thumb: <1.5 caution · <1.0 flag (non-financial)',
                         suppress=_FIN | {'REIT'}),
    'returnOnCapitalEmployed': dict(tick=0.10, warn=('<', 0.10), good=('>', 0.15),
                         note='rule-of-thumb: <10% weak · >15% strong (non-financial)',
                         suppress=_FIN),
    'interest_coverage': dict(tick=3.0, warn=('<', 3.0),
                         note='rule-of-thumb: <3× flag · <1.5× serious (non-financial)',
                         suppress=_FIN),
    'fcf_margin': dict(tick=0.0, warn=('<', 0.0),
                         note='rule-of-thumb: <0 caution (floor only; magnitude peer-only)',
                         suppress=_FIN),
    'sloan': dict(tick=0.10, warn=('>', 0.10),
                         note='rule-of-thumb: > 0.10 flag (high accruals); low/neg = cash-backed (non-financial)',
                         suppress=_FIN),
    # 'Altman-Z' DELIBERATELY ABSENT (2026-07-19 soundness audit): the pipeline's computed
    # variant deviates from published Altman-Z (wrong x2 variable, quarter-scale flow terms),
    # so 1.8/3.0 are not calibrated for the quantity we compute. No tick is drawn on its peer
    # bar and no verdict is derived from it -- the bar shows the PEER DISTRIBUTION ONLY
    # (relative-only). Do not restore a tick/threshold here without a recalibration decision.
    'Piotroski': dict(tick=3, warn=('<=', 3), good=('>=', 7),
                         note='rule-of-thumb: ≤3 weak · ≥7 strong'),
    # No peer bar for these -> inline verdict chip only (tick=None).
    'M-Score': dict(tick=None, warn=('>', 0.0),
                         note='rule-of-thumb: stored >0 flags (Beneish>−1.78); invalid financials',
                         suppress=_FIN),
    'C-Score': dict(tick=None, warn=('>=', 4),
                         note='rule-of-thumb: ≥4 flag · 0 clean; invalid financials',
                         suppress=_FIN),
}


def orient_chip(key):
    """Compact orientation pill for an index/score: 'scale · direction' with the one-line
    meaning as a tooltip. Returns '' for an unknown key (metric not in the spec table)."""
    o = ORIENTATION.get(key)
    if not o:
        return ''
    scale, direction, meaning = o
    return (f'<span class="orient-chip" title="{escape(meaning)}">'
            f'{escape(scale)} · {escape(direction)}</span>')


def _bench_cmp(v, cond):
    op, t = cond
    if op == '<':
        return v < t
    if op == '<=':
        return v <= t
    if op == '>':
        return v > t
    if op == '>=':
        return v >= t
    return False


def benchmark_tick(bkey, cohort_label=None):
    """The raw threshold to draw on the peer bar, or None when the metric has no universal
    benchmark / no tick / is suppressed for this cohort."""
    b = BENCHMARKS.get(bkey)
    if not b:
        return None
    if cohort_label is not None and cohort_label in b.get('suppress', set()):
        return None
    return b.get('tick')


def benchmark_verdict(bkey, value, cohort_label=None):
    """A small ✓ / ⚠ / • verdict chip comparing `value` to its universal benchmark, or '' when
    there is no universal benchmark, the value is missing, or it's suppressed for this cohort.
    ✓ = good side, ⚠ = flag side, • = neutral middle band."""
    b = BENCHMARKS.get(bkey)
    if not b:
        return ''
    if cohort_label is not None and cohort_label in b.get('suppress', set()):
        return ''
    v = safe_float(value)
    if np.isnan(v):
        return ''
    warn, good = b.get('warn'), b.get('good')
    if warn and _bench_cmp(v, warn):
        state, sym = 'warn', '⚠'
    elif good and _bench_cmp(v, good):
        state, sym = 'pass', '✓'
    elif good is None:
        state, sym = 'pass', '✓'          # binary rule (warn only): not warned => good side
    else:
        state, sym = 'neutral', '•'       # between weak and strong bands
    return (f'<span class="verdict-chip {state}" title="{escape(b["note"])}">{sym}</span>')


# ============================================================================
# 4-STATE VERDICT ICONS + CONFIDENCE-TAGGED SUSPICION FLAGS  (metric-icons-spec)
# ============================================================================
# 🟢 good · 🟡 tentative/borderline-or-low-confidence · 🔴 clearly bad side of a rule ·
# ⚪ no honest standalone rule (peer-relative only). Presentation-layer ONLY -- reads
# already-loaded page data; changes NO score/rank/marketCap/partition/currency value.
VERDICT_GLYPH = {'good': '🟢', 'neutral': '🟡', 'warn': '🔴', 'gray': '⚪'}

# A.1 dual/single-threshold rules. `good`='high'|'low' (which side is better); a value
# >= green (high) / <= green (low) is 🟢, past `red` is 🔴, the band between is 🟡. The
# raw thresholds encode the spec's A.1 table + Yellow-borderline bands directly.
VERDICT_RULES = {
    'currentRatio':            dict(good='high', green=1.5,  red=1.0,  note='rule-of-thumb: ≥1.5 healthy · <1.0 flag · 1.0–1.5 borderline (non-financial)', suppress=_FIN | {'REIT'}),
    'returnOnCapitalEmployed': dict(good='high', green=0.15, red=0.10, note='rule-of-thumb: >15% strong · <10% weak · 10–15% borderline (non-financial)', suppress=_FIN),
    'interest_coverage':       dict(good='high', green=3.3,  red=2.7,  note='rule-of-thumb: >3× ok · <3× flag · 2.7–3.3 borderline (non-financial)', suppress=_FIN),
    'sloan':                   dict(good='low',  green=0.09, red=0.11, note='rule-of-thumb: ≤0.10 cash-backed · >0.10 high-accrual flag · 0.09–0.11 borderline (non-financial)', suppress=_FIN),
    # Altman-Z DELIBERATELY ABSENT (2026-07-19 soundness audit): the computed variant
    # deviates from published Altman-Z, so 1.8/3.0 are not calibrated for it -> it is listed
    # in VERDICT_GRAY (⚪, relative-only) until the pipeline metric is fixed. Do not restore
    # a threshold verdict here without a recalibration decision.
    'Piotroski':               dict(good='high', green=7,    red=4,    note='rule-of-thumb: ≥7 strong · ≤3 weak · 4–6 middling'),
    'M-Score':                 dict(good='low',  green=-0.5, red=0.5,  note='rule-of-thumb: stored ≤0 clean · >0 flag · −0.5…+0.5 borderline (invalid financials)', suppress=_FIN),
    'C-Score':                 dict(good='low',  green=0.5,  red=3.5,  note='rule-of-thumb: 0 clean · ≥4 flag · 1–3 borderline (invalid financials)', suppress=_FIN),
    'incomeQuality':           dict(good='high', green=0.9,  red=0.7,  note='rule-of-thumb: ≥0.9 cash-backed · <0.7 flag · 0.7–0.9 borderline (non-financial)', suppress=_FIN),
}
# A.1 floor rules: 🔴 on the bad side, else ⚪ (no positive standalone rule).
VERDICT_FLOORS = {
    'fcf_margin':          dict(floor=0.0, bad='below', note='floor: <0 caution (TTM FCF margin negative)', suppress=_FIN),
    'freeCashFlowYield':   dict(floor=0.0, bad='below', note='floor: <0 caution (negative FCF yield)'),
    'earnYield':           dict(floor=0.0, bad='below', note='floor: <0 caution (negative earnings yield)'),
    'cash_conv':           dict(floor=0.0, bad='below', note='floor: <0 caution (FCF and net income disagree in sign)'),
    'affo_payout':         dict(floor=1.0, bad='above', note='floor: >1.0 caution (AFFO payout exceeds AFFO)'),
}
# A.2 gray metrics -> always ⚪ (no honest standalone rule; can still receive a 🚩).
VERDICT_GRAY = {
    # 'Altman-Z': gray by AUDIT DECISION (not because no rule exists in the literature) --
    # our computed variant is not the published statistic, so its thresholds are meaningless.
    'Altman-Z',
    'peRatio', 'pbRatio', 'bVpRatio', 'tbVpRatio', 'grossProfitMargin', 'op_margin',
    'effective_tax', 'dso', 'inv_days', 'returnOnEquity', 'RoA', 'revenueGrowth',
    'freeCashFlowPerShareGrowth', 'net_debt_ebitda', 'CycleHeat', 'AggScore', 'moatScore',
    'p_ffo', 'ffo_per_share', 'ltv', 'nav',
}
_MOAT_NEAR_ZERO = 0.02   # |stored (mean−threshold)| below this -> 🟡 "near its bar"

# Suspicion-flag confidence tiers (static per rule; "flag the flag").
FLAG_TIERS = {'R1': 'M', 'R2': 'H', 'R3': 'H', 'R4': 'M', 'R5': 'L', 'R6': 'M', 'R7': 'L'}
_TIER_RANK = {'H': 3, 'M': 2, 'L': 1}
_TIER_NAME = {'H': 'High', 'M': 'Medium', 'L': 'Low'}


def _verdict_state_band(rule, v):
    v = safe_float(v)
    if np.isnan(v):
        return None
    if rule['good'] == 'high':
        if v >= rule['green']:
            return 'good'
        if v < rule['red']:
            return 'warn'
        return 'neutral'
    else:  # lower is better
        if v <= rule['green']:
            return 'good'
        if v >= rule['red']:
            return 'warn'
        return 'neutral'


def compute_verdict(metric_key, value, cohort_label=None, low_conf=False):
    """4-state verdict for one metric. Precedence (spec Part C): (1) no rule / suppressed
    for cohort -> ⚪ 'gray'; (2) else Y2 low-confidence -> 🟡 'neutral'; (3) else value vs
    rule. Returns (state, note) where state in {good,neutral,warn,gray} or (None, None)
    when the metric takes NO icon. `low_conf` carries the caller's Y2 determination."""
    if metric_key in VERDICT_RULES:
        r = VERDICT_RULES[metric_key]
        if cohort_label is not None and cohort_label in r.get('suppress', set()):
            return 'gray', 'no universal rule for this cohort'
        if low_conf:
            return 'neutral', 'value present but flagged low-confidence (see 🚩/forensic)'
        st = _verdict_state_band(r, value)
        if st is None:
            return 'gray', 'value unavailable'
        return st, r['note']
    if metric_key in VERDICT_FLOORS:
        f = VERDICT_FLOORS[metric_key]
        if cohort_label is not None and cohort_label in f.get('suppress', set()):
            return 'gray', 'no universal rule for this cohort'
        if low_conf:
            return 'neutral', 'value present but flagged low-confidence (see 🚩/forensic)'
        v = safe_float(value)
        if np.isnan(v):
            return 'gray', 'value unavailable'
        bad = (v < f['floor']) if f['bad'] == 'below' else (v > f['floor'])
        return ('warn', f['note']) if bad else ('gray', 'no positive standalone rule (peer-relative)')
    if metric_key in VERDICT_GRAY:
        return 'gray', 'no honest standalone rule — read against the peer bar'
    if metric_key.startswith('moat:'):
        v = safe_float(value)
        if np.isnan(v):
            return 'gray', 'value unavailable'
        if v > _MOAT_NEAR_ZERO:
            return 'good', 'moat component passes its bar (stored mean−threshold > 0)'
        if v < -_MOAT_NEAR_ZERO:
            return 'warn', 'moat component fails its bar (stored mean−threshold < 0)'
        return 'neutral', 'moat component sits right at its bar (stored mean−threshold ≈ 0)'
    return None, None


def _clip01(v):
    v = safe_float(v)
    if np.isnan(v):
        return np.nan
    return min(max(v, 0.0), 1.0)


def _winsorize(arr, p=1.0):
    """Clip a 1-D array to its [p, 100-p] percentiles (NaNs already dropped)."""
    if arr.size == 0:
        return arr
    lo, hi = np.percentile(arr, [p, 100.0 - p])
    return np.clip(arr, lo, hi)


def _sloan_recomputed(g):
    """Sloan accruals over cdx_df: (TTM netIncome - TTM operating cash flow)/latest assets."""
    ni = ttm_sum(g, 'netIncome')
    cfo = ttm_sum(g, 'netCashProvidedByOperatingActivities')
    ta = latest_row_value(g, 'totalAssets')
    if np.isnan(ni) or np.isnan(cfo) or np.isnan(ta) or ta == 0:
        return np.nan
    return (ni - cfo) / ta


def _p_ffo_reducer(g):
    ffo_ps = compute_ffo_per_share(g)
    shares = latest_row_value(g, 'weightedAverageShsOut')
    mktcap = latest_row_value(g, 'marketCap')
    if np.isnan(ffo_ps) or np.isnan(shares) or np.isnan(mktcap) or ffo_ps * shares == 0:
        return np.nan
    return mktcap / (ffo_ps * shares)


def _affo_payout_reducer(g):
    div = abs(ttm_sum(g, 'dividendsPaid'))
    ffo = ttm_sum(g, 'netIncome') + ttm_sum(g, 'depreciationAndAmortization')
    capex = ttm_sum(g, 'netCashProvidedByOperatingActivities') - ttm_sum(g, 'freeCashFlow')
    denom = ffo - capex
    if np.isnan(div) or np.isnan(denom) or denom <= 0:
        return np.nan
    return div / denom


def _ext_reducer(g):
    """Per-source extended metrics, using the SAME helpers as the per-name markers so the
    pool value and the displayed marker are identical."""
    return pd.Series({
        'op_margin': compute_operating_margin(g),
        'fcf_margin': compute_fcf_margin_ttm(g),
        'interest_coverage': compute_interest_coverage(g),
        'net_debt_ebitda': latest_row_value(g, 'netDebtToEBITDA'),
        'effective_tax': _clip01(latest_row_value(g, 'effectiveTaxRate')),
        'dso': latest_row_value(g, 'daysSalesOutstanding'),
        'inv_days': latest_row_value(g, 'daysOfInventoryOutstanding'),
        'sloan': _sloan_recomputed(g),
        'p_ffo': _p_ffo_reducer(g),
        'affo_payout': _affo_payout_reducer(g),
        'ltv': compute_ltv_proxy(g),
    })


def build_cohort_membership(data):
    """Full carved + size-floored membership per cohort, reconstructed from carveout_labels
    (source->cohort) and the mcap floor. Reproduces the pipeline's n_<cohort> counts."""
    labels = data.get('carveout_labels')
    cdx = data.get('cdx_df')
    if labels is None or cdx is None or cdx.empty:
        # Graceful fallback: the shortlist membership (pools will be tiny -> min-N suppresses).
        carveout = data.get('carveout_sidelists', {})
        mem = {'general': list(data['postrank_df']['source'].unique())}
        for coh in COHORTS:
            cp = carveout.get(coh, {}).get('postRank', pd.DataFrame())
            if cp is not None and not cp.empty:
                mem[coh] = list(cp['source'].unique())
        return mem
    diag = data.get('carveout_diagnostics') or {}
    floor = diag.get('mcap_floor') or 0
    c = cdx[['source', 'date', 'marketCap']].copy()
    c['date'] = pd.to_datetime(c['date'], errors='coerce')
    c = c.sort_values(['source', 'date'], ascending=[True, False])
    latest_mcap = c.groupby('source', sort=False)['marketCap'].first()
    mem = {}
    for coh in ['general'] + COHORTS:
        names = labels[labels == coh].index.tolist()
        mem[coh] = [n for n in names if n in latest_mcap.index
                    and pd.notna(latest_mcap[n]) and latest_mcap[n] >= floor]
    return mem


def build_extended_pools(data):
    """Return (ext_per_src, ext_stats, membership).

    ext_per_src : DataFrame (source-indexed) of extended metrics for every cohort member
                  (cdx-derived + the non-dup moat components). Supplies markers.
    ext_stats   : {(cohort, metric): {n, p10, p50, p90, arr}} over the winsorized pool;
                  entries with n < MIN_POOL_N carry only {n} (render suppresses them).
    """
    try:
        membership = build_cohort_membership(data)
        cdx = data['cdx_df']
        cols = [c for c in _EXT_CDX_COLS if c in cdx.columns] + ['source']
        c = cdx[cols].copy()
        c['date'] = pd.to_datetime(c['date'], errors='coerce')
        c = c.sort_values(['source', 'date'], ascending=[True, False])
        all_members = set()
        for lst in membership.values():
            all_members |= set(lst)
        sub = c[c['source'].isin(all_members)]
        value_cols = [x for x in cols if x != 'source']
        per = sub.groupby('source', sort=False)[value_cols].apply(_ext_reducer)

        # Merge the non-dup moat components (markers + pools) from moatdf.
        moatdf = data.get('moatdf')
        if moatdf is not None and not moatdf.empty:
            mcols = [x for x in EXT_MOAT_COLS if x in moatdf.columns]
            mm = moatdf[['source'] + mcols].drop_duplicates('source').set_index('source')
            per = per.join(mm, how='left')

        metrics = list(per.columns)
        stats = {}
        for coh, members in membership.items():
            idx = [m for m in members if m in per.index]
            if not idx:
                continue
            block = per.loc[idx]
            for metric in metrics:
                arr = pd.to_numeric(block[metric], errors='coerce').replace(
                    [np.inf, -np.inf], np.nan).to_numpy(dtype='float64')
                nonnan = arr[~np.isnan(arr)]
                n = int(nonnan.size)
                if n == 0:
                    stats[(coh, metric)] = {'n': 0}
                    continue
                wins = _winsorize(nonnan)
                p10, p50, p90 = np.percentile(wins, [10, 50, 90])
                stats[(coh, metric)] = {'n': n, 'p10': float(p10), 'p50': float(p50),
                                        'p90': float(p90), 'arr': wins}
        return per, stats, membership
    except Exception as e:
        log.warning(f"Failed to build extended pools: {e}")
        import traceback
        traceback.print_exc()
        return None, {}, {}


# ============================================================================
# DATA LOADING & PROCESSING
# ============================================================================
def resolve_run_artifacts(run_dir, run_date=None):
    """Resolve ONE self-consistent run: pick a run-date from the postRank filename (the
    source of truth -- NOT mtime, which races during a Drive sync), then require the
    Boresults pickle and CSVs for that SAME date. Missing same-date artifact -> hard error,
    never a silent fall-back to another date's file (that mixing was the publish-blocker).

    Returns (run_date, postrank_file, boresults_file, aggscore_file, forensic_file).
    """
    run_dir = str(run_dir)
    postrank_by_date = {}
    for f in glob.glob(os.path.join(run_dir, "postRank_*.pickle")):
        m = re.search(r'postRank_(\d{4}-\d{2}-\d{2})_', os.path.basename(f))
        if m:
            postrank_by_date[m.group(1)] = f
    if not postrank_by_date:
        raise FileNotFoundError(f"No postRank_<date>_*.pickle found in {run_dir}")

    if run_date:
        if run_date not in postrank_by_date:
            raise FileNotFoundError(
                f"--run-date {run_date}: no postRank pickle for that date in {run_dir}. "
                f"Available run-dates: {sorted(postrank_by_date)}")
        chosen = run_date
    else:
        # Latest by the DATE IN THE FILENAME (ISO dates sort lexicographically), not mtime.
        chosen = max(postrank_by_date)
    postrank_file = postrank_by_date[chosen]

    def require(pattern, label):
        matches = glob.glob(os.path.join(run_dir, pattern.format(d=chosen)))
        if not matches:
            raise FileNotFoundError(
                f"Run-date {chosen}: required {label} not found "
                f"(pattern '{pattern.format(d=chosen)}'). Refusing to fall back to a "
                f"different date's file (prevents cross-run mixing). Ensure the {chosen} run "
                f"is fully synced to {run_dir}, or pass --run-date for a complete run.")
        return max(matches, key=os.path.getmtime)

    boresults_file = require("Boresults_dic-*_{d}_*.pickle", "Boresults pickle")
    aggscore_file = require("AggScoreTop100-{d}_*.csv", "AggScoreTop100 CSV")
    forensic_file = require("ForensicFlagsTop100-{d}_*.csv", "ForensicFlagsTop100 CSV")
    return chosen, postrank_file, boresults_file, aggscore_file, forensic_file


def load_run_data(run_dir, valuation_repo, run_date=None):
    """Load all run data from ONE date-consistent run (see resolve_run_artifacts)."""
    run_dir = Path(run_dir)

    log.info(f"Loading run artifacts from {run_dir}...")

    run_date, postrank_file, boresults_file, aggscore_file, forensic_file = \
        resolve_run_artifacts(run_dir, run_date=run_date)

    # One-line provenance log: single resolved run-date + exactly which files were loaded,
    # so any cross-run mixing would be visible and loud.
    log.info(f"Resolved run-date {run_date}; loading a single date-consistent run:")
    for label, f in [('postRank', postrank_file), ('Boresults', boresults_file),
                     ('AggScore', aggscore_file), ('ForensicFlags', forensic_file)]:
        log.info(f"    {label}: {os.path.basename(f)}")

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
    carveout_labels = boresults_dic.get('carveout_labels')
    carveout_diagnostics = boresults_dic.get('carveout_diagnostics')

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

    # Market-cap band partition (ADDITIVE size axis over the general pool). Recomputed
    # here from postrank_df + cdx_df so the offline presentation is SELF-CONTAINED and
    # does not depend on the pipeline having materialized it. Keys off the SAME
    # carveOut.MCAP_BANDS + shared FX/USD path as production selection and grading, so the
    # three consumers agree. Degrades gracefully: currency_pending -> all names read as
    # General, sub-bands empty (the render then labels/skips rather than misbanding).
    marketcap_bands_info = None
    try:
        sys.path.insert(0, str(valuation_repo))
        import carveOut as _co
        _bnames = (dict(zip(tickers_df['symbol'], tickers_df['name']))
                   if tickers_df is not None and 'symbol' in getattr(tickers_df, 'columns', [])
                      and 'name' in getattr(tickers_df, 'columns', []) else {})
        marketcap_bands_info = _co.partition_by_marketcap(postrank_df, cdx_df, _bnames)
    except Exception as _e:
        log.info(f"market-cap bands skipped: {type(_e).__name__}: {_e}")

    return {
        'run_date': run_date,
        'postrank_df': postrank_df,
        'cdx_df': cdx_df,
        'moatdf': moatdf,
        'tickers_df': tickers_df,
        'marketcap_bands_info': marketcap_bands_info,
        'carveout_sidelists': carveout_sidelists,
        'carveout_labels': carveout_labels,
        'carveout_diagnostics': carveout_diagnostics,
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
        # Extended peer-bar pools (metrics beyond the 16 playbook set), winsorized + min-N
        # gated + fin-suppressed. Separate from the 16-playbook machinery above.
        self.ext_per_src, self.ext_stats, self.ext_membership = build_extended_pools(data)
        self._eval_cache = {}   # (ticker,cohort) -> per-page verdict/flag evaluation
        self.html_parts = []

    def ext_val(self, ticker, metric):
        """Raw extended-metric value for a ticker (marker for its own bar)."""
        p = self.ext_per_src
        if p is None or ticker not in p.index or metric not in p.columns:
            return np.nan
        return safe_float(p.loc[ticker, metric])

    def ext_bar(self, cohort_label, metric, marker):
        """Trailing HTML for an extended-metric bar: percentile + dot-on-p10-p90 bar, or a
        gap tag (financial-suppressed / pool-too-small). Winsorized pool; percentile 0-100.
        `marker` is the RAW displayed value so the dot matches the number shown."""
        if cohort_label in FIN_COHORTS and metric in FIN_SUPPRESS:
            return '<span class="gap-inline">n/a for financials</span>'
        st = self.ext_stats.get((cohort_label, metric))
        if st is None:
            return '<span class="gap-inline">no cohort pool</span>'
        if st.get('n', 0) < MIN_POOL_N:
            return f'<span class="gap-inline">pool too small (n={st.get("n", 0)})</span>'
        if marker is None or np.isnan(safe_float(marker)):
            return ''
        pct = get_percentile_marker(safe_float(marker), st['arr'])
        tick = benchmark_tick(metric, cohort_label)
        bar = create_distribution_bar(safe_float(marker), st['p10'], st['p50'], st['p90'],
                                      benchmark=tick)
        # verdict now rendered as the 4-state icon at the VALUE (see _vf); bar keeps only
        # the percentile + spread (the old ✓/•/⚠ trailing chip is replaced there).
        return f'<span class="pctile">({pctile_format(pct)} pct)</span> {bar}'.rstrip()

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
        tick = benchmark_tick(pool_metric, cohort_label)
        bar = create_distribution_bar(safe_float(marker), p10, p50, p90, benchmark=tick)
        # verdict now rendered as the 4-state icon at the VALUE (see _vf); the old ✓/•/⚠
        # trailing chip is replaced there.
        return bar

    def _page_eval(self, ticker, cohort_label):
        """Cached per-page verdict/flag evaluation."""
        k = (ticker, cohort_label)
        if k not in self._eval_cache:
            self._eval_cache[k] = self.evaluate_page(ticker, cohort_label)
        return self._eval_cache[k]

    def _vf(self, ev, key):
        """Render ` {verdict-icon}{🚩+tier?}` for one metric from the page evaluation.
        Verdict icon (exactly one 4-state glyph) + an INDEPENDENT confidence-tagged flag.
        Returns '' when the metric takes neither (A.3 descriptive metrics)."""
        e = ev.get(key)
        if not e:
            return ''
        glyph, note, flag, tier, reason = e
        out = ''
        if glyph:
            out += (f'<span class="verdict-icon" title="{escape(note or "")}">{glyph}</span>')
        if flag:
            out += (f'<span class="flag-icon tier-{tier}" title="{escape(reason or "")}">'
                    f'🚩<sub>{tier}</sub></span>')
        return (' ' + out) if out else ''

    def evaluate_page(self, ticker, cohort_label):
        """Per-page metric evaluator (metric-icons-spec Part C). Returns
        {metric_key: (glyph, note, flag_bool, tier, reason)} for every metric that takes a
        verdict icon and/or a suspicion flag. Presentation-layer ONLY: reads already-loaded
        page data (cdx/aggscore/raw_all/ext/moatdf) and mutates nothing. Tiers static per rule."""
        from collections import defaultdict
        cdx = self.get_cdx_for_ticker(ticker)
        is_fin = cohort_label in FIN_COHORTS

        def L(c):
            return latest_row_value(cdx, c)
        # PAIRED TTM sums aligned on the SAME quarters (reviewer F4): R2 compares TTM net
        # income vs TTM operating cash flow, and Y2 compares TTM net income vs TTM revenue --
        # each pair must sum the same non-NaN period set, else a NaN quarter in one column
        # misaligns the comparison. (compute_cash_conversion, a DISPLAYED value, is left
        # untouched so no shown number moves.)
        ni_ttm, cfo_ttm = ttm_aligned_sums(cdx, ['netIncome', 'netCashProvidedByOperatingActivities'])
        ni_ttm_rev, rev_ttm = ttm_aligned_sums(cdx, ['netIncome', 'revenue'])
        roic = L('returnOnCapitalEmployed'); roe = L('returnOnEquity'); roa = L('returnOnAssets')
        gm = L('grossProfitMargin'); income_qual = L('incomeQuality')
        net_debt_ebitda = L('netDebtToEBITDA'); curr_ratio = L('currentRatio')
        op_margin = compute_operating_margin(cdx)
        fcf_margin = compute_fcf_margin_ttm(cdx)
        cash_conv = compute_cash_conversion(cdx)
        int_cov = compute_interest_coverage(cdx)
        fcf_yield = self.raw_metric(ticker, 'freeCashFlowYield')
        earn_yield = self.raw_metric(ticker, 'earnYield')
        altman_z = self.raw_metric(ticker, 'Altman-Z')
        rev_growth = self.raw_metric(ticker, 'revenueGrowth')
        piotroski = self.raw_metric(ticker, 'Piotroski')
        fcf_share_growth = self.raw_metric(ticker, 'freeCashFlowPerShareGrowth')
        sloan_val = self.ext_val(ticker, 'sloan')
        affo = self.ext_val(ticker, 'affo_payout')
        mk = self.markers.get(ticker, {})
        bvp = safe_float(mk.get('bVpRatio', np.nan)); tbvp = safe_float(mk.get('tbVpRatio', np.nan))
        moat = self.get_moat_components(ticker)
        tl_to_equity = safe_float(moat.get('TLtoEquity', np.nan))

        # aggscore forensic fields (Top-100 CSV; every rendered name is within Top-100).
        m_score = c_score = np.nan; forensic_valid = True; m_flag = False; m_drivers = ''
        agg = self.data.get('aggscore_df')
        if agg is not None and not agg.empty:
            ar = agg[agg['source'] == ticker]
            if not ar.empty:
                r0 = ar.iloc[0]
                m_score = safe_float(r0.get('M-Score')); c_score = safe_float(r0.get('C-Score'))
                fv = str(r0.get('forensicValid')).strip().lower()
                forensic_valid = fv not in ('false', '0', 'no', 'nan', 'none', '')
                m_flag = str(r0.get('M_flag_gt_-1.78')).strip().lower() in ('true', '1', 'yes')
                m_drivers = str(r0.get('M_drivers') or '')

        # dilution: shares +>10% over 3y (Section-H computation reused)
        dilution = False
        if 'weightedAverageShsOut' in cdx.columns:
            sh = cdx['weightedAverageShsOut'].dropna().values   # newest-first
            if len(sh) >= 12 and sh[-1] > 0 and (sh[0] / sh[-1] - 1) > 0.10:
                dilution = True

        # ---- Y2 low-confidence guards ----
        low_conf_forensic = not forensic_valid                   # -> M/C/Sloan 🟡
        denom_weak = False                                       # -> IncomeQuality/CashConv 🟡
        if not np.isnan(ni_ttm_rev):
            if ni_ttm_rev <= 0:
                denom_weak = True
            elif not np.isnan(rev_ttm) and rev_ttm != 0 and abs(ni_ttm_rev) / abs(rev_ttm) < 0.02:
                denom_weak = True

        # ---- verdicts (state, note) per metric_key ----
        verdicts = {
            'currentRatio':            compute_verdict('currentRatio', curr_ratio, cohort_label),
            'returnOnCapitalEmployed': compute_verdict('returnOnCapitalEmployed', roic, cohort_label),
            'interest_coverage':       compute_verdict('interest_coverage', int_cov, cohort_label),
            'sloan':                   compute_verdict('sloan', sloan_val, cohort_label, low_conf=low_conf_forensic),
            'Altman-Z':                compute_verdict('Altman-Z', altman_z, cohort_label),
            'Piotroski':               compute_verdict('Piotroski', piotroski, cohort_label),
            'M-Score':                 compute_verdict('M-Score', m_score, cohort_label, low_conf=low_conf_forensic),
            'C-Score':                 compute_verdict('C-Score', c_score, cohort_label, low_conf=low_conf_forensic),
            'incomeQuality':           compute_verdict('incomeQuality', income_qual, cohort_label, low_conf=denom_weak),
            'fcf_margin':              compute_verdict('fcf_margin', fcf_margin, cohort_label),
            'freeCashFlowYield':       compute_verdict('freeCashFlowYield', fcf_yield, cohort_label),
            'earnYield':               compute_verdict('earnYield', earn_yield, cohort_label),
            'cash_conv':               compute_verdict('cash_conv', cash_conv, cohort_label, low_conf=denom_weak),
            'affo_payout':             compute_verdict('affo_payout', affo, cohort_label),
            # A.2 gray metrics (always ⚪; may still receive a 🚩)
            'returnOnEquity':          compute_verdict('returnOnEquity', roe, cohort_label),
            'RoA':                     compute_verdict('RoA', roa, cohort_label),
            'grossProfitMargin':       compute_verdict('grossProfitMargin', gm, cohort_label),
            'op_margin':               compute_verdict('op_margin', op_margin, cohort_label),
            'revenueGrowth':           compute_verdict('revenueGrowth', rev_growth, cohort_label),
            'freeCashFlowPerShareGrowth': compute_verdict('freeCashFlowPerShareGrowth', fcf_share_growth, cohort_label),
            'net_debt_ebitda':         compute_verdict('net_debt_ebitda', net_debt_ebitda, cohort_label),
            'CycleHeat':               compute_verdict('CycleHeat', self.raw_metric(ticker, 'CycleHeat'), cohort_label),
            'moatScore':               compute_verdict('moatScore', mk.get('moatScore', np.nan), cohort_label),
            'bVpRatio':                compute_verdict('bVpRatio', bvp, cohort_label),
            'tbVpRatio':               compute_verdict('tbVpRatio', tbvp, cohort_label),
            'peRatio':                 compute_verdict('peRatio', np.nan, cohort_label),
        }
        for mc in EXT_MOAT_COLS + ['FCFyield', 'GrossMargin', 'RevtoASS', 'RoE', 'RoA', 'ROIC', 'NetMargin']:
            verdicts['moat:' + mc] = compute_verdict('moat:' + mc, moat.get(mc, np.nan), cohort_label)

        # ---- suspicion flags R1–R7 (all ship). Tier is static per rule EXCEPT R3, whose
        # tier is conditional on which limb fired (see below) -> `tier` override. ----
        flags = defaultdict(list)
        # Valuation metrics carry NO standalone verdict (⚪) but INHERIT earnings-quality /
        # forensic flags (CEO: a cheap number is suspicious when another is low). Added to
        # R1/R2/R3 target lists only (not R4/R5/R6).
        VAL_KEYS = ['peRatio', 'bVpRatio', 'earnYield', 'freeCashFlowYield']
        def add(rule_id, metrics, reason, tier=None):
            t = tier or FLAG_TIERS[rule_id]
            for m in metrics:
                flags[m].append((rule_id, t, reason))

        # R2 HIGH — profit without cash
        if (not np.isnan(ni_ttm) and ni_ttm > 0) and (not np.isnan(cfo_ttm) and cfo_ttm < 0):
            add('R2', ['incomeQuality', 'returnOnCapitalEmployed', 'returnOnEquity', 'grossProfitMargin']
                + VAL_KEYS,
                f"Profit without cash: TTM net income {ni_ttm:,.0f} > 0 but TTM operating cash flow "
                f"{cfo_ttm:,.0f} < 0 — earnings are not turning into cash.")
        # R3 — forensic contradiction on a pick (suppress FIN). M-Score / C-Score limbs ONLY.
        # The Altman-Z limb was REMOVED (2026-07-19 soundness audit): our computed Z deviates
        # from published Altman-Z, so a "<1.8" trip is not calibrated evidence of distress.
        # Both remaining limbs are HIGH-tier (the earlier Z-only -> Medium path is gone with it).
        if not is_fin:
            trips = []
            if not np.isnan(m_score) and m_score > 0: trips.append(f"M-Score {m_score:.2f} > 0")
            if not np.isnan(c_score) and c_score >= 4: trips.append(f"C-Score {c_score:.0f} ≥ 4")
            if trips:
                add('R3', ['moatScore', 'returnOnCapitalEmployed', 'grossProfitMargin'] + VAL_KEYS,
                    "Forensic contradiction on a pick: " + "; ".join(trips)
                    + " — the forensic layer disagrees with this pick.")
        # R1 MEDIUM — accrual-backed profitability
        prof_good = ((not np.isnan(roic) and roic > 0.15) or (not np.isnan(roe) and roe > 0.15)
                     or (not np.isnan(gm) and gm > 0.40) or (not np.isnan(op_margin) and op_margin > 0.20))
        acc_bits = []
        if not np.isnan(sloan_val) and sloan_val > 0.10: acc_bits.append(f"Sloan {sloan_val:.2f} > 0.10")
        if not np.isnan(cash_conv) and cash_conv < 0.5: acc_bits.append(f"cash-conversion {cash_conv:.2f} < 0.5")
        if m_flag: acc_bits.append("Beneish M-flag fired")
        if prof_good and acc_bits:
            add('R1', ['returnOnCapitalEmployed', 'returnOnEquity', 'grossProfitMargin', 'incomeQuality']
                + VAL_KEYS,
                "Accrual-backed profitability: strong profitability alongside " + "; ".join(acc_bits)
                + " — the returns may not be cash-backed.")
        # R4 MEDIUM — leverage-inflated ROE (suppress FIN)
        if not is_fin and not np.isnan(roe) and roe > 0.15:
            lev = []
            if not np.isnan(tl_to_equity) and tl_to_equity < 0:
                lev.append(f"TL/Equity fails its 0.8 bar (stored {tl_to_equity:.2f})")
            if not np.isnan(net_debt_ebitda) and net_debt_ebitda > 4:
                lev.append(f"Net Debt/EBITDA {net_debt_ebitda:.1f}×")
            if lev:
                add('R4', ['returnOnEquity'],
                    "ROE looks strong but is leverage-inflated: " + " and ".join(lev)
                    + " — the equity multiplier can inflate ROE without operating quality.")
        # R6 MEDIUM — dilution-masked per-share strength
        if dilution and ((not np.isnan(fcf_share_growth) and fcf_share_growth > 0)
                         or (not np.isnan(bvp) and bvp > 0)):
            add('R6', ['freeCashFlowPerShareGrowth', 'bVpRatio', 'tbVpRatio'],
                "Dilution-masked per-share strength: shares up > 10% over 3y, so per-share "
                "growth/value understates issuance — check totals, not per-share figures.")
        # R5 LOW — growth–accrual divergence
        strong_growth = not np.isnan(rev_growth) and rev_growth > 0.15
        mdriver_hit = any(tok in m_drivers.upper() for tok in ('DSRI', 'SGI', 'AQI'))
        sloan_hi = not np.isnan(sloan_val) and sloan_val > 0.10
        if strong_growth and (sloan_hi or mdriver_hit):
            why = "Sloan > 0.10" if sloan_hi else "Beneish drivers (DSRI/SGI/AQI) elevated"
            add('R5', ['revenueGrowth', 'freeCashFlowPerShareGrowth'],
                f"Growth–accrual divergence: strong revenue growth {rev_growth:.0%} with {why} "
                "— growth may be accrual-driven rather than cash-generative.")
        # R7 LOW — 'too good' one-off spike vs own history
        def _spike(col):
            if col not in cdx.columns:
                return False
            s = cdx[col].dropna().values     # newest-first
            if len(s) < 9:
                return False
            latest, prior = s[0], s[1:9]
            if latest <= np.max(prior):
                return False
            hist = s[1:]
            q1, q3 = np.percentile(hist, [25, 75]); iqr = q3 - q1
            return iqr > 0 and latest > (np.median(hist) + 3 * iqr)
        for col, key in [('grossProfitMargin', 'grossProfitMargin'),
                         ('returnOnCapitalEmployed', 'returnOnCapitalEmployed'),
                         ('returnOnEquity', 'returnOnEquity'), ('incomeQuality', 'incomeQuality')]:
            if _spike(col):
                add('R7', [key],
                    f"'Too good' one-off spike: latest {key} exceeds both its prior-8-quarter max "
                    "and median + 3×IQR of its own history — likely non-recurring.")

        # ---- merge verdicts + flags ----
        result = {}
        for k in set(verdicts) | set(flags):
            state, note = verdicts.get(k, (None, None))
            glyph = VERDICT_GLYPH.get(state) if state else None
            fl = flags.get(k, [])
            if fl:
                tier = max(fl, key=lambda x: _TIER_RANK[x[1]])[1]
                reason = " · ".join(sorted({f"{rid} [{_TIER_NAME[t]}]: {rs}" for rid, t, rs in fl}))
                result[k] = (glyph, note, True, tier, reason)
            else:
                result[k] = (glyph, note, False, None, None)
        return result

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
                <span><strong>AggScore:</strong> {ratio_format(agg_score, 1)} {orient_chip('AggScore')}</span>
                <span><strong>Rank:</strong> {rank} <span class="pctile">(1 = top pick)</span></span>
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
                <span class="score-item"><strong>AggScore:</strong> {ratio_format(agg_score, 1)} {orient_chip('AggScore')}</span>
                <span class="score-item"><strong>MoatScore:</strong> {ratio_format(moat_score, 1)} {orient_chip('moatScore')}</span>
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
        ev = self._page_eval(ticker, cohort_label)

        if cohort_label == 'REIT':
            # REIT-specific metrics (all from cdx_df raw values, not z-scores)
            ffo_per_share = compute_ffo_per_share(cdx_df)
            # P/FFO = marketCap / TTM FFO (where FFO = ffo_per_share * shares)
            shares = latest_row_value(cdx_df, 'weightedAverageShsOut')
            ffo_total = ffo_per_share * shares if ffo_per_share and shares and not np.isnan(ffo_per_share) and not np.isnan(shares) else np.nan
            p_ffo = mktcap / ffo_total if mktcap and ffo_total and not np.isnan(mktcap) and not np.isnan(ffo_total) else np.nan
            ltv = compute_ltv_proxy(cdx_df)
            pb_ratio = latest_row_value(cdx_df, 'pbRatio')
            # REIT-cohort peer bars (winsorized, min-N; REIT n=245). FFO/share is per-share
            # currency -> no bar; NAV/P-B uses the canonical bVpRatio bar in Section G (rule 4).
            pffo_bar = self.ext_bar(cohort_label, 'p_ffo', p_ffo)
            ltv_bar = self.ext_bar(cohort_label, 'ltv', ltv)
            affo = self.ext_val(ticker, 'affo_payout')
            affo_bar = self.ext_bar(cohort_label, 'affo_payout', affo)

            html = f"""
            <div class="section-c valuation">
                <h3>Valuation Ratios (REIT)</h3>
                <table class="metrics-table">
                    <tr><td><strong>FFO/Share (proxy)</strong></td><td>{ratio_format(ffo_per_share)}</td><td>TTM (per-share, no bar)</td></tr>
                    <tr><td><strong>P/FFO</strong></td><td>{ratio_format(p_ffo)} {pffo_bar}</td><td>(proxy)</td></tr>
                    <tr><td><strong>AFFO Payout (proxy)</strong></td><td>{pct_format(affo)}{self._vf(ev, 'affo_payout')} {affo_bar}</td><td>|div|/(FFO−capex)</td></tr>
                    <tr><td><strong>LTV (proxy)</strong></td><td>{pct_format(ltv)} {ltv_bar}</td><td>debt/assets</td></tr>
                    <tr><td><strong>NAV Disc/Prem (proxy)</strong></td><td>{ratio_format(pb_ratio)}</td><td>P/B proxy (see B/P bar, §G)</td></tr>
                </table>
                <div class="gap-note">[cap-rate, occupancy, WALE not obtainable from filter data]</div>
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
            nde_bar = self.ext_bar(cohort_label, 'net_debt_ebitda', net_debt_ebitda)

            html = f"""
            <div class="section-c valuation">
                <h3>Valuation Ratios (Mining)</h3>
                <table class="metrics-table">
                    <tr><td><strong>Net Debt/EBITDA</strong></td><td>{ratio_format(net_debt_ebitda)} {nde_bar}</td><td>latest · trailing EBITDA distorts cyclicals</td></tr>
                    <tr><td><strong>Free Cash Flow</strong></td><td>{ratio_format(fcf)}</td><td>latest Q (currency, no bar)</td></tr>
                    <tr><td><strong>CycleHeat</strong> {orient_chip('CycleHeat')}</td><td>{ratio_format(cycleheat)} {ch_bar}</td><td>strong signal</td></tr>
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
            opm_bar = self.ext_bar(cohort_label, 'op_margin', op_margin)  # -> n/a for financials
            eff_tax = self.ext_val(ticker, 'effective_tax')
            efftax_bar = self.ext_bar(cohort_label, 'effective_tax', eff_tax)

            html = f"""
            <div class="section-c valuation">
                <h3>Valuation Ratios ({'Bank' if cohort_label == 'BalanceSheetFin' else 'FinManager'})</h3>
                <table class="metrics-table">
                    <tr><td><strong>P/B</strong></td><td>{ratio_format(pb)}</td><td>latest (see B/P bar, §G)</td></tr>
                    <tr><td><strong>ROE</strong> {orient_chip('returnOnEquity')}</td><td>{pct_format(roe)} {roe_bar}</td><td>latest</td></tr>
                    <tr><td><strong>ROA</strong> {orient_chip('RoA')}</td><td>{pct_format(roa)} {roa_bar}</td><td>latest</td></tr>
                    <tr><td><strong>Op Margin</strong></td><td>{pct_format(op_margin)} {opm_bar}</td><td>TTM</td></tr>
                    <tr><td><strong>Effective Tax</strong></td><td>{pct_format(eff_tax)} {efftax_bar}</td><td>latest, clip[0,1]</td></tr>
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
                    <tr><td><strong>ROE</strong> {orient_chip('returnOnEquity')}</td><td>{pct_format(roe)} {roe_bar}</td><td>latest</td></tr>
                    <tr><td><strong>ROIC</strong> {orient_chip('returnOnCapitalEmployed')}</td><td>{ratio_format(roic)} {roic_bar}</td><td>latest</td></tr>
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
            # Extended peer bars (winsorized pool, min-N gated, fin-suppressed).
            opm_bar = self.ext_bar(cohort_label, 'op_margin', op_margin)
            fcfm_bar = self.ext_bar(cohort_label, 'fcf_margin', fcf_margin)
            intcov_bar = self.ext_bar(cohort_label, 'interest_coverage', int_cov)
            nde_bar = self.ext_bar(cohort_label, 'net_debt_ebitda', net_debt_ebitda)
            eff_tax = self.ext_val(ticker, 'effective_tax')
            dso = self.ext_val(ticker, 'dso')
            inv_days = self.ext_val(ticker, 'inv_days')
            efftax_bar = self.ext_bar(cohort_label, 'effective_tax', eff_tax)
            dso_bar = self.ext_bar(cohort_label, 'dso', dso)
            invd_bar = self.ext_bar(cohort_label, 'inv_days', inv_days)

            html = f"""
            <div class="section-c valuation">
                <h3>Valuation Ratios</h3>
                <table class="metrics-table">
                    <tr><td><strong>ROIC/ROCE</strong> {orient_chip('returnOnCapitalEmployed')}</td><td>{ratio_format(roic)}{self._vf(ev, 'returnOnCapitalEmployed')} {roic_bar}</td><td>proxy</td></tr>
                    <tr><td><strong>Gross Margin</strong> {orient_chip('grossProfitMargin')}</td><td>{pct_format(gm)}{self._vf(ev, 'grossProfitMargin')} {gm_bar}</td><td>latest</td></tr>
                    <tr><td><strong>Op Margin</strong></td><td>{pct_format(op_margin)}{self._vf(ev, 'op_margin')} {opm_bar}</td><td>TTM</td></tr>
                    <tr><td><strong>FCF Margin</strong></td><td>{pct_format(fcf_margin)}{self._vf(ev, 'fcf_margin')} {fcfm_bar}</td><td>TTM</td></tr>
                    <tr><td><strong>Cash Conversion</strong></td><td>{ratio_format(cash_conv)}{self._vf(ev, 'cash_conv')}</td><td>TTM FCF / NI</td></tr>
                    <tr><td><strong>Income Quality</strong> {orient_chip('incomeQuality')}</td><td>{ratio_format(income_qual)}{self._vf(ev, 'incomeQuality')} {iq_bar}</td><td>audit</td></tr>
                    <tr><td><strong>Net Debt/EBITDA</strong></td><td>{ratio_format(net_debt_ebitda)}{self._vf(ev, 'net_debt_ebitda')} {nde_bar}</td><td>latest</td></tr>
                    <tr><td><strong>Interest Coverage</strong></td><td>{ratio_format(int_cov)}{self._vf(ev, 'interest_coverage')} {intcov_bar}</td><td>op inc / int exp</td></tr>
                    <tr><td><strong>Effective Tax</strong></td><td>{pct_format(eff_tax)} {efftax_bar}</td><td>latest, clip[0,1]</td></tr>
                    <tr><td><strong>Days Sales Outstanding</strong></td><td>{ratio_format(dso)} {dso_bar}</td><td>latest</td></tr>
                    <tr><td><strong>Inventory Days</strong></td><td>{ratio_format(inv_days)} {invd_bar}</td><td>goods cohorts</td></tr>
                    <tr><td><strong>P/E</strong></td><td>{ratio_format(pe_ratio)}{self._vf(ev, 'peRatio')}</td><td>traded or yield inv</td></tr>
                    <tr><td><strong>FCF Yield</strong> {orient_chip('freeCashFlowYield')}</td><td>{pct_format(fcf_yield)}{self._vf(ev, 'freeCashFlowYield')} {fcfy_bar}</td><td>reviewRef raw (TTM)</td></tr>
                </table>
                <div class="gap-note">[WACC, EV/EBIT not obtainable from filter data; P/E withheld from peer bar — use earnings-yield bar in Section G]</div>
            </div>
            """
            return html

    def section_d_trends(self, ticker):
        """Section D: Multi-year trend charts (all 12 metrics per spec)."""
        cdx = self.get_cdx_for_ticker(ticker)
        if cdx.empty:
            return '<div class="section-d"><p>[No quarterly data available]</p></div>'

        # Get time-series data (oldest -> newest for charting)
        cdx = cdx.sort_values('date') if 'date' in cdx.columns else cdx
        has_dates = 'date' in cdx.columns

        # Helper to get a trend series AND its per-quarter dates (only the rows where the
        # metric is present), oldest->newest. The dates drive each chart's x-axis span.
        def get_trend(col):
            if col not in cdx.columns:
                return None
            if has_dates:
                sub = cdx[['date', col]].dropna(subset=[col])
                if sub.empty:
                    return None
                return sub[col].values.tolist(), sub['date'].tolist()
            vals = cdx[col].dropna().values.tolist()
            return (vals, []) if vals else None

        # Computed operating-margin trend, with its own dates.
        om_vals, om_dates = [], []
        for _, r in cdx.iterrows():
            if pd.notna(r.get('revenue')) and pd.notna(r.get('operatingIncome')) and r.get('revenue') != 0:
                om_vals.append(r['operatingIncome'] / r['revenue'])
                if has_dates:
                    om_dates.append(r['date'])
        op_margin_trend = (om_vals, om_dates) if om_vals else None

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

        sparklines = []
        for name, trend in metrics:
            if not trend:
                continue
            values, dates = trend
            spark = create_sparkline_svg(values)
            if dates:
                start = f'<span class="axis-start">{quarter_label(dates[0])}</span>'
                end = f'<span class="axis-end">{quarter_label(dates[-1])}</span>'
                span = span_caption(dates)
                span_html = f'<span class="axis-span">{span}</span>' if span else ''
            else:
                start = end = span_html = ''
            sparklines.append(
                f'<div class="trend-row"><span class="trend-label"><strong>{name}:</strong></span>'
                f'{start}{spark}{end}{span_html}</div>')

        html = f"""
        <div class="section-d trends">
            <h3>Multi-Year Trends (Quarterly)</h3>
            {''.join(sparklines) if sparklines else '<p>[No trend data]</p>'}
        </div>
        """
        return html

    def section_e_moat(self, ticker, cohort_label='general'):
        """Section E: Moat checklist. Adds peer bars for the NON-duplicate moat components
        (rule 4); the duplicate ones are covered by canonical cdx/Section-G bars."""
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
            <p><strong>Moat Score:</strong> {ratio_format(moat_score)} {orient_chip('moatScore')}</p>
            <table class="moat-components">
        """

        # Duplicate-of-canonical components (rule 4) render as plain values (bars live on the
        # cdx/Section-G side); the 6 non-dup components get peer bars. Each carries a 4-state
        # verdict icon: moat comps are stored as (mean−threshold), so >0 literally = passed.
        ev = self._page_eval(ticker, cohort_label)
        dedup = {'FCFyield', 'GrossMargin', 'RoE', 'RoA', 'ROIC'}
        for metric in ['FCFyield', 'GrossMargin', 'RevtoASS', 'RoE', 'RoA', 'ROIC',
                       'SGAtoGP', 'DeptoGP', 'NetMargin', 'CapExtoEarnings', 'TLtoEquity']:
            val = moat_comp.get(metric, np.nan)
            chip = orient_chip('moat:' + metric)
            vf = self._vf(ev, 'moat:' + metric)
            if metric in EXT_MOAT_COLS:
                bar = self.ext_bar(cohort_label, metric, val)
                html += f"<tr><td>{metric}: {chip}</td><td>{ratio_format(val)}{vf} {bar}</td></tr>"
            else:
                note = ' <span class="pctile">(see §G/C bar)</span>' if metric in dedup else ''
                html += f"<tr><td>{metric}: {chip}</td><td>{ratio_format(val)}{vf}{note}</td></tr>"

        html += f"""
            </table>
            <div class="editable">
                <label>Moat verdict: ____________________________________________</label>
            </div>
        </div>
        """
        return html

    def section_f_forensic(self, ticker, cohort_label='general'):
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

        # Sloan accruals RECOMPUTED over the full cohort membership (the as-saved CSV value
        # above pools only the shortlist, so it gets no bar). Winsorized, min-N gated.
        sloan_cohort = self.ext_val(ticker, 'sloan')
        sloan_bar = self.ext_bar(cohort_label, 'sloan', sloan_cohort)

        # 4-state verdict icons (+ any confidence-tagged flag); Y2 low-confidence (forensicValid
        # == False) forces M-Score/C-Score/Sloan to 🟡 inside the evaluator.
        ev = self._page_eval(ticker, cohort_label)
        mscore_v = self._vf(ev, 'M-Score')
        cscore_v = self._vf(ev, 'C-Score')
        sloan_vf = self._vf(ev, 'sloan')
        iq_vf = self._vf(ev, 'incomeQuality')

        html = f"""
        <div class="section-f forensic">
            <h3>Forensic / Accounting Quality</h3>
            <table class="forensic-table">
                <tr><td><strong>M-Score</strong> {orient_chip('M-Score')}</td><td>{m_score}{mscore_v}</td></tr>
                <tr><td><strong>C-Score</strong> {orient_chip('C-Score')}</td><td>{c_score}{cscore_v}</td></tr>
                <tr><td><strong>Sloan Accruals (shortlist CSV)</strong> {orient_chip('sloanAccruals')}</td><td>{sloan}</td></tr>
                <tr><td><strong>Sloan Accruals (cohort peer)</strong> {orient_chip('sloanAccruals')}</td><td>{ratio_format(sloan_cohort)}{sloan_vf} {sloan_bar}</td></tr>
                <tr><td><strong>Income Quality</strong> {orient_chip('incomeQuality')}</td><td>{ratio_format(income_qual)}{iq_vf}</td></tr>
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
        ev = self._page_eval(ticker, cohort_label)

        html = f"""
        <div class="section-g cohort">
            <h3>Peer Distribution (Cohort: {cohort_label})</h3>
            <div class="cohort-metrics">
        """

        # Raw marker (cdx-latest, or reviewReference raw pool for FCF yield) + 4-state verdict
        # icon + optional confidence-tagged flag + percentile + p10-p50-p90 spread bar.
        for label, _cdx_col, pool_metric, fmt in SECTION_G_METRICS:
            marker = ticker_markers.get(pool_metric, np.nan)
            val_str = pct_format(marker) if fmt == 'pct' else ratio_format(marker)
            vf = self._vf(ev, pool_metric)
            chip = orient_chip(pool_metric)
            # Financial-cohort suppression (rule 3): gross margin is meaningless for
            # banks/asset managers -> show the value, drop the bar/percentile.
            if pool_metric == 'grossProfitMargin' and cohort_label in FIN_COHORTS:
                html += (f'<div><strong>{label}:</strong> {val_str}{vf} {chip} '
                         f'<span class="gap-inline">n/a for financials</span></div>')
                continue
            pct = ticker_percentiles.get(pool_metric, np.nan)
            bar = self.dist_bar(ticker, cohort_label, pool_metric, marker=marker)
            html += (f'<div><strong>{label}:</strong> {val_str}{vf} {chip} '
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

        # Solvency check REMOVED (2026-07-19 soundness audit): it rested on the computed
        # Altman-Z being < 1.8, but the pipeline's variant deviates from published Altman-Z
        # (wrong x2 variable, quarter-scale flow terms), so the 1.8 cut is NOT calibrated for
        # the quantity we compute -- a "Solvency Risk (Z<1.8)" banner was an uncalibrated
        # verdict. Altman-Z is now presented relative-only (⚪ verdict, no tick on its peer
        # bar). The interest-coverage and leverage checks below remain as the solvency signals
        # that do NOT depend on it. Do not reinstate a Z-based solvency flag without a
        # recalibration decision.

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
        html += self.section_e_moat(ticker, cohort_label)
        html += self.section_f_forensic(ticker, cohort_label)
        html += self.section_g_cohort(ticker, cohort_label)
        html += self.section_h_flags(ticker)

        html += """
        </div>
        <hr class="page-break">
        """
        return html

    # Human-readable band titles (USD market-cap size axis over the general pool).
    _BAND_TITLES = {
        'General':       'General (>$300M)',
        'Mid_150_300M':  'Mid ($150–300M)',
        'Small_50_150M': 'Small ($50–150M)',
        'Micro_lt_50M':  'Micro (<$50M)',
    }

    def build_html(self):
        """Build complete HTML document.

        LAYOUT (2026-07-18): the market-cap BANDS are the PRIMARY structure -- the main
        content and the sidebar nav LEAD with General (>$300M) top-20, then Mid / Small /
        Micro, each name's deep-dive rendered EXACTLY ONCE under its band. There is NO
        separate flat unbanded top-20 section (that produced a confusing "flat top-20 +
        appendix bands" shape with duplicated deep-dives). The 5 sector carve cohorts stay
        orthogonal and unchanged. This is a presentation-layout change ONLY: no score,
        rank, market-cap value, band partition, or currency handling is touched. When
        currency data is PENDING (or bands are unavailable), it degrades to the original
        single flat general top-20 with a pending note -- never the broken half-banded shape.
        """
        run_date = self.data['run_date']
        postrank_df = self.data['postrank_df']

        # Sector carve cohorts (orthogonal size-independent axis; unchanged).
        cohort_names = {}
        carveout_sidelists = self.data.get('carveout_sidelists', {}) or {}
        for cohort_label in COHORTS:
            cohort_dic = carveout_sidelists.get(cohort_label, {}) or {}
            cohort_postrank = cohort_dic.get('postRank', pd.DataFrame())
            if cohort_postrank is not None and not cohort_postrank.empty:
                cohort_names[cohort_label] = cohort_postrank.head(5)['source'].tolist()

        # Resolve the band partition (already computed + reviewed upstream).
        band_info = self.data.get('marketcap_bands_info')
        try:
            import carveOut as _co
            band_defs = _co.MCAP_BANDS
        except Exception:
            band_defs = []
        banded = bool(band_info and band_info.get('bands')
                      and not band_info.get('currency_pending', True) and band_defs)

        # Per-band rendered rosters (MCAP_BANDS order: General, Mid, Small, Micro).
        band_names = {}
        if banded:
            for label, lo, hi, N in band_defs:
                bdf = band_info['bands'].get(label)
                if bdf is not None and not bdf.empty:
                    band_names[label] = bdf.head(N)['source'].tolist()

        content = """<div class="content">""" + self._icon_legend()
        if banded:
            # PRIMARY: banded partition. Each general-pool name renders once, under its
            # band. cohort_label stays 'general' so cohort-percentile / valuation lookups
            # use the general pool (the band is a size grouping, NOT a scoring cohort).
            nav_html = self._build_nav_banded(band_names, cohort_names)
            content += '<div class="cohort-section"><h1>Market-cap bands</h1></div>\n'
            for label, lo, hi, N in band_defs:
                tickers = band_names.get(label)
                if not tickers:
                    continue
                title = escape(self._BAND_TITLES.get(label, label))
                content += (f'<div class="cohort-section"><h2>Band: {title} '
                            f'(top-{N})</h2></div>\n')
                for i, ticker in enumerate(tickers, 1):
                    content += self.build_name_page(ticker, i, 'general')
        else:
            # FALLBACK (pending currency / no band data): the original single flat general
            # top-20, with a note. NEVER the confusing flat-top-20 + appendix-bands shape.
            top_20 = postrank_df.head(20)['source'].tolist()
            nav_html = self._build_nav(top_20, cohort_names)
            pend = band_info.get('currency_pending', True) if band_info else True
            note = (' — market-cap bands pending currency data (correct from next full run)'
                    if pend else '')
            content += f'<div class="cohort-section"><h1>General Top-20{note}</h1></div>\n'
            for i, ticker in enumerate(top_20, 1):
                content += self.build_name_page(ticker, i, 'general')

        # Sector cohort sections (orthogonal; unchanged).
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

    def _icon_legend(self):
        """Small self-documenting legend for the verdict icons + flag confidence tiers."""
        return (
            '<div class="icon-legend">'
            '<strong>Legend —</strong> '
            '<span class="leg">🟢 good</span>'
            '<span class="leg">🟡 borderline / low-confidence</span>'
            '<span class="leg">🔴 bad side of a rule</span>'
            '<span class="leg">⚪ no standalone rule (peer-relative)</span>'
            '<span class="leg-sep">·</span>'
            '<span class="leg">🚩<sub>H</sub> High — treat as fact, investigate before buying</span>'
            '<span class="leg">🚩<sub>M</sub> Medium — real signal, needs business context</span>'
            '<span class="leg">🚩<sub>L</sub> Low — a prompt to look, not a verdict</span>'
            '<div class="leg-note">The 🚩 is independent of the verdict icon — a 🟢🚩 or ⚪🚩 '
            'is valid (a good/no-rule number can still be flagged). Hover any icon for the rule '
            'and, for a flag, the mechanism + offending metric + values + tier + rule id.</div>'
            '</div>')

    def _build_nav_banded(self, band_tickers, cohort_tickers):
        """Sidebar nav with market-cap BANDS as the primary structure (General / Mid /
        Small / Micro, each listing its members), then the 5 sector cohort sections
        unchanged. Replaces the old flat "General Top-20" nav block."""
        nav = """<nav class="sidebar">
            <div class="nav-header">Investment Filter</div>
        """
        for label, tickers in band_tickers.items():
            if not tickers:
                continue
            title = escape(self._BAND_TITLES.get(label, label))
            nav += f"""
            <div class="nav-section">
                <h4>{title}</h4>
            """
            for ticker in tickers:
                nav += f'<a href="#{ticker}">{ticker}</a>\n'
            nav += """
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

/* Additive orientation + benchmark-verdict chips (no existing number/bar affected). */
.orient-chip {
    display: inline-block;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 0.72em;
    font-weight: normal;
    color: #555;
    background: #eef1f5;
    border: 1px solid #d6dbe2;
    border-radius: 10px;
    padding: 0 7px;
    margin-left: 6px;
    white-space: nowrap;
    vertical-align: middle;
    cursor: help;
}

.verdict-chip {
    display: inline-block;
    font-size: 0.8em;
    font-weight: bold;
    border-radius: 8px;
    padding: 0 5px;
    margin-left: 4px;
    vertical-align: middle;
    cursor: help;
}

.verdict-chip.pass {
    color: #155724;
    background: #d4edda;
}

.verdict-chip.warn {
    color: #721c24;
    background: #f8d7da;
}

.verdict-chip.neutral {
    color: #555;
    background: #e9e9e9;
}

/* 4-state verdict icon + confidence-tagged suspicion flag (metric-icons-spec) */
.verdict-icon {
    margin-left: 4px;
    font-size: 0.85em;
    vertical-align: middle;
    cursor: help;
}
.flag-icon {
    margin-left: 2px;
    font-size: 0.85em;
    vertical-align: middle;
    cursor: help;
    white-space: nowrap;
}
.flag-icon sub {
    font-size: 0.65em;
    font-weight: bold;
    vertical-align: sub;
}
.flag-icon.tier-H sub { color: #b30000; }   /* High   */
.flag-icon.tier-M sub { color: #c77800; }   /* Medium */
.flag-icon.tier-L sub { color: #7a7a00; }   /* Low    */

.icon-legend {
    background: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 0 0 14px 0;
    font-size: 0.85em;
    line-height: 1.7;
}
.icon-legend .leg { margin-right: 12px; white-space: nowrap; }
.icon-legend .leg-sep { margin-right: 12px; color: #999; }
.icon-legend .leg-note { margin-top: 4px; color: #666; font-size: 0.95em; }

.trends {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.trend-row {
    display: flex;
    align-items: center;
    gap: 8px;
}

.trend-label {
    min-width: 180px;
    flex: 0 0 auto;
}

.axis-start,
.axis-end,
.axis-span {
    font-size: 0.75em;
    color: #999;
    font-family: monospace;
    white-space: nowrap;
}

.axis-start {
    text-align: right;
    min-width: 58px;
}

.axis-end {
    min-width: 58px;
}

.axis-span {
    color: #0066cc;
    font-weight: bold;
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
    parser.add_argument('--run-date', type=str, default=None, metavar='YYYY-MM-DD',
                       help='Pin a specific run-date. Default: the latest postRank date on '
                            'disk. All artifacts (Boresults + CSVs) are loaded for this same '
                            'date; a missing same-date file is a hard error (no cross-run mix).')

    args = parser.parse_args()

    augment = (args.augment == 'on') and not args.no_augment
    if args.refresh_yahoo is None:
        refresh_yahoo = None
    elif args.refresh_yahoo == '__ALL__':
        refresh_yahoo = True
    else:
        refresh_yahoo = {t.strip() for t in args.refresh_yahoo.split(',') if t.strip()}

    try:
        # Load data (single date-consistent run; hard error on any same-date file missing)
        data = load_run_data(args.run_dir, VALUATION_REPO, run_date=args.run_date)
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
