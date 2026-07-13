"""
carveOut.py  --  Phase-1 cohort carve-out (partition, not re-score)

Partitions the full BoScore-ranked universe into a GENERAL pool plus three
DISJOINT side-cohorts (REITs, Mining, investment vehicles) at the postBo seam,
BEFORE the historical head(100) selection, and applies a gentle market-cap floor
uniformly.  The MAIN shortlist is then drawn only from the general pool; each
cohort is ranked with the SAME machinery (postBoScoreRanking) and presented as a
labeled side-list.  Nothing here re-scores a name -- names are ROUTED; the
ranking of each pool is the existing pipeline, unchanged.

Design source: valuation-specialist carve-out design + fmp-specialist detection
findings (2026-07-13).

Cohort rules (Phase-1, tonight's data; sector from sectorsdic_fmp.pickle):
  REIT     : sector == 'Real Estate'
  Mining   : sector == 'Basic Materials'   (whole-sector -- deliberately NOT
             name-matched: name-matching MISSES the large miners we most want
             removed, e.g. Newmont/Fresnillo/Agnico carry no keyword.  Whole-
             sector over-carves ~chem/steel into the mining side-list; accepted
             for Phase-1.  Phase-2 mining-only refinement needs the `industry`
             field, which is not in tonight's pickle yet.)
  Vehicle  : within sector == 'Financial Services', a pass-through vehicle
             (closed-end fund / investment trust / BDC).  Operating fee-earning
             asset managers (e.g. City of London, Polar Capital) are KEPT in the
             general pool.  Metric-fingerprint is the backbone; name corroborates.

Label assignment: each symbol maps to exactly ONE sector via _load_sector_map
(which uses setdefault -- the FIRST sector list containing the symbol wins, by the
pickle's dict iteration order).  classify() then maps that single sector to a
label, so the elif chain is NOT a priority mechanism (a symbol cannot be both
'Real Estate' and 'Basic Materials').  Any cross-list ambiguity is resolved
earlier, inside _load_sector_map's setdefault, by dict order -- not here.

The market-cap floor ($25M default) is applied to the general pool AND every
side-list uniformly.  Names with UNKNOWN market cap are KEPT (never dropped on a
data gap -- conservative; better to keep a good name than silently discard it).
"""

import os
import re
import numpy as np
import pandas as pd

# --- investment-vehicle name patterns (corroborate the metric fingerprint) ----
_NAME_VEHICLE = re.compile(
    r'investment trust|investment company|\btrust\b|\bfund\b|split[- ]?share|closed[- ]?end',
    re.IGNORECASE)
_NAME_VEHICLE_EXCL = re.compile(
    r'banc|bank|reinsur|assurance|life insur|mutual|savings',
    re.IGNORECASE)
_NAME_BDC = re.compile(r'\bBDC\b|business development', re.IGNORECASE)

# fingerprint of a pass-through vehicle (balance-sheet is nearly all equity;
# almost no operating revenue relative to assets)
_VEHICLE_EQ_ASSETS_MIN = 0.70
_VEHICLE_REV_ASSETS_MAX = 0.10

REIT_SECTOR = 'Real Estate'
MINING_SECTOR = 'Basic Materials'
FINANCIAL_SECTOR = 'Financial Services'


def _load_sector_map(sector_pickle='sectorsdic_fmp.pickle'):
    """symbol -> sector, from the local sector pickle (dict sector -> [symbols]).
    Reuses the forensicFlags convention.  Returns {} if the pickle is absent."""
    try:
        from forensicFlags import _load_sector_map as _ff_load
        return _ff_load(sector_pickle)
    except Exception:
        # Self-contained fallback (identical logic) so the carve-out never hard-
        # depends on forensicFlags importing cleanly.
        if not os.path.exists(sector_pickle):
            return {}
        sectordic = pd.read_pickle(sector_pickle)
        m = {}
        for sector, symbols in sectordic.items():
            for s in symbols:
                m.setdefault(s, sector)
        return m


def _latest_fundamentals(cdx_df):
    """Per-source latest-available marketCap, equity/assets, revenue/assets.

    cdx_df is stored oldest-first; we forward-fill within each source and take the
    last row so a NaN in the final quarter falls back to the most recent non-NaN.
    """
    cols = ['marketCap', 'totalStockholdersEquity', 'totalAssets', 'revenue']
    have = [c for c in cols if c in cdx_df.columns]
    df = cdx_df[['source', 'date'] + have].copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['source', 'date'])
    df[have] = df.groupby('source')[have].ffill()
    latest = df.groupby('source')[have].last()
    out = pd.DataFrame(index=latest.index)
    out['marketCap'] = latest.get('marketCap', np.nan)
    ta = latest.get('totalAssets', pd.Series(np.nan, index=latest.index))
    eq = latest.get('totalStockholdersEquity', pd.Series(np.nan, index=latest.index))
    rev = latest.get('revenue', pd.Series(np.nan, index=latest.index))
    with np.errstate(divide='ignore', invalid='ignore'):
        out['eq_assets'] = eq / ta.replace(0, np.nan)
        out['rev_assets'] = rev / ta.replace(0, np.nan)
    return out


def _is_investment_vehicle(name, eq_assets, rev_assets):
    """(is_vehicle, reason).  (a)-only definition: carve pass-through vehicles
    (CEF / investment trust / BDC), KEEP operating fee-earning asset managers."""
    n = name if isinstance(name, str) else ''
    # Rule (2): BDC name alone is sufficient.
    if _NAME_BDC.search(n):
        return True, 'BDC-name'
    # Rule (1): trust/fund name (not a bank/insurer) AND the pass-through fingerprint.
    if _NAME_VEHICLE.search(n) and not _NAME_VEHICLE_EXCL.search(n):
        if (pd.notna(eq_assets) and pd.notna(rev_assets)
                and eq_assets >= _VEHICLE_EQ_ASSETS_MIN
                and rev_assets <= _VEHICLE_REV_ASSETS_MAX):
            return True, 'trust/fund-name+fingerprint'
    return False, ''


def classify(symbols, sector_map, fund, names):
    """Return (label_series, vehicle_reason_series) over `symbols`.
    label in {'general','REIT','Mining','InvestmentVehicle'}."""
    labels, reasons = {}, {}
    for s in symbols:
        sec = sector_map.get(s)
        label, reason = 'general', ''
        if sec == REIT_SECTOR:
            label = 'REIT'
        elif sec == MINING_SECTOR:
            label = 'Mining'
        elif sec == FINANCIAL_SECTOR:
            eq = fund['eq_assets'].get(s, np.nan) if s in fund.index else np.nan
            rev = fund['rev_assets'].get(s, np.nan) if s in fund.index else np.nan
            is_veh, reason = _is_investment_vehicle(names.get(s, ''), eq, rev)
            if is_veh:
                label = 'InvestmentVehicle'
        labels[s] = label
        reasons[s] = reason
    return pd.Series(labels), pd.Series(reasons)


def partition_universe(BoScore_df, cdx_df, tickers_df,
                       sector_pickle='sectorsdic_fmp.pickle',
                       mcap_floor=25e6, cohort_head=25):
    """Partition the full BoScore-ranked universe.

    Returns dict:
      general      : BoScore_df rows for the general pool (size-floored, sorted)
      cohorts      : {'REIT','Mining','InvestmentVehicle'} -> BoScore_df rows
                     (each size-floored, sorted; ready to .head(cohort_head))
      labels       : Series symbol -> cohort label (pre-floor, full universe)
      diagnostics  : counts + the investment-vehicle caught set (with reasons)
    """
    bs = BoScore_df.copy()
    symbols = list(bs['source'])

    sector_map = _load_sector_map(sector_pickle)
    if not sector_map:
        print("carveOut WARNING: sector map empty/absent -- REIT & Mining cohorts "
              "will be EMPTY and everything stays in the general pool (size floor "
              "still applies).", flush=True)

    names = {}
    cols = getattr(tickers_df, 'columns', [])
    if tickers_df is not None and 'symbol' in cols and 'name' in cols:
        names = dict(zip(tickers_df['symbol'], tickers_df['name']))

    fund = _latest_fundamentals(cdx_df)

    labels, reasons = classify(symbols, sector_map, fund, names)
    bs['carve_label'] = bs['source'].map(labels).fillna('general')

    # --- market-cap floor (uniform); keep unknown-mcap names ------------------
    mcap = bs['source'].map(fund['marketCap'])
    below = mcap.notna() & (mcap < mcap_floor)
    n_below = int(below.sum())
    n_unknown_mcap = int(mcap.isna().sum())
    below_sources = set(bs.loc[below, 'source'])
    bs_floored = bs[~below].reset_index(drop=True)

    general = bs_floored[bs_floored['carve_label'] == 'general'].reset_index(drop=True)
    cohorts = {}
    for lab in ('REIT', 'Mining', 'InvestmentVehicle'):
        cohorts[lab] = bs_floored[bs_floored['carve_label'] == lab].reset_index(drop=True)

    # Investment-vehicle caught set: the FULL pre-floor detection, with a
    # `below_floor` flag so the diagnostic is reconcilable with the post-floor
    # cohort count (n_InvestmentVehicle = kept; n_InvestmentVehicle_prefloor = all
    # caught; the difference is the below-floor rows).
    veh = bs[bs['carve_label'] == 'InvestmentVehicle'][['source']].copy()
    veh['name'] = veh['source'].map(names)
    veh['reason'] = veh['source'].map(reasons)
    veh['below_floor'] = veh['source'].isin(below_sources)
    veh = veh.reset_index(drop=True)

    diagnostics = {
        'n_universe': len(bs),
        'n_general': len(general),
        'n_REIT': len(cohorts['REIT']),
        'n_Mining': len(cohorts['Mining']),
        'n_InvestmentVehicle': len(cohorts['InvestmentVehicle']),          # post-floor (kept)
        'n_InvestmentVehicle_prefloor': int(len(veh)),                     # all caught (pre-floor)
        'n_below_floor': n_below,
        'n_unknown_mcap': n_unknown_mcap,
        'mcap_floor': mcap_floor,
        'vehicle_caught': veh,   # pre-floor detection; `below_floor` column reconciles to post-floor
    }
    return {'general': general, 'cohorts': cohorts, 'labels': bs.set_index('source')['carve_label'],
            'diagnostics': diagnostics}
