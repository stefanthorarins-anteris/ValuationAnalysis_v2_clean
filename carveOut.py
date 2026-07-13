"""
carveOut.py  --  cohort carve-out (partition + per-cohort re-weight, not re-score)

Partitions the full BoScore-ranked universe into a GENERAL pool plus FIVE DISJOINT
side-cohorts at the postBo seam, BEFORE the historical head(100) selection, applies
a uniform market-cap floor, and ranks each pool with the SAME machinery
(postBoScoreRanking) under a per-cohort weight override.  The MAIN shortlist is drawn
only from the general pool; each cohort is presented as a labeled top-N side-list.
Nothing here alters a name's raw metrics -- names are ROUTED and the AggScore weights
are swapped per cohort; the scoring machinery itself is the existing pipeline.

Design source: valuation-specialist carve-out design + fmp-specialist detection
findings + CEO cohort taxonomy (2026-07-13).

The five side-cohorts (sector from sectorsdic_fmp.pickle; industry from
industrydic_fmp_*.pickle):
  REIT     : sector == 'Real Estate'.
  Mining   : sector == 'Basic Materials' (whole-sector -- name-matching MISSES the
             large miners we most want carved out, e.g. Newmont/Fresnillo/Agnico
             carry no keyword; whole-sector over-carves some chem/steel, accepted).
  FIN-1 InvestmentVehicle : within Financial Services, a pass-through vehicle
             (closed-end fund / investment trust / BDC).  Decided BEFORE industry,
             by fundamental fingerprint + name (incl. a name-free fingerprint branch
             for CEFs that dodge the name rule) -- because the FMP `industry` field
             cannot separate vehicles from managers (BDCs/CEFs also carry
             'Asset Management').
  FIN-2 FinManager : operating fee-earning asset managers / brokers / platforms
             (e.g. City of London, Polar Capital).  Per CEO, these belong with the
             investment group -- NOT the general pool.  Classified PRIMARILY by the
             FMP `industry` string (Asset Management* / Capital Markets / etc.), with
             FIN2_MANAGER_OVERRIDE as a curated safety net for names the auto-signal
             misses.
  FIN-3 BalanceSheetFin : the residual financials -- banks, lenders (incl. consumer
             lenders, per CEO), insurers.  Everything financial not caught above.

Routing order per symbol (classify()): 1. FIN-1 vehicle (fingerprint + name);
2. FIN-2 vs FIN-3 by industry (primary) with the manager override; the rest of the
universe falls to REIT / Mining by sector, else general.

The market-cap floor ($25M default) is applied to the general pool AND every
side-list uniformly.  Names with UNKNOWN market cap are KEPT (never dropped on a
data gap -- conservative; better to keep a good name than silently discard it).
"""

import os
import re
import numpy as np
import pandas as pd

# --- Financial-Services sub-classification patterns --------------------------
# FIN-1 Investment Vehicles (CEF / investment trust / BDC): widened CEF keyword set
# (adds investors / income fund / high income) so "...Investors"-style CEFs are caught.
_NAME_VEHICLE = re.compile(
    r'investment trust|investment company|\btrust\b|\bfund\b|split[- ]?share|'
    r'closed[- ]?end|\binvestors\b|income fund|high income',
    re.IGNORECASE)
_NAME_VEHICLE_EXCL = re.compile(
    r'banc|bank|reinsur|assurance|life insur|mutual|savings',
    re.IGNORECASE)
_NAME_BDC = re.compile(r'\bBDC\b|business development', re.IGNORECASE)
# "capital corp" is BDC-ish but ambiguous (some bank holdings) -> require the
# pass-through fingerprint below rather than trusting the name alone.
_NAME_CAPCORP = re.compile(r'capital corp', re.IGNORECASE)

# Pass-through vehicle fingerprint (balance-sheet nearly all equity; little
# operating revenue vs assets). Equity/assets floor LOWERED 0.70 -> 0.45 so
# LEVERAGED BDCs (equity/assets ~0.45) are captured; still well above banks
# (equity/assets < 0.25) so it does not pull in balance-sheet financials.
_VEHICLE_EQ_ASSETS_MIN = 0.45
_VEHICLE_REV_ASSETS_MAX = 0.10

# name-FREE pass-through fingerprint -- catches CEFs / investment trusts / holding
# vehicles whose NAMES dodge the keyword rule (e.g. "abrdn Asia Focus",
# "Invesco Bond Income Plus", "BioPharma Credit") and which industry cannot help
# with (a CEF and a manager both tag 'Asset Management'). Signature: nearly all
# equity, ~no operating revenue. Thresholds set from the data (see review notes):
# the 3 known-missed CEFs sit at revenue/assets 0.045-0.095 with equity/assets
# 0.90-0.99, while the nearest genuine fee-earning manager at equity/assets>0.85
# (City of London, CLIG) is at revenue/assets 0.231 -- a wide gap. 0.12 catches the
# CEF/trust cluster with margin and stays well below any real manager (and the
# equity/assets>0.85 gate excludes banks ~0.1 / insurers ~0.2-0.3 / lenders and
# the asset-light or bank-like managers like BlackRock 0.33 / Schwab 0.10).
_VEHICLE_EQ_ASSETS_HI = 0.85
_VEHICLE_REV_ASSETS_HI = 0.12

# FIN-2 Managers/Brokers/Platforms: name-keyword APPROXIMATION (the fundamental
# fingerprint CANNOT separate a broker from a bank -- flatexDEGIRO has a bank's
# balance sheet), with a bank exclusion. WEAK boundary vs FIN-3; leakage both ways
# is expected until the FMP `industry` field lands. Flagged, not precise.
_NAME_MANAGER = re.compile(
    r'asset management|investment management|\bsecurities\b|brokerage|\bwealth\b|'
    r'\bpartners\b|\bplatform\b',
    re.IGNORECASE)
# 'capital' is over-broad on its own (Capital One = a consumer lender; Arch Capital /
# Greenlight Capital Re = insurers -> all belong in FIN-3). Require it to CO-OCCUR
# with a manager qualifier; bare 'X Capital Group/Holdings/Financial/Re' no longer
# qualifies as a manager.
_NAME_CAPITAL_MGR = re.compile(
    r'\bcapital\b.*\b(management|partners|advisors|advisers|markets)\b|'
    r'\b(management|partners|advisors|advisers|markets)\b.*\bcapital\b',
    re.IGNORECASE)
_NAME_BANK_EXCL = re.compile(r'banc|bank|savings|mutual', re.IGNORECASE)
# FIN-3 sanity (residual = banks/lenders/insurers): a bank name + a bank balance
# sheet. Used for flagging/validation only; FIN-3 membership is the RESIDUAL.
_NAME_BANK = re.compile(r'bancshares|bancorp|\bbank\b', re.IGNORECASE)

REIT_SECTOR = 'Real Estate'
MINING_SECTOR = 'Basic Materials'
FINANCIAL_SECTOR = 'Financial Services'
# Financial-Services sub-cohort labels
FIN1_VEHICLE = 'InvestmentVehicle'
FIN2_MANAGER = 'FinManager'
FIN3_BALSHEET = 'BalanceSheetFin'

# --- FMP `industry` -> FIN-2 vs FIN-3 (PRIMARY financial classifier) ----------
# The industry field cleanly separates managers/brokers from banks/lenders/insurers
# (it does NOT separate FIN-1 vehicles from FIN-2 managers -- BDCs/CEFs also carry
# 'Asset Management' -- so FIN-1 is decided BEFORE this by fingerprint+name). Any
# 'Asset Management*' variant (incl. -Global/-Income/-Cryptocurrency) is FIN-2 via
# startswith. Exact FIN-2 industries below; EVERYTHING ELSE financial (Banks*,
# Insurance*, Credit Services, Mortgages, Conglomerates, Shell, mis-sectored
# non-financial industries) falls to FIN-3 residual -- which keeps the general pool
# byte-unchanged (we never move a Financial-Services name into general here).
# AMBIGUOUS CALL (reported): 'Investment - Banking & Investment Services' (investment
# banks/broker-dealers) -> FIN-2 as capital-markets, per coordinator lean; they carry
# bank-like balance sheets, so flipping them to FIN-3 is a one-line change.
_FIN2_INDUSTRIES = {
    'Financial - Capital Markets',
    'Financial - Data & Stock Exchanges',
    'Investment - Banking & Investment Services',
}


def _fin_industry_label(industry):
    """FIN-2 / FIN-3 from the FMP industry string; None if industry is missing/
    unknown (caller then falls back to the name-keyword rule)."""
    if not industry or industry in _UNKNOWN_SECTORS:
        return None
    if industry.startswith('Asset Management') or industry in _FIN2_INDUSTRIES:
        return FIN2_MANAGER
    return FIN3_BALSHEET

# --- FIN-2 manager override (curated safety net; EDITABLE) -------------------
# Large asset managers / brokers / platforms whose BRAND names carry no
# manager/broker keyword (e.g. "AJ Bell", "Plus500", "flatexDEGIRO") and so fall
# through the FIN-2 name rule into FIN-3. Force them into FIN-2, keyed by the DEDUP
# SURVIVOR ticker so the override hits the surviving line.
#
# ROLE (CEO, 2026-07-13): the FMP `industry` field (ingestion capture wired in
# findAllSectors.py; a future run maps Asset-Management/Capital-Markets->FIN-2,
# Banks/Credit->FIN-3, etc.) is the PRIMARY automatic classifier. This manual list
# is the COMPLEMENTARY post-classification SAFETY NET: the final say for big names
# the rule still gets wrong, and the fallback where `industry` is missing. It is
# NOT purely temporary -- it persists as the override layer once industry routing
# lands (and can then shrink to genuine exceptions). Curated 2026-07-13.
FIN2_MANAGER_OVERRIDE = {
    # CEO-named large managers/platforms (coordinator directive):
    'CLIG.L',  # City of London Investment Group
    'AJB.L',   # AJ Bell (platform)
    'FTK.DE',  # flatexDEGIRO (broker/platform)
    'PLUS.L',  # Plus500 (trading platform)
    'RAT.L',   # Rathbones Group (wealth)
    'QLT.L',   # Quilter (wealth platform)
    # Polar Capital: matched FIN-2 via the old bare-'capital' keyword; after
    # narrowing that keyword it no longer does, so pin it here explicitly.
    'POLR.L',  # Polar Capital Holdings (asset manager)
    # Other obvious brand-named managers/advisors/platforms/brokers found in the
    # FIN-3 fall-through scan (capital-light, fee-earning; NOT balance-sheet banks):
    'SDR.L',   # Schroders (asset manager)
    'EMG.L',   # Man Group (asset manager)
    'ASHM.L',  # Ashmore Group (asset manager)
    'ABDN.L',  # abrdn / Aberdeen Group (asset manager)
    'BEN',     # Franklin Resources (asset manager)
    'DWS.DE',  # DWS Group (asset manager)
    'STJ.L',   # St. James's Place (wealth platform)
    'BRK.L',   # Brooks Macdonald Group (wealth; NOT Berkshire, which is BRK-A/BRK-B)
    'TCAP.L',  # TP ICAP (interdealer broker)
    'IBKR',    # Interactive Brokers (broker/platform)
    'EVR',     # Evercore (advisory)
    'HLI',     # Houlihan Lokey (advisory)
    'BLK',     # BlackRock (asset manager)
    'CG',      # Carlyle Group (asset manager)
    'SCHW',    # Charles Schwab (broker/platform; NOTE bank-like balance sheet -- borderline)
    'RJF',     # Raymond James (brokerage w/ bank sub -- borderline)
}

# --- per-cohort weight vectors (CEO-approved, valuation-specialist proposal) ----
# Keys are the REAL metric keys from createDicts.getPostDict() (all 21). Threaded
# into postBoScoreRanking(weight_override=...) per cohort; the general/main pool
# uses NO override (default weights). weight 0 -> metric dropped from AggScore
# (constant/neutral in rankOfRanks); does not change cohort membership.
COHORT_WEIGHTS = {
    'Mining': {
        'earnYield': 0.5, 'RoA': 1.0, 'returnOnEquity': 0.5, 'returnOnCapitalEmployed': 0.5,
        'grahamNumberToPrice': 0.25, 'bVpRatio': 0.75, 'tbVpRatio': 1.0, 'freeCashFlowYield': 1.5,
        'freeCashFlowPerShareGrowth': 0.5, 'revenueGrowth': 0.5, 'incomeQuality': 1.25,
        'grossProfitMargin': 0.25, 'Altman-Z': 1.0, 'Piotroski': 0.75, 'currentRatio': 0.75,
        'DcfToPrice': 0.25, 'EPStoEPSmean': 1.0, 'CycleHeat': -1.5, 'priceGrowth': 0.25,
        'marketCapRevQuants': 0.25, 'BoScore': 0.1,
    },
    'REIT': {
        'earnYield': 0, 'RoA': 0.5, 'returnOnEquity': 0.5, 'returnOnCapitalEmployed': 0.25,
        'grahamNumberToPrice': 0, 'bVpRatio': 0.5, 'tbVpRatio': 0.5, 'freeCashFlowYield': 1.0,
        'freeCashFlowPerShareGrowth': 0.75, 'revenueGrowth': 1.0, 'incomeQuality': 1.25,
        'grossProfitMargin': 0, 'Altman-Z': 0, 'Piotroski': 0.25, 'currentRatio': 0,
        'DcfToPrice': 0.25, 'EPStoEPSmean': 0, 'CycleHeat': -0.25, 'priceGrowth': 0.5,
        'marketCapRevQuants': 0.25, 'BoScore': 0.1,
    },
    FIN1_VEHICLE: {   # FIN-1 Investment Vehicles
        'earnYield': 0, 'RoA': 0, 'returnOnEquity': 0.25, 'returnOnCapitalEmployed': 0,
        'grahamNumberToPrice': 0, 'bVpRatio': 2.0, 'tbVpRatio': 1.0, 'freeCashFlowYield': 0,
        'freeCashFlowPerShareGrowth': 0, 'revenueGrowth': 0, 'incomeQuality': 0,
        'grossProfitMargin': 0, 'Altman-Z': 0, 'Piotroski': 0, 'currentRatio': 0,
        'DcfToPrice': 0, 'EPStoEPSmean': 0, 'CycleHeat': 0, 'priceGrowth': 0.25,
        'marketCapRevQuants': 0, 'BoScore': 0.1,
    },
    FIN2_MANAGER: {   # FIN-2 Managers / Brokers / Platforms
        'earnYield': 1.5, 'RoA': 0.5, 'returnOnEquity': 2.0, 'returnOnCapitalEmployed': 1.0,
        'grahamNumberToPrice': 0.25, 'bVpRatio': 0.25, 'tbVpRatio': 0.25, 'freeCashFlowYield': 2.0,
        'freeCashFlowPerShareGrowth': 1.5, 'revenueGrowth': 1.5, 'incomeQuality': 1.0,
        'grossProfitMargin': 0, 'Altman-Z': 0.25, 'Piotroski': 0.5, 'currentRatio': 0.25,
        'DcfToPrice': 0.35, 'EPStoEPSmean': 0.5, 'CycleHeat': -0.5, 'priceGrowth': 0.5,
        'marketCapRevQuants': 0.25, 'BoScore': 0.1,
    },
    FIN3_BALSHEET: {  # FIN-3 Balance-Sheet Financials (banks / lenders / insurers)
        'earnYield': 1.0, 'RoA': 0.5, 'returnOnEquity': 2.0, 'returnOnCapitalEmployed': 0,
        'grahamNumberToPrice': 0.75, 'bVpRatio': 1.5, 'tbVpRatio': 1.0, 'freeCashFlowYield': 0,
        'freeCashFlowPerShareGrowth': 0, 'revenueGrowth': 0.75, 'incomeQuality': 0.25,
        'grossProfitMargin': 0, 'Altman-Z': 0, 'Piotroski': 0.25, 'currentRatio': 0,
        'DcfToPrice': 0, 'EPStoEPSmean': 1.0, 'CycleHeat': -1.0, 'priceGrowth': 0.5,
        'marketCapRevQuants': 0.25, 'BoScore': 0.1,
    },
}

# FMP no-sector sentinels -- treat as UNKNOWN, NOT a legitimate sector. 'Unspecified'
# is the single most common map value (~9,300 names). If it counted as "known" it
# would win the dedup sector-propagation vote (tie -> insertion order) and overwrite a
# real REIT/Mining/Financial tag on a sibling line, routing the whole issuer to the
# general pool -- breaking the "0 Basic-Materials in general by construction" guarantee.
_UNKNOWN_SECTORS = {'Unspecified', 'Unknown', ''}


def _is_known_sector(sec):
    """True only for a real sector label (not a no-sector sentinel / None / empty)."""
    return bool(sec) and sec not in _UNKNOWN_SECTORS


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


def _load_industry_map(industry_pickle=None):
    """symbol -> industry (FMP `industry` field), from the local industry pickle
    (already a flat dict symbol->industry). If no explicit path is given, use the
    NEWEST industrydic_fmp_*.pickle present. Returns {} if none found -> callers
    fall back to the name-keyword rule for FIN-2/FIN-3."""
    import glob
    path = industry_pickle
    if not path:
        cands = sorted(glob.glob('industrydic_fmp_*.pickle'))
        path = cands[-1] if cands else None
    if not path or not os.path.exists(path):
        return {}
    d = pd.read_pickle(path)
    return dict(d) if isinstance(d, dict) else {}


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
    """FIN-1: (is_vehicle, reason). Carve pass-through vehicles (CEF / investment
    trust / BDC); KEEP operating fee-earning asset managers (-> FIN-2)."""
    n = name if isinstance(name, str) else ''
    # BDC name alone is sufficient (unambiguous).
    if _NAME_BDC.search(n):
        return True, 'BDC-name'
    # CEF/trust/fund names, and the ambiguous "capital corp", need the pass-through
    # fingerprint (not a bank/insurer; nearly-all-equity, negligible revenue).
    if (_NAME_VEHICLE.search(n) or _NAME_CAPCORP.search(n)) and not _NAME_VEHICLE_EXCL.search(n):
        if (pd.notna(eq_assets) and pd.notna(rev_assets)
                and eq_assets >= _VEHICLE_EQ_ASSETS_MIN
                and rev_assets <= _VEHICLE_REV_ASSETS_MAX):
            return True, 'vehicle-name+fingerprint'
    # NAME-FREE pass-through fingerprint: nearly all equity + ~no operating revenue
    # => a CEF/trust/holding vehicle regardless of name or industry. Catches the
    # closed-end funds that dodge the name rule and would otherwise land in FIN-2.
    if (pd.notna(eq_assets) and pd.notna(rev_assets)
            and eq_assets > _VEHICLE_EQ_ASSETS_HI
            and rev_assets < _VEHICLE_REV_ASSETS_HI):
        return True, 'fingerprint-only(pass-through)'
    return False, ''


def _is_fin_manager(name):
    """FIN-2: name-keyword match for a manager/broker/platform, with bank exclusion.
    WEAK boundary (name-only approximation) -- see module patterns note. The curated
    FIN2_MANAGER_OVERRIDE handles brand-named managers this keyword rule cannot see."""
    n = name if isinstance(name, str) else ''
    if _NAME_BANK_EXCL.search(n):
        return False
    return bool(_NAME_MANAGER.search(n) or _NAME_CAPITAL_MGR.search(n))


def classify(symbols, sector_map, fund, names, industry_map=None):
    """Return (label_series, reason_series) over `symbols`. label in
    {'general','REIT','Mining','InvestmentVehicle','FinManager','BalanceSheetFin'}.

    REIT / Mining by sector. Within Financial Services, LAYERED:
      1. FIN-1 vehicle  (fingerprint + name -- BEFORE industry, which cannot tell
         a BDC/CEF from a manager);
      2. FIN-2 vs FIN-3 by INDUSTRY (the primary, clean signal);
      3. fallback to the name-keyword rule where industry is missing;
      4. FIN2_MANAGER_OVERRIDE applied LAST -- the final say / safety net."""
    industry_map = industry_map or {}
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
            nm = names.get(s, '')
            # 1. FIN-1 vehicle first (industry can't separate vehicles from managers)
            is_veh, reason = _is_investment_vehicle(nm, eq, rev)
            if is_veh:
                label = FIN1_VEHICLE
            else:
                # 2. FIN-2 vs FIN-3 by industry
                fin = _fin_industry_label(industry_map.get(s))
                if fin is not None:
                    label, reason = fin, 'industry:' + str(industry_map.get(s))
                # 3. fallback: name-keyword when industry is missing/unknown
                elif _is_fin_manager(nm):
                    label, reason = FIN2_MANAGER, 'manager-keyword(no-industry)'
                else:
                    label, reason = FIN3_BALSHEET, 'residual(no-industry)'
            # 4. manual override LAST -- wins over industry
            if s in FIN2_MANAGER_OVERRIDE:
                label, reason = FIN2_MANAGER, 'manager-override'
        labels[s] = label
        reasons[s] = reason
    return pd.Series(labels), pd.Series(reasons)


# --- issuer-level de-duplication --------------------------------------------
# Same-issuer lines (share-classes, preferreds, notes, cross-listings) occupy
# multiple slots for ONE economic bet and -- worse -- a cross-listing can leak
# past the carve-out when the sector map tags only the primary line. We collapse
# to one line per issuer BEFORE the carve-out partition, so secondary listings
# inherit the issuer's (propagated) sector.
#
# Reuses baseline_tools/universe_dedup's fundamental-fingerprint idea (exact-equal
# revenue/netIncome/totalAssets/shares == same issuer), but EXTENDS it: that
# signal alone MISSES currency-converted cross-listings (verified: Lundin Gold's
# London line 0R4M.L reports FX-shifted revenue/NI/TA vs LUG.TO/LUG.ST, so its
# fingerprint differs by ~1% and it stays a separate line -> the leak). We add a
# currency-INVARIANT edge -- normalized companyName + weightedAverageShsOut
# (share count is FX-independent) -- and union-find the two edge types.
_ISSUER_STRIP = re.compile(
    r'\b(inc|incorporated|corp|corporation|company|co|plc|ltd|limited|lp|llc|'
    r'sa|s\.a|ag|se|nv|asa|ab|oyj|spa|the|holdings?|group|ordinary|shares?|'
    r'class|senior|notes?|due|preferred|pref|units?|warrants?|adr|ads)\b', re.I)

def _norm_issuer_name(x):
    if not isinstance(x, str) or not x.strip():
        return ''
    s = x.lower()
    s = re.sub(r'\d+(\.\d+)?%.*$', ' ', s)      # drop "6.125% senior notes due 2026" tails
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    s = _ISSUER_STRIP.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def _latest_raw(cdx_df, cols):
    have = [c for c in cols if c in cdx_df.columns]
    df = cdx_df[['source', 'date'] + have].copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['source', 'date'])
    df[have] = df.groupby('source')[have].ffill()
    return df.groupby('source')[have].last()


def dedup_to_issuers(BoScore_df, cdx_df, sector_map, names):
    """Collapse same-issuer lines to ONE survivor each, with sector propagation.

    Returns dict: survivors(set), member_to_survivor(dict), sector_override(dict
    survivor->propagated sector), diagnostics(dict with report DataFrame + counts).

    SURVIVOR RULE (stated explicitly): within an issuer group, prefer
      (1) a line the sector map already tags (the recognised primary), then
      (2) largest latest market cap (most investable), then
      (3) a symbol NOT starting with a digit (deprioritise LSE IOB/grey-market
          depositary lines, e.g. 0R4M.L), then
      (4) fewest punctuation (bare ticker), shortest, alphabetical (deterministic).
    SECTOR PROPAGATION: the survivor inherits the group's known sector (majority of
    tagged members) -- this is what plugs the cross-listing leak regardless of which
    line survives. A conflict (two DIFFERENT known sectors in one group) is flagged.
    """
    syms = list(BoScore_df['source'])
    fp_cols = ['revenue', 'netIncome', 'totalAssets', 'weightedAverageShsOut']
    latest = _latest_raw(cdx_df, fp_cols + ['marketCap'])

    parent = {s: s for s in syms}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def _val(s, c):
        if s in latest.index and c in latest.columns:
            v = latest.at[s, c]
            if pd.notna(v) and np.isfinite(v):
                return round(float(v), 4)
        return None

    # edge A: identical fundamental fingerprint (same-currency lines)
    fpmap = {}
    for s in syms:
        vals = [_val(s, c) for c in fp_cols]
        if all(v is not None for v in vals):
            fpmap.setdefault(tuple(vals), []).append(s)
    # edge B: currency-invariant (normalized name + shares outstanding)
    nsmap = {}
    for s in syms:
        nm = _norm_issuer_name(names.get(s, '')) if names else ''
        sh = _val(s, 'weightedAverageShsOut')
        if nm and sh is not None:
            nsmap.setdefault((nm, sh), []).append(s)
    for grp in list(fpmap.values()) + list(nsmap.values()):
        for s in grp[1:]:
            union(grp[0], s)

    comps = {}
    for s in syms:
        comps.setdefault(find(s), []).append(s)

    from collections import Counter
    def _key(s):
        known = 0 if _is_known_sector(sector_map.get(s)) else 1
        mc = _val(s, 'marketCap')
        mc = mc if mc is not None else -1.0
        digitpfx = 1 if s[:1].isdigit() else 0
        punct = sum(ch in '-.' for ch in s)
        return (known, -mc, digitpfx, punct, len(s), s)

    survivors, member_to_survivor, sector_override = set(), {}, {}
    rows, conflicts = [], []
    for members in comps.values():
        surv = sorted(members, key=_key)[0]
        survivors.add(surv)
        secs = [x for x in (sector_map.get(m) for m in members) if _is_known_sector(x)]
        prop = None
        if secs:
            # On a known-vs-known conflict, PREFER a cohort-relevant sector
            # (REIT / Mining / Financial) so a conflicting non-cohort sibling tag
            # (e.g. a baby-bond line mistagged 'Industrials') cannot demote a REIT/
            # miner/BDC issuer out of its cohort. The three cohort sectors are
            # mutually disjoint, so at most one is present; ties within the chosen
            # pool break by majority then insertion order.
            cohort_secs = [x for x in secs
                           if x in (REIT_SECTOR, MINING_SECTOR, FINANCIAL_SECTOR)]
            pool = cohort_secs if cohort_secs else secs
            prop = Counter(pool).most_common(1)[0][0]
            if len(set(secs)) > 1:
                conflicts.append((surv, dict(Counter(secs)), prop))
            sector_override[surv] = prop
        for m in members:
            member_to_survivor[m] = surv
            if m != surv:
                rows.append((m, surv, names.get(m, ''), sector_map.get(m, ''), prop,
                             '|'.join(sorted(members))))
    report = pd.DataFrame(rows, columns=['dropped', 'survivor', 'name',
                                         'orig_sector', 'propagated_sector', 'issuer_group'])
    diagnostics = {'n_lines_in': len(syms), 'n_issuers_out': len(comps),
                   'n_collapsed': len(syms) - len(comps),
                   'sector_conflicts': conflicts, 'report': report}
    return {'survivors': survivors, 'member_to_survivor': member_to_survivor,
            'sector_override': sector_override, 'diagnostics': diagnostics}


def partition_universe(BoScore_df, cdx_df, tickers_df,
                       sector_pickle='sectorsdic_fmp.pickle', industry_pickle=None,
                       mcap_floor=25e6, cohort_head=25, dedup=True):
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

    industry_map = _load_industry_map(industry_pickle)
    if not industry_map:
        print("carveOut WARNING: industry map absent -- FIN-2/FIN-3 split falls back "
              "to the name-keyword rule (weak). Provide industrydic_fmp_*.pickle.",
              flush=True)

    names = {}
    cols = getattr(tickers_df, 'columns', [])
    if tickers_df is not None and 'symbol' in cols and 'name' in cols:
        names = dict(zip(tickers_df['symbol'], tickers_df['name']))

    # --- issuer-level de-dup FIRST (before carve-out) so secondary listings
    # collapse to one survivor and inherit the issuer's propagated sector, plugging
    # the cross-listing leak. A no-op only if every issuer already has one line.
    dedup_diag = None
    if dedup:
        ded = dedup_to_issuers(bs, cdx_df, sector_map, names)
        bs = bs[bs['source'].isin(ded['survivors'])].reset_index(drop=True)
        symbols = list(bs['source'])
        sector_map = {**sector_map, **ded['sector_override']}   # propagated sectors win
        dedup_diag = ded['diagnostics']
        print("carveOut dedup: %d lines -> %d issuers (collapsed %d)"
              % (dedup_diag['n_lines_in'], dedup_diag['n_issuers_out'],
                 dedup_diag['n_collapsed']), flush=True)

    fund = _latest_fundamentals(cdx_df)

    labels, reasons = classify(symbols, sector_map, fund, names, industry_map)
    bs['carve_label'] = bs['source'].map(labels).fillna('general')
    # industry coverage over the (deduped) working universe
    _ind_cov = sum(1 for s in symbols
                   if industry_map.get(s) and industry_map.get(s) not in _UNKNOWN_SECTORS)

    # --- market-cap floor (uniform); keep unknown-mcap names ------------------
    mcap = bs['source'].map(fund['marketCap'])
    below = mcap.notna() & (mcap < mcap_floor)
    n_below = int(below.sum())
    n_unknown_mcap = int(mcap.isna().sum())
    below_sources = set(bs.loc[below, 'source'])
    bs_floored = bs[~below].reset_index(drop=True)

    general = bs_floored[bs_floored['carve_label'] == 'general'].reset_index(drop=True)
    cohorts = {}
    for lab in ('REIT', 'Mining', FIN1_VEHICLE, FIN2_MANAGER, FIN3_BALSHEET):
        cohorts[lab] = bs_floored[bs_floored['carve_label'] == lab].reset_index(drop=True)

    # Financial-Services sub-cohort caught sets: FULL pre-floor detection with a
    # `below_floor` flag so each diagnostic reconciles to its post-floor cohort count.
    def _caught(lab):
        c = bs[bs['carve_label'] == lab][['source']].copy()
        c['name'] = c['source'].map(names)
        c['reason'] = c['source'].map(reasons)
        c['below_floor'] = c['source'].isin(below_sources)
        return c.reset_index(drop=True)
    veh = _caught(FIN1_VEHICLE)

    diagnostics = {
        'n_universe': len(bs),
        'n_general': len(general),
        'n_REIT': len(cohorts['REIT']),
        'n_Mining': len(cohorts['Mining']),
        'n_InvestmentVehicle': len(cohorts[FIN1_VEHICLE]),                 # FIN-1 post-floor (kept)
        'n_InvestmentVehicle_prefloor': int(len(veh)),                     # FIN-1 all caught (pre-floor)
        'n_FinManager': len(cohorts[FIN2_MANAGER]),                        # FIN-2 post-floor
        'n_BalanceSheetFin': len(cohorts[FIN3_BALSHEET]),                  # FIN-3 post-floor
        'n_below_floor': n_below,
        'n_unknown_mcap': n_unknown_mcap,
        'mcap_floor': mcap_floor,
        'industry_coverage': (_ind_cov, len(symbols)),   # (names with a real industry, universe)
        'vehicle_caught': veh,             # FIN-1 pre-floor; `below_floor` reconciles to post-floor
        'finmanager_caught': _caught(FIN2_MANAGER),
        'balancesheet_caught': _caught(FIN3_BALSHEET),
        'dedup': dedup_diag,     # None if dedup disabled; else {n_lines_in,n_issuers_out,n_collapsed,report,...}
    }
    return {'general': general, 'cohorts': cohorts, 'labels': bs.set_index('source')['carve_label'],
            'diagnostics': diagnostics}
