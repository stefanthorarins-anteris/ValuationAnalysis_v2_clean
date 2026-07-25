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
import sys
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

# --- market-cap band segmentation (ADDITIVE size axis over the GENERAL pool) ---
# SINGLE SOURCE OF TRUTH for BOTH band SELECTION (partition_by_marketcap) and per-
# band GRADING (baseline_tools/pipeline_analysis.beat_rate_vs_urth). Each tuple is
# (label, lo_usd, hi_usd, head_N): a company sits in exactly ONE band by its USD
# market cap in the half-open interval [lo, hi). The bands are ORTHOGONAL to the 5
# sector carve-cohorts and apply ONLY to the general pool. The existing universe-
# wide ranking is GROUPED, never re-scored or re-ranked -- each band takes its top-N
# in the existing order (CEO 2026-07-17: group, do NOT re-rank within band).
MCAP_BANDS = [
    ("General",       300e6, float('inf'), 20),
    ("Mid_150_300M",  150e6, 300e6,         5),
    ("Small_50_150M",  50e6, 150e6,         5),
    ("Micro_lt_50M",    0.0,  50e6,         5),
]

# --- approximate FX -> USD (coarse buckets; banding only needs the RIGHT band) ---
# cdx_df['marketCap'] is stored in each company's REPORTING currency, MIXED across the
# universe (verified: DORO.ST in SEK ~962M ~= $92M USD; FRES.L reports USD), so banding
# on the raw field would misband every non-USD name. We convert to USD via the captured
# reportedCurrency + this table. Rates are approximate mid-2020s spot; the cutoffs
# (50/150/300M) are coarse so exact FX is unnecessary. Unknown currency -> None -> the
# name's USD market cap is unknown -> routed to General, NEVER misbanded, NEVER dropped.
# TODO: wire a live/dated FX source (an FMP forex endpoint or a stored dated rate file)
# to replace this hardcoded snapshot. Do NOT build a bespoke FX-fetch subsystem for it.
FX_TO_USD = {
    'USD': 1.0, 'EUR': 1.08, 'GBP': 1.27, 'GBp': 0.0127, 'GBX': 0.0127,
    'CHF': 1.12, 'JPY': 0.0067, 'SEK': 0.095, 'NOK': 0.093, 'DKK': 0.145,
    'CAD': 0.73, 'AUD': 0.66, 'NZD': 0.61, 'HKD': 0.128, 'SGD': 0.74,
    'CNY': 0.138, 'CNH': 0.138, 'INR': 0.012, 'KRW': 0.00073, 'TWD': 0.031,
    'ZAR': 0.054, 'BRL': 0.185, 'MXN': 0.055, 'PLN': 0.25, 'ILS': 0.27,
    'AED': 0.272, 'SAR': 0.267, 'THB': 0.028, 'IDR': 0.000063, 'TRY': 0.030,
    'RUB': 0.011, 'CZK': 0.043, 'HUF': 0.0028, 'PHP': 0.017, 'MYR': 0.22,
    'ISK': 0.0072,
}


# --- COARSE exchange-suffix -> reporting-currency fallback --------------------------
# ONLY for use where the alternative is treating a mixed-currency marketCap AS IF it
# were USD -- i.e. the $25M universe FLOOR and the mcapQuants size tilt, both of which
# run unconditionally today and therefore mis-select rather than degrade (audit H-3/H8).
#
# It is deliberately NOT used by the market-cap BAND emission: the exchange suffix does
# NOT determine reporting currency (fmp-specialist, 2026-07-18: FRES.L reports USD;
# many .L lines are IOB depositary lines of foreign issuers), and the CEO's 2026-07-18
# decision is that the bands SKIP rather than ship a guess. Bands stay gated on the real
# reportedCurrency (currency_data_present) and become correct from the next full fetch.
#
# THE REAL CONTRACT (corrected, review M1 2026-07-25 -- the previous wording claimed this
# "can never drop a name it would have kept", which is FALSE and needs stating plainly):
#
#   The suffix is a PRIOR on reporting currency, not a fact. It is RIGHT for the typical
#   domestic issuer on a local exchange and WRONG for a USD- or EUR-reporting issuer
#   listed there -- and for a wrong-currency name the conversion moves it AWAY from its
#   true USD size, which CAN push it below the $25M floor and out of the universe.
#
# Measured on the 2026-07-17 universe (7,752 names with a market cap):
#   * 182 names are NEWLY EXCLUDED by the USD floor (168 .ST, 11 .TO, 3 .IC). Correct for
#     the SEK/CAD/ISK reporters among them; WRONG for any USD reporter on those exchanges,
#     which is divided by ~10 and drops out.
#   * 25 names are newly KEPT (23 .L, 2 .DE) on rate>1 inflation.
#   * 21,800 panel rows change size quartile.
# RESIDUAL ERROR, accepted: the per-name direction cannot be known without the real
# reportedCurrency. This is a net improvement over treating a mixed-currency field as USD
# (a 10-15x error on every non-USD reporter), not a correct conversion, and it goes dormant
# from the next full fetch as reportedCurrency populates.
#
# An UNKNOWN or absent suffix maps to 1.0 = exactly today's raw behaviour; only a KNOWN
# suffix can move a name, in either direction.
SUFFIX_TO_CURRENCY = {
    'L': 'GBP', 'IL': 'USD',
    'ST': 'SEK', 'OL': 'NOK', 'CO': 'DKK', 'HE': 'EUR', 'IC': 'ISK',
    'DE': 'EUR', 'F': 'EUR', 'MU': 'EUR', 'SG': 'EUR', 'BE': 'EUR', 'DU': 'EUR',
    'HM': 'EUR', 'PA': 'EUR', 'AS': 'EUR', 'BR': 'EUR', 'LS': 'EUR', 'MC': 'EUR',
    'MI': 'EUR', 'VI': 'EUR', 'IR': 'EUR', 'AT': 'EUR', 'SW': 'CHF',
    'TO': 'CAD', 'V': 'CAD', 'CN': 'CAD', 'NE': 'CAD',
    'AX': 'AUD', 'NZ': 'NZD', 'HK': 'HKD', 'SI': 'SGD', 'T': 'JPY',
    'KS': 'KRW', 'KQ': 'KRW', 'TW': 'TWD', 'TWO': 'TWD',
    'SS': 'CNY', 'SZ': 'CNY', 'NS': 'INR', 'BO': 'INR',
    'JO': 'ZAR', 'SA': 'BRL', 'MX': 'MXN', 'WA': 'PLN', 'TA': 'ILS',
    'PR': 'CZK', 'BD': 'HUF', 'IS': 'TRY', 'SR': 'SAR', 'AE': 'AED',
    'BK': 'THB', 'JK': 'IDR', 'KL': 'MYR', 'PS': 'PHP',
}


def _fx_to_usd(currency):
    """USD-per-unit for a reportedCurrency code, or None if missing / unknown code."""
    if not isinstance(currency, str):
        return None
    return FX_TO_USD.get(currency.strip())


# LSE International Order Book / grey-market DEPOSITARY lines: zero-prefixed .L tickers
# (0R4M.L, 0HQ7.L, 0A28.L...). These are SECONDARY listings of FOREIGN issuers -- mostly
# US/EU companies reporting USD or EUR -- so the ".L -> GBP" prior is wrong for the whole
# family, not occasionally wrong. 886 of the 1,640 .L sources on the 2026-07-17 panel are
# of this shape (20,666 rows), and 2,742 of them changed size quartile on a 1.27x GBP
# inflation that does not apply to them. They are therefore EXCLUDED from the suffix
# fallback and keep their raw market cap (rate 1.0) -- which for a USD reporter is the
# correct number anyway. This is the same population the audit identified as the
# "byte-identical clone" cross-listings (M-5).
_IOB_LSE_SYMBOL_RE = re.compile(r'^0[A-Z0-9]{2,4}\.L$', re.I)


def _suffix_fx_to_usd(symbol):
    """COARSE USD-per-unit rate guessed from a ticker's exchange suffix; 1.0 when the
    symbol has no suffix, an unrecognised one, or is an LSE IOB depositary line
    (= today's raw behaviour). See SUFFIX_TO_CURRENCY for the contract and its limits."""
    if not isinstance(symbol, str) or '.' not in symbol:
        return 1.0
    if _IOB_LSE_SYMBOL_RE.match(symbol.strip()):
        return 1.0
    cur = SUFFIX_TO_CURRENCY.get(symbol.rsplit('.', 1)[1].strip())
    if cur is None:
        return 1.0
    return FX_TO_USD.get(cur, 1.0)


def marketcap_usd_series(cdx_df, allow_suffix_fallback=False):
    """Row-aligned USD market cap for cdx_df: marketCap * FX(reportedCurrency).

    THE single currency-conversion path -- shared by partition_by_marketcap, the
    presentation, and the PIT beat-rate grading, so all three key off the SAME field.
    DEGRADES GRACEFULLY when reportedCurrency has not yet flowed (the pre-fetch saved
    pickles): returns all-NaN so every name reads as unknown-mcap (-> General), i.e.
    NOTHING is misbanded. NaN wherever marketCap is missing or the currency is unknown.
    Prefers a live reportedCurrency recompute over any materialized marketCap_usd column
    (so a stale FX snapshot on disk can never override the current table).

    allow_suffix_fallback (OPT-IN, default OFF): fill rows the real reportedCurrency
    cannot resolve using the coarse exchange-suffix guess (SUFFIX_TO_CURRENCY), with an
    unknown/absent suffix meaning rate 1.0 = raw marketCap. Callers that MUST produce a
    number for every name -- the universe floor and the mcapQuants size tilt -- pass
    True; the band emission does NOT (see SUFFIX_TO_CURRENCY note)."""
    cols = getattr(cdx_df, 'columns', [])
    if cdx_df is None or 'marketCap' not in cols:
        return pd.Series(np.nan, index=getattr(cdx_df, 'index', None))
    mc = pd.to_numeric(cdx_df['marketCap'], errors='coerce')
    out = None
    if 'reportedCurrency' in cols:
        rate = cdx_df['reportedCurrency'].map(_fx_to_usd).astype('float64')
        out = mc * rate
    elif 'marketCap_usd' in cols:        # materialized at ingest (belt-and-suspenders)
        out = pd.to_numeric(cdx_df['marketCap_usd'], errors='coerce')
    else:
        out = pd.Series(np.nan, index=cdx_df.index)
    if allow_suffix_fallback and 'source' in cols:
        srate = cdx_df['source'].map(_suffix_fx_to_usd).astype('float64')
        out = out.where(out.notna(), mc * srate)
    return out


def currency_data_present(cdx_df):
    """True only when currency data is actually USABLE -- i.e. reportedCurrency resolves
    to a known FX rate for at least one row, or a materialized marketCap_usd carries at
    least one finite value. Column PRESENCE alone is NOT enough: an all-NaN column (e.g.
    reportedCurrency coerced to NaN by a numeric-cast, or an empty materialization) would
    otherwise masquerade as 'present' and suppress the pending banners while every name
    silently routes to General. This is the backstop that keeps 'nothing wrong ships'
    true even if the ingest string-preservation regresses (CEO 2026-07-17)."""
    cols = getattr(cdx_df, 'columns', [])
    if cdx_df is None:
        return False
    if 'reportedCurrency' in cols:
        try:
            if cdx_df['reportedCurrency'].map(_fx_to_usd).notna().any():
                return True
        except Exception:
            pass
    if 'marketCap_usd' in cols:
        try:
            if pd.to_numeric(cdx_df['marketCap_usd'], errors='coerce').notna().any():
                return True
        except Exception:
            pass
    return False


def marketcap_usd_by_source(cdx_df, as_of=None, allow_suffix_fallback=False):
    """source -> latest USD market cap (latest non-NaN row). If `as_of` is given,
    restrict to date <= as_of, i.e. the POINT-IN-TIME market cap as-of that date.
    Returns {} when the frame is unusable. Used by partition_by_marketcap (latest,
    fallback OFF) and by the PIT beat-rate grading (as_of=buy).

    allow_suffix_fallback is forwarded to marketcap_usd_series -- pass True only from
    the universe floor / size-tilt callers that must produce a number for every name."""
    cols = getattr(cdx_df, 'columns', [])
    if cdx_df is None or 'source' not in cols or 'marketCap' not in cols:
        return {}
    keep = ['source', 'date', 'marketCap']
    for extra in ('reportedCurrency', 'marketCap_usd'):
        if extra in cols:
            keep.append(extra)
    df = cdx_df[keep].copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if as_of is not None:
        df = df[df['date'] <= pd.Timestamp(as_of)]
    df['_mcap_usd'] = marketcap_usd_series(
        df, allow_suffix_fallback=allow_suffix_fallback).values
    df = df.dropna(subset=['_mcap_usd']).sort_values(['source', 'date'])
    if df.empty:
        return {}
    return df.groupby('source')['_mcap_usd'].last().to_dict()


def band_for_marketcap_usd(v):
    """Band label for a USD market cap; None if unknown (caller routes to General).
    Bands are contiguous half-open [lo, hi), so every finite non-negative value maps to
    EXACTLY one band."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        return None
    for label, lo, hi, _N in MCAP_BANDS:
        if lo <= v < hi:
            return label
    return None


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


# Cross-listing near-equal tolerance (edge C). FMP reports an issuer's two listings
# in a common reporting currency, so their revenue/netIncome/totalAssets agree to
# within ~0.3-1% (verified: Barrick B vs ABX.TO 0.26%; Lundin Gold ~1%). 5% gives
# headroom above that while staying far below the gap between genuinely distinct
# firms -- and it is gated by an EXACT share-count match, which does the real work.
_XLIST_FUND_TOL = 0.05


def _fund_near_equal(a, b, latest, cols, tol=_XLIST_FUND_TOL):
    """True iff EVERY fundamental in `cols` is present for both a and b and agrees
    within relative tolerance `tol`. Backs the FX-/rename-invariant cross-listing
    edge: a missing value on either side is a NON-match (never merge on a data gap)."""
    for c in cols:
        va = latest.at[a, c] if (a in latest.index and c in latest.columns) else None
        vb = latest.at[b, c] if (b in latest.index and c in latest.columns) else None
        if va is None or vb is None or pd.isna(va) or pd.isna(vb):
            return False
        va, vb = float(va), float(vb)
        denom = max(abs(va), abs(vb))
        if denom == 0.0:
            if va != vb:
                return False
            continue
        if abs(va - vb) / denom > tol:
            return False
    return True


def _latest_raw(cdx_df, cols):
    have = [c for c in cols if c in cdx_df.columns]
    df = cdx_df[['source', 'date'] + have].copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['source', 'date'])
    df[have] = df.groupby('source')[have].ffill()
    return df.groupby('source')[have].last()


def _issuer_components(syms, cdx_df, names):
    """Union-find grouping of same-issuer lines by fundamental fingerprint.

    Groups share-classes / preferreds / notes / cross-listings of ONE economic issuer
    via three edges (see dedup_to_issuers for the full rationale):
      A  identical fundamental fingerprint (same-currency lines);
      B  currency-invariant (normalized companyName + weightedAverageShsOut);
      C  FX-/rename-invariant (EXACT shares + near-equal revenue/netIncome/totalAssets).

    Returns (comps, latest, _val):
      comps  : dict root_symbol -> [member symbols]  (insertion order = order in syms)
      latest : per-source latest raw fundamentals (from _latest_raw)
      _val   : (symbol, col) -> rounded finite float or None
    Shared by dedup_to_issuers (sector-survivor rule) and dedup_ranked (rank-survivor
    rule) so both resolve issuer identity IDENTICALLY."""
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
    # edge C: FX-/rename-invariant fingerprint -- EXACT weightedAverageShsOut +
    # NEAR-equal revenue/netIncome/totalAssets. Catches cross-listings both other edges
    # miss: edge A (exact fingerprint) fails because FMP reports the two lines with
    # tiny (~0.3-1%) reporting differences, not byte-equal; edge B (name+shares) fails
    # when the listings carry DIFFERENT names -- e.g. Barrick's NYSE line "Barrick
    # Mining Corp" vs its TSX line still "Barrick Gold Corp" after the 2025 rename, so
    # the normalized names diverge ("barrick mining" != "barrick gold") and B (which FMP
    # mis-sectors as Industrials) escaped the Mining carve into the general pool. Share
    # count is currency- and name-invariant; requiring an exact share match AND three
    # near-equal fundamentals makes a false merge of two distinct issuers effectively
    # impossible. Grouped by exact shares first so the pairwise check stays O(k^2) over
    # tiny (usually 1-3 name) share-collision groups.
    shmap = {}
    for s in syms:
        sh = _val(s, 'weightedAverageShsOut')
        if sh is not None and sh > 0:
            shmap.setdefault(sh, []).append(s)
    _xlist_cols = ['revenue', 'netIncome', 'totalAssets']
    for grp in shmap.values():
        if len(grp) < 2:
            continue
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                if _fund_near_equal(grp[i], grp[j], latest, _xlist_cols):
                    union(grp[i], grp[j])

    for grp in list(fpmap.values()) + list(nsmap.values()):
        for s in grp[1:]:
            union(grp[0], s)

    comps = {}
    for s in syms:
        comps.setdefault(find(s), []).append(s)
    return comps, latest, _val


# Decimals at which two AggScores count as TIED for the dedup survivor tie-break.
# Cross-listed clone lines carry byte-identical fundamentals and therefore produce
# EXACTLY equal scores, so this is effectively exact equality with last-bit slack.
_TIE_DECIMALS = 12


def _investability_key(sym, val_fn, sector_map=None):
    """Deterministic "most investable line" ordering key for an issuer's listings.

    Prefer (1) a line the sector map already tags (the recognised primary -- only when
    a sector_map is supplied), then (2) largest latest market cap (most investable),
    then (3) a symbol NOT starting with a digit (deprioritise LSE IOB/grey-market
    depositary lines, e.g. 0R4M.L / 0HQ7.L), then (4) fewest punctuation (bare ticker),
    shortest, alphabetical (fully deterministic).

    Shared by dedup_to_issuers (which passes sector_map, so criterion 1 is live) and by
    dedup_ranked's tie-break (no sector_map -> criterion 1 is constant and inert), so
    "which line of an issuer do we prefer" means ONE thing across the pipeline.
    """
    known = 0 if (sector_map is not None
                  and _is_known_sector(sector_map.get(sym))) else 1
    mc = val_fn(sym, 'marketCap')
    mc = mc if mc is not None else -1.0
    digitpfx = 1 if sym[:1].isdigit() else 0
    punct = sum(ch in '-.' for ch in sym)
    return (known, -mc, digitpfx, punct, len(sym), sym)


def dedup_ranked(ranked_sources, cdx_df, names, scores=None, sector_map=None):
    """Collapse same-issuer lines in a RANK-ORDERED source list, keeping the
    HIGHEST-RANKED (earliest-appearing) line per issuer and dropping every later
    same-issuer line. Order-preserving.

    TIE-BREAK (audit M3 fix, 2026-07-19).  When `scores` is supplied and an issuer's
    best-ranked lines are TIED on score, the survivor is picked by
    _investability_key instead of by SORT STABILITY.  This matters because a
    cross-listed clone line carries byte-identical fundamentals and therefore an
    EXACTLY equal AggScore, so which of the two the ranking happened to emit first was
    arbitrary -- and it decided which ticker the CEO sees.  On the 2026-01-09 pool 18
    of 87 ranked rows sit in exact-score pairs, and the arbitrary winner was the LSE
    grey-market depositary line in both cited cases: 0HQ7.L over BKE and 0IJO.L over
    EXEL, both at identical AggScore.  With the tie-break the bare US ticker wins.

    NOT changed (deliberately): when the scores genuinely DIFFER the highest-RANKED
    line still survives -- that is the documented survivor rule (see below) and the
    audit produced no evidence against it.  Concretely, HNNAZ (Hennessy Advisors
    NOTES) survived over HNNA on the Jan pool at AggScore 0.1159 vs 0.1026, i.e. NOT a
    tie: the score really did prefer the notes line.  Overriding a real score
    difference is a survivor-RULE change, not a tie-break, and needs a design decision
    (the underlying problem there is that a notes line entered the universe as
    type=='stock' at all -- audit M-5).

    CEO standing principle: NO duplicate issuers in the emitted top-N -- a dual-listing
    or share-class must not occupy two slots (the TFPM / TFPM.TO case). This is the
    SELECTION-TIME dedup: apply it to the full ranked list BEFORE taking head(N), so the
    emitted top-N contains N DISTINCT issuers. It reuses the EXACT issuer-fingerprint
    grouping (_issuer_components, edges A/B/C) the carve-out uses, so "same issuer"
    means the same thing across the pipeline.

    Note the survivor rule differs from dedup_to_issuers by design: here we keep the
    highest-RANKED line (the pick the score actually surfaced); the carve-out keeps the
    most-investable/recognised line for sector propagation. Both are correct for their
    purpose.

    Returns (kept, dropped):
      kept    : deduped rank-ordered source list (>= 1 line per distinct issuer)
      dropped : list of (dropped_symbol, kept_survivor) in rank order (audit trail)
    """
    ranked = list(ranked_sources)
    comps, _latest, _val = _issuer_components(ranked, cdx_df, names)
    root_of = {s: r for r, members in comps.items() for s in members}

    # --- resolve exact-score ties inside each issuer group -----------------------
    chosen = {}
    if scores is not None:
        sv = {}
        for s, x in zip(ranked, list(scores)):
            try:
                fx = float(x)
            except (TypeError, ValueError):
                fx = np.nan
            sv[s] = round(fx, _TIE_DECIMALS) if np.isfinite(fx) else np.nan
        by_root = {}
        for s in ranked:                      # ranked order -> members[0] is best-ranked
            by_root.setdefault(root_of.get(s, s), []).append(s)
        for r, members in by_root.items():
            if len(members) < 2:
                continue
            top_score = sv.get(members[0], np.nan)
            if not np.isfinite(top_score):
                continue
            tied = [m for m in members
                    if np.isfinite(sv.get(m, np.nan)) and sv[m] == top_score]
            if len(tied) > 1:
                chosen[r] = sorted(
                    tied, key=lambda m: _investability_key(m, _val, sector_map))[0]

    kept, dropped, done = [], [], {}
    for s in ranked:
        r = root_of.get(s, s)
        if r in done:
            if s != done[r]:        # never record the survivor as dropped-onto-itself
                dropped.append((s, done[r]))
        else:
            # The issuer is represented at its BEST rank position; the surviving TICKER
            # is the tie-break winner when the best-ranked lines were tied, else the
            # best-ranked line itself.
            surv = chosen.get(r, s)
            done[r] = surv
            kept.append(surv)
            if surv != s:
                dropped.append((s, surv))
    return kept, dropped


def partition_by_marketcap(ranked_df, cdx_df, names=None):
    """GROUP the existing general-pool ranking into market-cap bands (USD). ADDITIVE
    size axis: NO re-score, NO re-rank -- it only PARTITIONS the given ordering.

    Steps (CEO 2026-07-17 + valuation-specialist build spec):
      1. FIRST apply dedup_ranked to the rank-ordered sources (collapse same-issuer
         lines, keep the highest-ranked, order-preserving) so each band has DISTINCT
         issuers;
      2. take each survivor's LATEST USD market cap (marketcap_usd_series);
      3. assign it to exactly ONE band by the MCAP_BANDS cutoff -- unknown-mcap -> General,
         counted separately, NEVER dropped (mirrors the carve keep-unknown stance);
      4. return {label: band_rows.head(N)} in the EXISTING order (no re-sort).

    Args:
      ranked_df : the general-pool ranking (a DataFrame with a 'source' column in rank
                  order, e.g. resdic['postRank']).
      cdx_df    : the fundamentals frame carrying marketCap (+ reportedCurrency once the
                  next full fetch has run) used for both dedup and USD market cap.
      names     : optional {symbol: name} for the dedup name+shares edge (edges A/C work
                  without it).

    Returns dict:
      bands            {label: DataFrame (<= head_N rows, existing order)}
      band_counts      {label: full member count BEFORE head(N)}
      band_head_n      {label: the requested top-N for that band}
      band_selective   {label: bool} -- False when count <= head_N, i.e. the "top-N" is
                       ALL of the band's members and NOTHING was selected (audit M5). A
                       4-member Micro band presented as a "top-5" has ZERO selectivity and
                       must not be read as a shortlist; consumers LABEL or SUPPRESS it.
      band_note        {label: human-readable selectivity string} -- ships into the CSVs so
                       the file cannot be read out of context.
      unknown_mcap     # survivors with no USD market cap (routed to General)
      currency_pending True on pre-fetch data (reportedCurrency not yet flowed) -> the
                       USD banding is NOT trustworthy; consumers must LABEL or SKIP it.
      dropped_dupes    dedup_ranked audit trail [(dropped, survivor), ...]
    """
    labels = [lab for lab, *_ in MCAP_BANDS]
    head_n = {lab: N for lab, _lo, _hi, N in MCAP_BANDS}
    if not isinstance(ranked_df, pd.DataFrame) or 'source' not in getattr(ranked_df, 'columns', []) \
            or ranked_df.empty:
        empty = ranked_df.iloc[0:0] if isinstance(ranked_df, pd.DataFrame) else None
        return {'bands': {lab: empty for lab in labels},
                'band_counts': {lab: 0 for lab in labels},
                'band_head_n': head_n,
                'band_selective': {lab: False for lab in labels},
                'band_note': {lab: 'EMPTY band -- no members' for lab in labels},
                'unknown_mcap': 0, 'currency_pending': True, 'dropped_dupes': []}

    names = names or {}
    ranked_sources = list(ranked_df['source'])
    # AggScore is passed so an issuer's tied clone lines resolve by investability
    # instead of by sort stability (see dedup_ranked TIE-BREAK).
    _sc = ranked_df['AggScore'] if 'AggScore' in ranked_df.columns else None
    kept, dropped = dedup_ranked(ranked_sources, cdx_df, names, scores=_sc)

    # Order-preserving reduce of ranked_df to the deduped survivors (kept is already
    # rank-ordered and unique). drop_duplicates guards any accidental repeat source row.
    base = ranked_df.drop_duplicates('source', keep='first')
    df = base.set_index('source').reindex(kept).reset_index()

    mcu = marketcap_usd_by_source(cdx_df)            # source -> latest USD market cap
    pending = not currency_data_present(cdx_df)
    general_label = MCAP_BANDS[0][0]

    members = {lab: [] for lab in labels}
    unknown = 0
    for s in df['source']:
        lab = band_for_marketcap_usd(mcu.get(s))
        if lab is None:                               # unknown mcap -> General, counted
            lab = general_label
            unknown += 1
        members[lab].append(s)

    bands, band_counts, band_selective, band_note = {}, {}, {}, {}
    for label, lo, hi, N in MCAP_BANDS:
        rows = df[df['source'].isin(set(members[label]))]   # preserves df (rank) order
        cnt = int(len(rows))
        band_counts[label] = cnt
        bands[label] = rows.head(N).reset_index(drop=True)
        # MIN-N SELECTIVITY LABEL (audit M5, 2026-07-19). head(N) on a band with <= N
        # members selects NOTHING -- it just re-lists the whole band. Shipping that as a
        # "top-5" misrepresents zero selectivity as a shortlist, and the thin bands (Micro
        # especially: only ~4 names under $50M exist in this universe) are exactly where it
        # bites. The count was already stored; this makes the CONSEQUENCE explicit and
        # carries it into the emitted files.
        band_selective[label] = cnt > N
        if cnt == 0:
            band_note[label] = 'EMPTY band -- no members'
        elif cnt <= N:
            band_note[label] = (f'NOT A SELECTION: all {cnt} member(s) of {label} shown '
                                f'(top-{N} requested, band has only {cnt})')
        else:
            band_note[label] = f'top-{N} of {cnt} member(s) in {label}'

    return {'bands': bands, 'band_counts': band_counts, 'band_head_n': head_n,
            'band_selective': band_selective, 'band_note': band_note,
            'unknown_mcap': int(unknown),
            'currency_pending': bool(pending), 'dropped_dupes': dropped}


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
    comps, latest, _val = _issuer_components(syms, cdx_df, names)

    from collections import Counter
    def _key(s):
        # THE shared "most investable line" key (module-level _investability_key), so
        # dedup_ranked's tie-break and this survivor rule can never diverge.
        return _investability_key(s, _val, sector_map)

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
        # CATASTROPHIC: no sector map => empty REIT/Mining cohorts => Basic-Materials
        # (miners) and Real-Estate would silently leak back into the general pool,
        # defeating the whole carve while the output still LOOKS carved. Do NOT proceed
        # with a half-carved pool. Banner on BOTH streams, then RAISE so the postBo
        # guard trips the LOUD fallback (ships legacy un-carved WITH its banner) rather
        # than a silently-degraded carve. (Banner here too, in case partition_universe
        # is ever called directly, not via the postBo guard.)
        msg = ("carveOut: sector map empty/absent (looked for '%s' in CWD %s). "
               "Cannot carve REIT/Mining without it." % (sector_pickle, os.getcwd()))
        banner = ("\n" + "!" * 78 + "\n"
                  "!!! CARVE-OUT ABORTED -- SECTOR MAP MISSING/EMPTY !!!\n"
                  "!!! REIT & Mining CANNOT be carved; Basic-Materials/REITs would leak !!!\n"
                  "!!! into the general pool. Refusing to ship a silently half-carved   !!!\n"
                  "!!! general pool -- aborting the carve so the fallback is LOUD.       !!!\n"
                  "!!! " + msg + "\n"
                  + "!" * 78 + "\n")
        print(banner, file=sys.stderr, flush=True)
        print(banner, flush=True)
        raise RuntimeError(msg)

    industry_map = _load_industry_map(industry_pickle)
    if not industry_map:
        # DEGRADES SAFELY: FIN-2/FIN-3 falls back to the (weak) name-keyword rule. This
        # does NOT leak miners/REITs into the general pool, so PROCEED -- but make the
        # degradation unmistakable (loud warning on BOTH streams), never silent.
        wbanner = ("\n" + "!" * 78 + "\n"
                   "!!! CARVE-OUT WARNING -- INDUSTRY MAP MISSING/EMPTY !!!\n"
                   "!!! FIN-2 (managers) vs FIN-3 (banks/insurers) split DEGRADED to the !!!\n"
                   "!!! weak name-keyword rule; financial sub-cohorts may be mislabeled. !!!\n"
                   "!!! General-pool integrity is UNAFFECTED (no miner/REIT leak) so the !!!\n"
                   "!!! run PROCEEDS. Provide industrydic_fmp_*.pickle in the run CWD.    !!!\n"
                   + "!" * 78 + "\n")
        print(wbanner, file=sys.stderr, flush=True)
        print(wbanner, flush=True)

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

    # --- market-cap floor, applied in USD; keep unknown-mcap names ------------------
    # mcap_floor is a USD figure ($25M) but cdx_df['marketCap'] is in each company's
    # REPORTING currency, mixed across the universe (audit H-3/H8), so comparing the raw
    # field against it made the floor a different height on every exchange -- SEK ~$2.4M,
    # KRW ~$18k. Measured on the 2026-07-17 universe (7,752 names with a market cap):
    # 179 names passed the raw floor while being under $25M USD (31 of them under $5M),
    # and 25 names above $25M USD were floored OUT (GBP/EUR reporters just under the raw
    # cutoff). Both directions matter because the score deliberately rewards small caps
    # (mcapQuants w=0.080), so the leak-ins land where the tilt is strongest.
    # Suffix fallback ON: this floor runs on every name every run, so it must produce a
    # number for all of them; an unknown suffix yields rate 1.0 = the previous raw
    # behaviour, and a name with NO market cap is still KEPT (never dropped on missing).
    mcap_raw = bs['source'].map(fund['marketCap'])
    _mcu = marketcap_usd_by_source(cdx_df, allow_suffix_fallback=True)
    mcap = bs['source'].map(_mcu)
    mcap = mcap.where(mcap.notna(), mcap_raw)
    below = mcap.notna() & (mcap < mcap_floor)
    n_below = int(below.sum())
    n_unknown_mcap = int(mcap.isna().sum())
    below_sources = set(bs.loc[below, 'source'])
    bs_floored = bs[~below].reset_index(drop=True)
    # The currency change moves names IN and OUT of the universe, so NAME THEM -- two
    # integers is not "loud, never silent" for a universe change of this size, and the
    # suffix prior is wrong for some of them by construction (review M1). Same dated-CSV
    # treatment the share-class filter gets.
    _flip_out = bs.loc[below & mcap_raw.notna() & (mcap_raw >= mcap_floor), 'source']
    _flip_in = bs.loc[~below & mcap_raw.notna() & (mcap_raw < mcap_floor)
                      & mcap.notna(), 'source']
    print("carveOut floor: applied in USD (suffix-FX fallback where reportedCurrency "
          "absent) -- %d name(s) newly excluded that the raw-currency floor kept, "
          "%d name(s) newly kept that it wrongly excluded"
          % (len(_flip_out), len(_flip_in)), flush=True)
    if len(_flip_out):
        print("  newly EXCLUDED: %s" % ', '.join(sorted(_flip_out)), flush=True)
    if len(_flip_in):
        print("  newly KEPT: %s" % ', '.join(sorted(_flip_in)), flush=True)
    try:
        _fx_rows = pd.DataFrame({
            'source': list(_flip_out) + list(_flip_in),
            'direction': (['newly_excluded'] * len(_flip_out)
                          + ['newly_kept'] * len(_flip_in))})
        _fx_rows['marketCap_raw'] = _fx_rows['source'].map(fund['marketCap'])
        _fx_rows['marketCap_usd'] = _fx_rows['source'].map(_mcu)
        _fx_rows['suffix_fx_rate'] = _fx_rows['source'].map(_suffix_fx_to_usd)
        _fn = ('CurrencyFloorFlips_%s.csv'
               % pd.Timestamp.today().strftime('%Y-%m-%d'))
        _fx_rows.to_csv(_fn, index=False)
        print('  currency-floor flip list written to: %s' % _fn, flush=True)
    except Exception as _e:
        print('  WARNING: could not write currency-floor flip list (%s)' % _e, flush=True)

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
