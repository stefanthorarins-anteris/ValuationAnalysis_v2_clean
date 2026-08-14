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

import math
import os
import re
import sys
import numpy as np
import pandas as pd

import scoringWeights as sw     # SINGLE SOURCE OF TRUTH for every scoring weight
import transfer_utils as _tu   # EVIDENCE_DIR: where the run's evidence CSVs are written

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

# Sector-map COVERAGE thresholds for partition_universe's guard (2026-08-02).
# Measured, not chosen: the live maps cover 87.1% of the 2026-01-08 panel's 9,012
# sources; a map authored by the 142-name test universe would cover 1.6% of that same
# pool. Two orders of magnitude apart, so these cuts separate "healthy" from "wrong
# artifact" without firing on a normal run. See the guard for the full reasoning.
SECTOR_COVERAGE_ABORT_BELOW = 0.50
SECTOR_COVERAGE_WARN_BELOW = 0.75
SECTOR_COVERAGE_HEALTHY_REF = 0.871
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
# THE VECTORS AND THEIR PROVENANCE NOW LIVE IN `scoringWeights.py`, the single source of
# truth for every scoring weight in the repo (single-source refactor, 2026-08-02).  That
# is where to edit a cohort weight, and where the S5 (priceGrowth / DcfToPrice zeroed in
# every cohort) and S7 (normalised to Sigma|w| = 1) rulings are recorded, verbatim --
# including the KNOWN OPEN ISSUE that `BoScore` carries 0.1 in all five cohorts while it
# is 0.000 in the general vector (the CEO has not ruled; preserved exactly).
#
# Both names are RE-EXPORTED here unchanged, because every consumer reads them off this
# module: postBo (`co.COHORT_WEIGHTS.get(label)`), tune_run (cohort priors),
# test_post_fetch_hardening, reviewReference.  Keys are the REAL metric keys from
# createDicts.getPostDict() (all 25 as of 2026-08-06) and are enforced to be exactly that set by
# scoringWeights._validate().  Threaded into postBoScoreRanking(weight_override=...) per
# cohort; the general/main pool uses NO override (default weights).  weight 0 -> metric
# dropped from AggScore (constant/neutral in rankOfRanks); does not change cohort
# membership.
COHORT_WEIGHTS_RAW = sw.COHORT_WEIGHTS_RAW   # kept for provenance / A-B
COHORT_WEIGHTS = sw.COHORT_WEIGHTS           # normalised to Sigma|w| = 1
# The old private helper `_normalise_cohort_weights(vectors)` (a dict OF vectors -> dict of
# normalised vectors) is gone; scoringWeights.normalise(vector) is the per-vector primitive
# and COHORT_WEIGHTS is already normalised. No alias is provided on purpose -- the two have
# different signatures, so a same-named shim would be a footgun. Nothing outside this
# module ever called it.

# The cohort LABELS are owned by THIS module (the constants above); scoringWeights has to
# spell them as literals because carveOut imports it, so importing back would be a cycle.
# Assert the two agree at import rather than discovering it as a missing weight_override
# -- postBo does `co.COHORT_WEIGHTS.get(label)`, which returns None on a mismatch, and a
# None override means the cohort silently ranks on the GENERAL vector.
_EXPECTED_COHORT_LABELS = {'REIT', 'Mining', FIN1_VEHICLE, FIN2_MANAGER, FIN3_BALSHEET}
if set(COHORT_WEIGHTS) != _EXPECTED_COHORT_LABELS:
    raise RuntimeError(
        'carveOut: cohort labels and scoringWeights.COHORT_WEIGHTS_RAW disagree. '
        'carveOut expects %s, scoringWeights supplies %s. A cohort with no weight '
        'vector silently ranks on the GENERAL weights.'
        % (sorted(_EXPECTED_COHORT_LABELS), sorted(COHORT_WEIGHTS)))

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

# --- FX -> USD : the SANITY BAND, no longer the rate SOURCE (2026-08-08) ------------
# cdx_df['marketCap'] is stored in each company's REPORTING currency, MIXED across the
# universe (verified: DORO.ST in SEK ~962M ~= $92M USD; FRES.L reports USD), so banding
# on the raw field would misband every non-USD name. We convert to USD via the captured
# reportedCurrency.
#
# ###################################################################################
# ## WHICH CURRENCY DRIVES THIS CONVERSION -- READ BEFORE TOUCHING ANYTHING HERE.   ##
# ## THERE ARE **TWO** CURRENCIES IN THIS CODEBASE AND THEY ARE NOT THE SAME FIELD: ##
# ##                                                                               ##
# ##   reportedCurrency  the STATEMENT currency (income statement / balance sheet). ##
# ##                     `marketCap` is denominated in THIS one.  Proven panel-wide:##
# ##                     FMP's own `pbRatio` equals marketCap / totalStockholders-  ##
# ##                     Equity to a median of 1.0000 across 20 currencies,         ##
# ##                     including 7,001 KRW rows.                                  ##
# ##   profile `currency` the TRADING currency of a LISTING LINE, captured into the ##
# ##                     volavgdic entry by findAllSectors.py (as of 90b0d5f).      ##
# ##                     It differs for EVERY ADR and cross-listing -- SHEL.L quotes##
# ##                     in pence and reports USD.                                  ##
# ##                                                                               ##
# ## WIRING THE PROFILE `currency` AS THE CONVERTER FOR `marketCap` WOULD REINTRODUCE##
# ## THE EXACT UNIT MISMATCH THAT HALTED THE LIQUIDITY FLOOR.  Do not change which  ##
# ## currency field drives this conversion.  The not-wired note on the trading side ##
# ## lives in findAllSectors.py, which is NOT where an FX author looks -- hence this##
# ## copy, here, next to the table.                                                 ##
# ##                                                                               ##
# ## SECOND-ORDER, MEASURED 2026-08-09 -- the swap would also be a LIVE-PATH change.##
# ## On a 100-row profile sample, 11 LSE lines carry TRADING currency `GBp`, while  ##
# ## ZERO sources REPORT in `GBp` (fx_rates MINOR_UNITS note, 2026-08-07 panel).  So##
# ## feeding profile `currency` in would fire the pence minor-unit path for the     ##
# ## FIRST TIME EVER -- correct on that sample (0.013490), but newly live, not inert.##
# ###################################################################################
#
# THESE CONSTANTS ARE NO LONGER A RATE SOURCE IN PRODUCTION.  They were a hardcoded,
# UNDATED snapshot: measured 2026-08-08 the median absolute drift against live rates was
# ~7% and 13 currencies were past 10% (TRY -30.1% ... CHF +10.5%), which recomputed on the
# 2026-08-07 CUR3K panel gets 11 universe-membership decisions wrong at the $25M floor
# (7 EUR names wrongly DELETED, 4 wrongly kept) and puts 32 names in the wrong band.
#
# They now serve TWO purposes and only two:
#   1. THE SUPPORTED SET -- a currency with no entry here is not convertible.
#   2. THE SANITY BAND -- every LIVE rate must land within +-fx_rates.FX_SANITY_BAND of
#      its constant or it is REFUSED and treated as absent.  That is the one new failure
#      mode a live feed has and a constant does not: a vendor-side unit flip or an
#      inverted quote.  It is a UNITS check, not an accuracy check -- TRY sat ~30% from
#      its constant on real 2026-08-08 data (re-seeded 2026-08-09, see the log below), so
#      a band tight enough to police accuracy would reject good rates.
#
# The live table is installed by fx_rates.install_for_run (one v3/quotes/forex call at
# run start) via set_live_fx_rates below.  See fx_rates.py for the whole contract,
# including why a missing/stale rate routes into the unknown-currency path rather than
# falling back to these numbers.
#
# 'PEN' and 'MAD' were ADDED 2026-08-08 (quotable, clean, free once the feed exists);
# their constants are seeded from that day's live quote, so they are anchors, not history.
# 'ARS' is DELIBERATELY ABSENT -- see fx_rates.ABSTAIN_CURRENCIES for the reasoning
# (the rate is fine; our ARS statement data is broken by three orders of magnitude).
# As of 2026-08-08 ARS reporters are also EXCLUDED FROM THE UNIVERSE outright by
# currency_exclusions.EXCLUDED_CURRENCIES (applied in data_quality PASS 0b), because
# abstaining on the rate fixed only the currency while every metric downstream of those
# statements stayed contaminated.  The abstention above is retained as the backstop for
# paths that never reach the quality filter -- do not delete it as redundant.
FX_TO_USD = {
    'USD': 1.0, 'EUR': 1.08, 'GBP': 1.27, 'GBp': 0.0127, 'GBX': 0.0127,
    'CHF': 1.12, 'JPY': 0.0067, 'SEK': 0.095, 'NOK': 0.093, 'DKK': 0.145,
    'CAD': 0.73, 'AUD': 0.66, 'NZD': 0.61, 'HKD': 0.128, 'SGD': 0.74,
    'CNY': 0.138, 'CNH': 0.138, 'INR': 0.012, 'KRW': 0.00073, 'TWD': 0.031,
    'ZAR': 0.054, 'BRL': 0.185, 'MXN': 0.055, 'PLN': 0.25, 'ILS': 0.33326,
    'AED': 0.272, 'SAR': 0.267, 'THB': 0.028, 'IDR': 0.000063, 'TRY': 0.020969,
    'RUB': 0.011, 'CZK': 0.043, 'HUF': 0.0028, 'PHP': 0.017, 'MYR': 0.22,
    'ISK': 0.0072, 'PEN': 0.29656, 'MAD': 0.10738,
}

#  THE ANCHORS AGE (flagged 2026-08-08).
#  The band is measured against a FIXED constant, so a currency in a sustained trend walks
#  toward the edge and is eventually REFUSED while being perfectly correct.  That must not
#  be DISCOVERED as a name disappearing, so fx_rates warns when a rate gets within
#  FX_BAND_EDGE_WARN of the edge.  The remedy is a one-line DATED re-seed of the constant
#  above -- NOT a wider band.  A wider band buys drift tolerance by giving up the only
#  thing the band detects (an order-of-magnitude unit flip), so it trades a real check for
#  a cosmetic one.  This is recorded here, in fx_rates' band constants, in the drift
#  warning text and in the shipped FxRates CSV; the four must stay in agreement.
#
#  ============ RE-SEED LOG -- how old is each anchor, without archaeology ============
#  Undated entries above are the ORIGINAL hardcoded snapshot; treat them as of unknown
#  age (measured 2026-08-08, their median absolute drift against live rates was ~7% and
#  13 of them were past 10%).  Anything re-seeded is dated here, with the observation it
#  came from, so the next reader can see the age of the number rather than infer it.
#
#    2026-08-08  PEN 0.29656, MAD 0.10738   seeded on ADD from that day's live quote.
#    2026-08-09  TRY 0.030 -> 0.020969      was 0.699x its anchor = 60.2% of the half-band
#                                           consumed, and the ONLY `[fx] ANCHOR` warning on
#                                           the 2026-08-09 live call.  Re-seeded from
#                                           FxRates_2026-08-09.csv (in output/ when it was written; root since 2026-08-10) (TRYUSD
#                                           0.02096854, quote_age 0.32d, status ok,
#                                           reciprocal-checked).
#    2026-08-09  ILS 0.27  -> 0.33326       the runner-up at 46.9% consumed -- UNDER the
#                                           50% warn threshold, so it produced NO warning
#                                           and would have become the next one silently.
#                                           Re-seeded in the same pass, from the same
#                                           artifact (ILSUSD 0.33326, status ok), because
#                                           doing it when it warns means doing it under
#                                           time pressure before a fetch.
#  NO API CALL WAS MADE FOR THIS RE-SEED.  Both numbers come from the FxRates CSV the
#  pre-flight gate run already wrote, which is the same feed the run would have used.
#  ====================================================================================


# --- the run's FX SOURCE STATE ------------------------------------------------------
# THREE states, because two would force a choice between breaking every offline tool and
# letting production fall back to the undated constants:
#
#   'unset'   no feed was ever attempted -> FX_TO_USD is used, exactly today's behaviour.
#             This is the OFFLINE state: the test suite, baseline_tools/, any hand-run
#             script.  Production never sits here -- Sbocker.main always installs.
#   'live'    a feed resolved.  ONLY the live table answers; a currency absent from it
#             (missing pair / stale quote / refused by the sanity band) resolves to None,
#             i.e. UNKNOWN CURRENCY, never to its constant.
#   'failed'  a feed was attempted and produced nothing usable.  EVERY currency resolves
#             to None, and a materialized `marketCap_usd` column is refused as well (it
#             was computed with whatever FX was live when the panel was fetched, so
#             honouring it would re-admit exactly the stale number this change removes).
#
# The point of the 'failed' state is the CEO's load-bearing requirement: ON FX FAILURE THE
# FLOOR DOES NOT RUN ON THE OLD CONSTANTS.  It runs on nothing, loudly -- which is the
# already-built unresolvable-reportedCurrency path (see partition_universe's floor block).
_FX_STATE = 'unset'
_LIVE_FX = {}
_LIVE_FX_META = {}


def set_live_fx_rates(mapping, meta=None):
    """Install the run's live {currency: usd_per_unit} table.  An EMPTY mapping is a
    FAILURE, not an install -- it is routed to mark_fx_unavailable so the two can never
    be confused by a caller passing {}."""
    global _FX_STATE, _LIVE_FX, _LIVE_FX_META
    if not mapping:
        return mark_fx_unavailable('empty rate table', meta=meta)
    _LIVE_FX = dict(mapping)
    _LIVE_FX_META = dict(meta or {})
    _FX_STATE = 'live'
    return _FX_STATE


def mark_fx_unavailable(reason, meta=None):
    """Record that the feed was ATTEMPTED and failed.  Distinct from 'unset': it makes
    every currency unknown instead of quietly reverting to the hardcoded snapshot."""
    global _FX_STATE, _LIVE_FX, _LIVE_FX_META
    _LIVE_FX = {}
    _LIVE_FX_META = dict(meta or {})
    _LIVE_FX_META.setdefault('failure_reason', reason)
    _FX_STATE = 'failed'
    return _FX_STATE


def clear_live_fx_rates():
    """Back to 'unset' (offline/constants).  For tests and offline tools ONLY."""
    global _FX_STATE, _LIVE_FX, _LIVE_FX_META
    _FX_STATE, _LIVE_FX, _LIVE_FX_META = 'unset', {}, {}
    return _FX_STATE


def fx_source_state():
    """'unset' | 'live' | 'failed' -- see the block above."""
    return _FX_STATE


def live_fx_meta():
    """The provenance dict fx_rates.install_for_run produced, for RunProvenance.

    Read from MODULE STATE rather than from resdic on purpose: resdic is rebuilt from a
    LOADED pickle on the -loadbometric / -loadboresults paths, and a loaded pickle can
    carry a PREVIOUS run's fx_rates_as_of.  Module state always describes THIS process."""
    return dict(_LIVE_FX_META)


def live_fx_table():
    """A copy of the installed table (empty when unset/failed)."""
    return dict(_LIVE_FX)


# --- COARSE exchange-suffix -> reporting-currency fallback --------------------------
# ONLY for use where the alternative is treating a mixed-currency marketCap AS IF it
# were USD -- i.e. the $25M universe FLOOR, which runs unconditionally today and therefore
# mis-selects rather than degrades (audit H-3/H8). The floor is now its LAST caller.
#
# THE mcapQuants SIZE TILT NO LONGER PASSES True (register D-5, CEO 2026-08-06). It did
# while the metric was a POOL QUARTILE, where a relative cut partly cancels a systematic
# currency error; `marketCapRevQuants` is now ABSOLUTE USD bands and absolute edges do not
# cancel, so a won-reporting issuer would read as >= $10B whatever its size. That metric
# is therefore gated on the real reportedCurrency and scores NEUTRAL when it is absent --
# see stage2_metrics._mcap_for_quants. The asymmetry with the floor is DELIBERATE and
# OPEN: the floor still guesses, because a name with no derivable USD size would otherwise
# have to be either dropped or admitted unscreened. It is the remaining guess in the
# currency story.
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


def _fx_to_usd(currency, fx=None):
    """USD-per-unit for a reportedCurrency code, or None if missing / unknown / not
    resolvable from the run's FX source.

    None is the LOAD-BEARING return value: it is what makes the name unknown-mcap, which
    the $25M floor KEEPS and the bands SKIP.  A stale or refused live rate returns None
    for exactly that reason -- it must be indistinguishable from a currency we never knew,
    because it is the same kind of wrong number.

    `fx` overrides the module state with an explicit {currency: rate} table (used by the
    offline PIT path and by tests); it does NOT fall back to the constants."""
    if not isinstance(currency, str):
        return None
    code = currency.strip()
    if fx is not None:
        r = fx.get(code)
        return None if r is None else float(r)
    if _FX_STATE == 'unset':
        return FX_TO_USD.get(code)
    r = _LIVE_FX.get(code)          # 'live' -> the live table; 'failed' -> {} -> None
    return None if r is None else float(r)


def _is_pit_table(fx):
    """True for a dated (point-in-time) FX table -- anything exposing `rate_for`."""
    return fx is not None and hasattr(fx, 'rate_for')


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
    #  Routed through _fx_to_usd so an OFFLINE tool that opted into the suffix guess AND
    #  installed a live feed converts with the live rate rather than the sanity constant.
    #  An unresolvable rate still means 1.0 here = today's raw behaviour, which is this
    #  helper's whole contract (only a KNOWN suffix may move a name).
    #
    #  EXCEPT UNDER A DEAD FEED (F-5, reviewer 2026-08-08).  With _FX_STATE == 'failed'
    #  every currency resolves to None, so a KNOWN suffix would fall to 1.0 and be read as
    #  raw-as-USD -- DORO.ST would go from ~$9.5M to ~$100M, a wrong number wearing a right
    #  label, in the one state where we have explicitly decided not to guess. Return None
    #  instead (-> NaN -> unknown) so the dead-feed decision is not silently reversed by
    #  the fallback. Offline-only and unreachable from the pipeline today (no caller passes
    #  allow_suffix_fallback=True), guarded now rather than left for whoever wires it next.
    if _FX_STATE == 'failed':
        return None
    r = _fx_to_usd(cur)
    return 1.0 if r is None else r


def marketcap_usd_series(cdx_df, allow_suffix_fallback=False, fx=None):
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
    unknown/absent suffix meaning rate 1.0 = raw marketCap.

    NO PIPELINE CALLER PASSES True ANY MORE (2026-08-06). The three consumers that key off
    an absolute USD number now all refuse to guess: the band emission
    (`partition_by_marketcap`) gates on `currency_data_present`, the `mcapQuants` size tilt
    scores an unknown currency NEUTRAL (register D-5 -- absolute bands cannot absorb a
    currency guess; see stage2_metrics._mcap_for_quants), and the $25M universe FLOOR was
    the last guesser until it too was gated: it now excludes only names whose reporting
    currency really resolves, and KEEPS the unknowns, because a wrong exclusion is
    invisible and unrecoverable while a wrong inclusion still faces the whole filter (MD +
    senior-dev joint call, CEO-delegated -- see partition_universe's floor block).
    The flag survives for OFFLINE TOOLING that must produce a number for every row and is
    explicit about the guess (baseline_tools/run_corrected_current.py); production must not
    use it.

    fx (2026-08-08): the FX source to convert with.
      * None                     -> the run's installed source (see fx_source_state()).
      * {currency: rate}         -> that flat table, no fallback to the constants.
      * an object with rate_for(currency, date) -> POINT-IN-TIME conversion, each row
        converted at ITS OWN date's rate (fx_rates.PitFxTable).  This is what removes the
        look-ahead flavour from grading a 2021 market cap with today's spot; it requires a
        `date` column and resolves to NaN where the dated series does not reach."""
    cols = getattr(cdx_df, 'columns', [])
    if cdx_df is None or 'marketCap' not in cols:
        return pd.Series(np.nan, index=getattr(cdx_df, 'index', None))
    mc = pd.to_numeric(cdx_df['marketCap'], errors='coerce')
    out = None
    if 'reportedCurrency' in cols:
        if _is_pit_table(fx):
            #  PIT: the rate is a function of (currency, row date), so it cannot be a
            #  column map.  A row with no usable date resolves to None -> NaN -> unknown.
            dts = (pd.to_datetime(cdx_df['date'], errors='coerce') if 'date' in cols
                   else pd.Series(pd.NaT, index=cdx_df.index))
            rate = pd.Series(
                [fx.rate_for(c, d) for c, d in zip(cdx_df['reportedCurrency'], dts)],
                index=cdx_df.index, dtype='float64')
        else:
            rate = cdx_df['reportedCurrency'].map(
                lambda c: _fx_to_usd(c, fx=fx)).astype('float64')
        out = mc * rate
    elif 'marketCap_usd' in cols and _FX_STATE != 'failed':
        #  Materialized at ingest (belt-and-suspenders).  REFUSED when the feed was
        #  attempted and failed: that column was computed with whatever FX was live when
        #  the panel was fetched, and honouring it on a run whose own FX is dead would
        #  re-admit the stale number by the back door -- i.e. exactly the "floor runs on
        #  old constants" outcome this design forbids.
        out = pd.to_numeric(cdx_df['marketCap_usd'], errors='coerce')
    else:
        out = pd.Series(np.nan, index=cdx_df.index)
    if allow_suffix_fallback and 'source' in cols:
        srate = cdx_df['source'].map(_suffix_fx_to_usd).astype('float64')
        out = out.where(out.notna(), mc * srate)
    return out


def currency_data_present(cdx_df, fx=None):
    """True only when currency data is actually USABLE -- i.e. reportedCurrency resolves
    to a known FX rate for at least one row, or a materialized marketCap_usd carries at
    least one finite value. Column PRESENCE alone is NOT enough: an all-NaN column (e.g.
    reportedCurrency coerced to NaN by a numeric-cast, or an empty materialization) would
    otherwise masquerade as 'present' and suppress the pending banners while every name
    silently routes to General. This is the backstop that keeps 'nothing wrong ships'
    true even if the ingest string-preservation regresses (CEO 2026-07-17).

    IT MUST MIRROR marketcap_usd_series EXACTLY (tightened 2026-08-08).  The two used to
    disagree in one reachable case: this function fell through to `marketCap_usd` when
    `reportedCurrency` was PRESENT but resolved nothing, while the series only consults
    `marketCap_usd` when the reportedCurrency COLUMN IS ABSENT.  That divergence was
    harmless while FX was a constant table (reportedCurrency always resolved), and becomes
    live the moment a rate can fail: a post-fetch panel + a dead FX feed would have
    reported "currency present" (via the stale materialized column) while every name
    converted to NaN -- i.e. the floor would have printed "applied, 0 excluded" instead of
    the NOT-ENFORCED banner.  So: when the column exists, the column decides.

    AND IT MUST REQUIRE `marketCap` TOO (F-3, reviewer 2026-08-08).  The mirror was still
    partial: `marketcap_usd_series` returns an all-NaN series the moment `marketCap` is
    absent, but this function never looked at that column -- so a frame carrying
    `reportedCurrency` and no `marketCap` reported "currency present" against a series
    that could not produce a single number.  Same shape as the divergence above, one
    column over."""
    cols = getattr(cdx_df, 'columns', [])
    if cdx_df is None:
        return False
    if 'marketCap' not in cols:
        return False
    if 'reportedCurrency' in cols:
        try:
            if _is_pit_table(fx):
                return bool(len(marketcap_usd_series(cdx_df, fx=fx).dropna()))
            return bool(cdx_df['reportedCurrency'].map(
                lambda c: _fx_to_usd(c, fx=fx)).notna().any())
        except Exception:
            return False
    if 'marketCap_usd' in cols and _FX_STATE != 'failed':
        try:
            if pd.to_numeric(cdx_df['marketCap_usd'], errors='coerce').notna().any():
                return True
        except Exception:
            pass
    return False


def currency_coverage(cdx_df, fx=None):
    """(n_covered, n_sources, fraction) -- how much of the PANEL gets a USD market cap.

    THE BOOLEAN IS NOT ENOUGH (F-2, reviewer 2026-08-08).  `currency_data_present` answers
    "did ANY name resolve", which is what the floor needs to decide whether to run at all.
    It says nothing about HOW MUCH of the universe the floor then covers -- and with, say,
    only {USD, KRW} resolving, the run installs 'live', this returns True, no banner fires
    and `floor_enforced: True` is stamped while barely half the names have a USD cap (EUR
    alone is 23.8% of the universe). Nothing is wrongly deleted, but a `floor_enforced`
    label on a half-floored universe is the label-means-something-else defect this project
    keeps producing. Callers report the FRACTION, and it ships in the artifact.

    Counts SOURCES, not rows: the floor is a per-name decision.

    Built on `marketcap_usd_series` DIRECTLY rather than on `marketcap_usd_by_source`,
    which hard-requires a `date` column and raises without one. This function is called
    from the floor block on the critical path of a ~12-hour run and from the provenance
    sidecar; neither may be the thing that crashes it, and a diagnostic that needs a column
    its subject may not carry is not a diagnostic."""
    cols = getattr(cdx_df, 'columns', [])
    if cdx_df is None or 'source' not in cols or not len(cdx_df):
        return 0, 0, 0.0
    n_total = int(cdx_df['source'].nunique())
    if not n_total:
        return 0, 0, 0.0
    usd = marketcap_usd_series(cdx_df, fx=fx)
    n_cov = int(cdx_df.loc[usd.notna(), 'source'].nunique())
    return n_cov, n_total, (n_cov / float(n_total))


def marketcap_usd_by_source(cdx_df, as_of=None, allow_suffix_fallback=False, fx=None):
    """source -> latest USD market cap (latest non-NaN row). If `as_of` is given,
    restrict to date <= as_of, i.e. the POINT-IN-TIME market cap as-of that date.
    Returns {} when the frame is unusable. Used by partition_by_marketcap (latest,
    fallback OFF) and by the PIT beat-rate grading (as_of=buy).

    allow_suffix_fallback is forwarded to marketcap_usd_series -- pass True only from
    the universe floor / size-tilt callers that must produce a number for every name.

    fx is forwarded too.  Passing a dated (PIT) table together with `as_of` is the
    correct point-in-time read: the market cap is the last one reported on or before
    `as_of`, converted at the rate that was live ON ITS OWN DATE -- not today's."""
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
        df, allow_suffix_fallback=allow_suffix_fallback, fx=fx).values
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


#  --- THE SECTOR MAP IN THIS REPO IS NOT THE ONE THE RUNS USE.  READ THIS BEFORE MEASURING
#  --- ANYTHING SECTOR-RELATED OFFLINE.  (2026-08-10) --------------------------------------
#  `sectorsdic_fmp.pickle` is .gitignore'd and REBUILT BY THE RUN, so the copy sitting in a
#  working tree is whatever that machine last built -- and the two that exist right now are
#  not the same KIND of object:
#
#      repo copy (2025-12-10)   40 keys / 40,164 symbols   'Banking', 'Chemicals', 'Airlines',
#                                                          'Biotechnology' ... an INDUSTRY
#                                                          taxonomy misfiled as sectors
#      run copy  (2026-08-07)   11 keys / 11,479 symbols   the FMP SECTOR taxonomy this module
#                                                          is written against
#
#  THIS IS NOT STALENESS, IT IS A DIFFERENT TAXONOMY, and it silently breaks the carve: the
#  cohort constants below (`FINANCIAL_SECTOR` = 'Financial Services') cannot match a map whose
#  banks are filed under 'Banking', so FIN-3 under-carves wholesale.  Measured on the 08-10
#  panel: the repo map gives general 1,489 / REIT 67 / Mining 217 where the run produced
#  1,388 / 77 / 265.
#
#  IT CHANGES INDIVIDUAL VERDICTS, not just counts.  `MAS` reads 'Basic Materials' under the
#  run's map and 'Industrials' under the repo's; `B` and `ABX.TO` are BOTH 'Basic Materials'
#  under the run's (no conflict at all) and split Industrials/Basic Materials under the repo's;
#  `AFG`, `AFGB` and `AFGE` are all 'Financial Services' under the run's.  Two independent
#  offline measurements of the SAME change disagreed for exactly this reason and neither was
#  wrong -- they were reading different worlds.
#
#  WHAT TO DO: measure against the map the run actually used.  `Sbocker` ships all four profile
#  maps, so the run's copy travels with its outputs (the 2026-08-07 build is the one the 08-10
#  run used -- it skipped the rebuild, which is what `-force_rebuild_maps` now exists to
#  override).  Do not assume the working tree's copy is it.
#
#  --- AMBIGUOUS SECTOR -> GENERAL (CEO, 2026-08-10) ---------------------------------------
#  The sector an issuer is given when its member lines DISAGREE and no tag has a plurality.
#  It is a REAL, non-sentinel string so it survives `_is_known_sector` (an ambiguous issuer is
#  not an UNMAPPED one -- we know its tags, we cannot choose between them), and it matches
#  none of REIT_SECTOR / MINING_SECTOR / FINANCIAL_SECTOR, so `classify` routes it to
#  'general' by its existing default rather than by a new branch.
AMBIGUOUS_SECTOR = 'Ambiguous (sector conflict)'


#  REPO-ROOT anchor for the undated data pickles (fix, 2026-07-27).
#  `sectorsdic_fmp.pickle` was resolved as a BARE RELATIVE PATH, i.e. against the CALLER'S
#  CWD.  Every shipped caller happens to run from the repo root, so this never bit in
#  production -- but any tool invoked from a subdirectory (e.g. `python baseline_tools/x.py`
#  executed while cd'd into baseline_tools) silently got an EMPTY sector map, which makes
#  partition_universe raise, which sends its callers down their carve FALLBACK path and
#  emits a NON-CARVED, NON-DEDUPED top-100 under the normal filenames.  The data file lives
#  next to this module, so resolve it from here and stop depending on where the process
#  happens to be standing.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_repo_data(path):
    """Resolve an undated repo data file: absolute/found-as-given wins, else repo-root."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    cand = os.path.join(_MODULE_DIR, path)
    return cand if os.path.exists(cand) else path


def _load_sector_map(sector_pickle='sectorsdic_fmp.pickle'):
    """symbol -> sector, from the local sector pickle (dict sector -> [symbols]).
    Reuses the forensicFlags convention.  Returns {} if the pickle is absent.
    Path is resolved against the REPO ROOT, not the CWD (see _resolve_repo_data)."""
    sector_pickle = _resolve_repo_data(sector_pickle)
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
        # Glob the REPO ROOT, not the CWD -- same defect class as the sector map above.
        cands = sorted(glob.glob(os.path.join(_MODULE_DIR, 'industrydic_fmp_*.pickle')))
        if not cands:
            cands = sorted(glob.glob('industrydic_fmp_*.pickle'))
        path = cands[-1] if cands else None
    else:
        path = _resolve_repo_data(path)
    if not path or not os.path.exists(path):
        return {}
    d = pd.read_pickle(path)
    return dict(d) if isinstance(d, dict) else {}


def _load_isin_map(isin_pickle=None):
    """symbol -> ISIN (FMP v3/profile `isin` field), from the local ISIN pickle
    (already a flat dict symbol->isin, written by findAllSectors). If no explicit path
    is given, use the NEWEST isindic_fmp_*.pickle present. Returns {} if none found ->
    the survivor rule then falls back to exactly its pre-ISIN behaviour.

    Same shape as _load_industry_map on purpose: findAllSectors writes both maps from the
    SAME already-fetched profile payload, so they share the dated-glob convention. An
    ABSENT pickle is the normal state on every run before the next full profile build, so
    {} is a supported answer, not a degraded one.
    """
    import glob
    path = isin_pickle
    if not path:
        # Glob the REPO ROOT, not the CWD -- same defect class as the sector map above.
        cands = sorted(glob.glob(os.path.join(_MODULE_DIR, 'isindic_fmp_*.pickle')))
        if not cands:
            cands = sorted(glob.glob('isindic_fmp_*.pickle'))
        path = cands[-1] if cands else None
    else:
        path = _resolve_repo_data(path)
    if not path or not os.path.exists(path):
        return {}
    d = pd.read_pickle(path)
    return dict(d) if isinstance(d, dict) else {}


def _load_volavg_map(volavg_pickle=None):
    """symbol -> (volAvg, asof_date) from the local volAvg pickle, written by
    findAllSectors from the SAME already-fetched v3/profile payload as the ISIN map.

    TWO ON-DISK SHAPES ARE ACCEPTED, and the reason is the whole point of this wiring
    (register K-1, 2026-08-06):
      * {sym: {'volAvg': v, 'asof': 'YYYY-MM-DD'}}  -- the DATED shape, written from
        2026-08-06 onwards;
      * {sym: v}                                   -- the UNDATED shape the capture-only
        version wrote. Returned with asof None.
    Both are normalised to (value, asof) here so no consumer has to branch.

    EVERY OTHER KEY IN A DATED ENTRY IS IGNORED BY **THIS** LOADER, ON PURPOSE.
    findAllSectors folds the capture-only profile fields into the SAME entry so they share the
    one `asof` -- `price` and `currency` from 2026-08-08, and `isActivelyTrading`, `exchange`,
    `exchangeShortName`, `country`, `beta` from 2026-08-09.  The two `.get()`s here are what
    keep "a new captured field cannot move a dedup decision" a PROPERTY rather than an
    intention, and that property is INTACT: every consumer of THIS function still sees a
    2-tuple.

    THE BLANKET CLAIM "NONE OF THEM IS WIRED" WAS TRUE UNTIL 2026-08-13 AND IS NOT ANY MORE.
    `price` and `currency` are now READ -- by `_load_volavg_profile_map`, a SEPARATE accessor
    over the same file, for the traded-value report column (`dollar_volume_frame`).  Nothing
    reaches this loader's callers from there; the two views are deliberately distinct so the
    dedup path keeps its guarantee.  The remaining five fields are still unwired.  This
    paragraph is updated rather than left standing because a docstring that asserts a
    property the code no longer has is the defect, not the documentation of one.

    THE COROLLARY FOR ANYONE WIRING ANOTHER FIELD: this map merges NEVER-OVERWRITE at the
    ENTRY level, so an entry carried forward from an older run has an older KEY SET as well as
    an older `asof` -- a missing key means "not captured at that asof", never False/0.

    WHY DATING WAS REQUIRED BEFORE WIRING.  Unlike a sector, an industry or an ISIN,
    AVERAGE VOLUME IS TIME-VARYING, and findAllSectors merges MERGE-NEVER-OVERWRITE --
    correct (the map must not shrink) but it therefore carries FORWARD a stale reading for
    any symbol a run did not fetch. Without a per-entry date, a comparison between two
    lines of one issuer could silently be a comparison between a fresh reading and a
    six-month-old one -- i.e. between two market regimes -- and it would look exactly like
    a liquidity difference. The date is what lets `_volavg_liquidity_term` refuse that
    comparison instead of making it.

    Returns {} when no pickle exists, which is the state of every run before the next
    profile build -> the survivor rule falls back to exactly its pre-volAvg behaviour.
    """
    import glob
    path = volavg_pickle
    if not path:
        # Glob the REPO ROOT, not the CWD -- same defect class as the sector map above.
        cands = sorted(glob.glob(os.path.join(_MODULE_DIR, 'volavgdic_fmp_*.pickle')))
        if not cands:
            cands = sorted(glob.glob('volavgdic_fmp_*.pickle'))
        path = cands[-1] if cands else None
    else:
        path = _resolve_repo_data(path)
    if not path or not os.path.exists(path):
        return {}
    d = pd.read_pickle(path)
    if not isinstance(d, dict):
        return {}
    out = {}
    for sym, v in d.items():
        if isinstance(v, dict):
            out[sym] = (v.get('volAvg'), v.get('asof'))
        else:
            out[sym] = (v, None)
    return out


#  Loaded ONCE per process and memoised, because `_investability_key` is a sort key
#  called O(group size * log) times per group over ~1,300 groups -- re-globbing the repo
#  root there would be a disk hit per comparison. `None` means "not yet looked",
#  `{}` means "looked, nothing there" (the pre-fetch steady state).
_ISIN_MAP_CACHE = None
_VOLAVG_MAP_CACHE = None


def _volavg_map_cached():
    global _VOLAVG_MAP_CACHE
    if _VOLAVG_MAP_CACHE is None:
        try:
            _VOLAVG_MAP_CACHE = _load_volavg_map()
        except Exception:
            # A corrupt/unreadable volAvg pickle must not break the carve: degrade to the
            # pre-volAvg survivor rule, the same path as "no pickle at all".
            _VOLAVG_MAP_CACHE = {}
    return _VOLAVG_MAP_CACHE


def _isin_map_cached():
    global _ISIN_MAP_CACHE
    if _ISIN_MAP_CACHE is None:
        try:
            _ISIN_MAP_CACHE = _load_isin_map()
        except Exception:
            # A corrupt/unreadable ISIN pickle must not break the carve: degrade to the
            # pre-ISIN survivor rule, which is the same path as "no pickle at all".
            _ISIN_MAP_CACHE = {}
    return _ISIN_MAP_CACHE


def _clean_isin(x):
    """Normalised ISIN, or '' when unusable. Deliberately strict-ish but NOT a check-digit
    validator: this value is only ever compared for EQUALITY inside one issuer group, so a
    malformed-but-consistent string is harmless, while a `None`/NaN/'' must not group."""
    if not isinstance(x, str):
        return ''
    s = x.strip().upper()
    return s if len(s) >= 6 and s.isalnum() else ''


def _isin_plurality_term(sym, group, isin_map):
    """The ONE ordering signal an ISIN actually supports: how many of the group's members
    share THIS line's ISIN. Lower (more negative) sorts first, so the plurality security
    wins. Returns 0 -- a constant for every member, i.e. NO effect on the ordering -- when
    the ISIN cannot discriminate within this group.

    WHY PLURALITY AND NOT SOMETHING STRONGER. An ISIN (ISO 6166) is a country prefix + an
    OPAQUE national security number + a check digit. It carries NO security-type field, so
    NOTHING inside an ISIN says "this line is the common and that one is the preference
    share". Any rule of the form "lower NSIN is the common" or "the .KS-style class digit"
    is a national-numbering-agency convention, not a property of the identifier, and would
    be an invented rule dressed as a derivation -- so it is not here.
    What an ISIN DOES tell you is IDENTITY: two lines with the SAME ISIN are the same
    security on two venues; two lines with DIFFERENT ISINs are genuinely different
    securities. Plurality is the only ordering that follows from identity alone, and it
    follows for a real reason: an issuer's COMMON line is the one that gets cross-listed
    and depositary-programmed, so it tends to appear under one ISIN on SEVERAL lines of
    the group, while a certificat / preference / notes line appears once.
    CONSEQUENCE, STATED PLAINLY: in a 2-member group with 2 distinct ISINs this term is a
    TIE and decides nothing. That is exactly the shape of all three known failing groups
    (CBE.PA/RBT.PA, PREVA.AS/VALUE.AS, SMSD.L/SMSN.L), so THIS TERM DOES NOT FIX THEM.
    See design/stage1-veto-decisions.md (register K-1) for what would.

    ABSTENTION IS NOT "ISIN SAYS THE PICK DOES NOT MATTER". A group whose members all
    share one ISIN still contains an LSE IOB line the CEO cannot buy at size (register
    J-1) -- same security, different tradeability. Abstaining here hands that decision
    back to the canonicity markers, which are the terms that see it.

    NO-DATA MUST NOT BE A PENALTY  (reviewer, 2026-08-05 -- this was a real defect).
    The first cut returned the abstain value 0 for a member whose ISIN was missing or
    unusable, WHILE the discriminating branch emitted only values <= -1. 0 is therefore
    the WORST value in the term's own range, so a member the profile map merely LACKED
    sorted BELOW a member holding a SINGLETON ISIN -- the survivor of a mixed-availability
    group got decided by ISIN DATA AVAILABILITY rather than by the plurality signal. Two
    properties are now structural rather than incidental, and each is a case-exhaustive
    consequence of the code shape below (see the two tests in test_dedup_issuer):
      (a) NO-DATA TIES WITH A SINGLETON. "No ISIN for this line" and "an ISIN only this
          line holds" both mean exactly `ISIN TELLS US NOTHING HERE`, so they must be the
          same value. In the discriminating branch a member's value is
          `-max(count, 1)`, and an absent ISIN yields count 0 -> -1, the same -1 a
          singleton holder gets. Since every present ISIN has count >= 1, -1 is also the
          LEAST-PREFERRED value the branch can emit: absence can never rank below an
          active value, and can never rank above one either.
      (b) GROUP-WIDE ABSTENTION SURVIVES MIXED AVAILABILITY. The abstain test is now
          evaluated BEFORE this line's own ISIN is looked up, so the abstain/discriminate
          decision is a function of (isin_map, group) ALONE -- it cannot differ between
          two members of the same group. When the term abstains, every member gets a
          literal 0, including the unmapped ones.
    The "auditable as a value, not as a cancellation" property is kept deliberately: an
    abstention is the literal 0 in the emitted key, not a coincidental tie.
    """
    if not isin_map or len(group) < 2:
        return 0
    counts = {}
    for m in group:
        i = _clean_isin(isin_map.get(m))
        if i:
            counts[i] = counts.get(i, 0) + 1
    # ABSTAIN -> literal 0 for every member, so "this term did nothing" is auditable as a
    # value and not merely as a tie that happens to cancel. Two abstain cases:
    #   * one distinct usable ISIN in the group -- nothing to separate;
    #   * no ISIN held by more than one member (the 1-1 split of a 2-member group) --
    #     there IS no plurality, and a uniform -1 would only masquerade as a decision.
    # THIS TEST READS ONLY `group` AND `isin_map`, NEVER `sym` -- that is what makes the
    # abstention group-wide (property (b) above) instead of per-member.
    if len(counts) < 2 or max(counts.values()) < 2:
        return 0
    #  `max(..., 1)` IS THE FIX, not a defensive tidy: it floors a missing/unusable ISIN
    #  (count 0) onto the SINGLETON value -1 instead of letting it fall out at 0, which
    #  the branch above has already reserved for "this term did nothing".
    mine = _clean_isin(isin_map.get(sym))
    return -max(counts.get(mine, 0), 1)


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


# =========================================================================== #
#  ISSUER-LEVEL DE-DUPLICATION -- ONE LISTING PER ISSUER                        #
#  (rebuilt 2026-08-05; spec: design/dedup-policy.md, register E-4 + B-7)       #
#                                                                               #
#  Same-issuer lines (share-classes, preferreds, notes, cross-listings) occupy    #
#  multiple slots for ONE economic bet and -- worse -- a cross-listing can leak   #
#  past the carve-out when the sector map tags only the primary line. We collapse  #
#  to one line per issuer BEFORE the carve-out partition, so secondary listings    #
#  inherit the issuer's (propagated) sector.                                      #
#                                                                               #
#  THE ARCHITECTURE IS AN INVERSION, AND THAT IS THE WHOLE POINT.                 #
#  Everything before this was DETECTORS -- spot each non-canonical line and        #
#  REMOVE it from the universe -- and it failed repeatedly and DIFFERENTLY each    #
#  time: Korean preferreds (numeric suffix, no rule sees it), Continental          #
#  certificats (CBE.PA shares no prefix with RBT.PA), London depositary lines,     #
#  FMP-truncated company names (BWNB never joins BW's name group at all). The      #
#  rule is now: GROUP an issuer's lines, PICK the main one, DROP the rest.         #
#                                                                               #
#  WHY THAT IS BETTER -- KEEP THIS PROPERTY VISIBLE. The detector pile does not    #
#  get retired; it gets MOVED. It becomes an ORDERING INSIDE A GROUP               #
#  (`_non_canonical_tag`, consumed by `_investability_key`) instead of a REMOVAL   #
#  FROM THE UNIVERSE. As an ordering, a detector's false positive costs NOTHING:   #
#  we just pick the sibling. As a removal, a false positive DELETES A REAL         #
#  COMPANY, which is the expensive error this register kept booking. Same          #
#  heuristics, inverted consequence. That is why the picking rule below is allowed  #
#  to be aggressive, and why a marker that fails to fire is a DEGRADATION (a       #
#  slightly worse ticker) and never a DEFECT (a missing company).                  #
# =========================================================================== #
_ISSUER_STRIP = re.compile(
    r'\b(inc|incorporated|corp|corporation|company|co|plc|ltd|limited|lp|llc|'
    r'sa|s\.a|ag|se|nv|asa|ab|oyj|spa|the|group|ordinary|shares?|'
    r'class|senior|notes?|due|preferred|pref|units?|warrants?|adr|ads)\b', re.I)

#  `holding(s)` is stripped SEPARATELY, because exactly ONE caller must not strip it.
#
#  HEIA.AS "Heineken N.V." and HEIO.AS "Heineken Holding N.V." are GENUINELY SEPARATE
#  ISSUERS with separate statements -- Holding CONSOLIDATES the operating company and
#  reports materially different netIncome -- and both now arrive in the universe on the
#  AMS code the 2026-08-02 Europe fix restored, so this is LIVE, not hypothetical.
#  Stripping `Holding` makes their normalised names COLLIDE, which is the one way the
#  name+shares key (K3) could merge two real companies. So K3, and only K3, normalises
#  with keep_holding=True.
#
#  It is NOT free to strip it everywhere and NOT free to strip it nowhere; both
#  directions were MEASURED rather than assumed:
#    * DEDUP GROUPING is identical either way -- 2,842 merged pairs on the 2026-01-09
#      panel with `holding` stripped and 2,842 without. K3 loses nothing.
#    * getData_gen.filter_non_common_instruments RULE C is not: on the live 2026-08-04
#      table (51,703 type=='stock' lines) it catches 532 lines with `holding` stripped
#      and 531 without. The single loss is VRXAW, the Veraxa Biotech WARRANT, whose
#      enabling sibling VRXA is named "Veraxa Biotech HOLDING AG" -- rule A (no
#      "Warrants" token in the name) and rule B (no -P suffix) both miss it, so
#      deleting `holdings?` outright would put a warrant back into the pre-fetch
#      universe AND into the Stage-2 z-pool, which dedup does not police.
#  Hence a flag instead of a deletion: the dedup key gets the Heineken separation and
#  the share-class filter keeps its removal. The DEFAULT is today's behaviour, so every
#  pre-existing caller (rule C, isin_same_issuer_groups) is bit-for-bit unchanged.
_ISSUER_STRIP_HOLDING = re.compile(r'\bholdings?\b', re.I)


def _norm_issuer_name(x, keep_holding=False):
    """Normalise a company name for issuer matching.

    keep_holding=True preserves the `Holding`/`Holdings` token, so a consolidating
    holding company does NOT normalise onto its operating subsidiary (Heineken).  Only
    the K3 dedup key sets it; see the note above for why the default is the other way.
    """
    if not isinstance(x, str) or not x.strip():
        return ''
    s = x.lower()
    s = re.sub(r'\d+(\.\d+)?%.*$', ' ', s)      # drop "6.125% senior notes due 2026" tails
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    s = _ISSUER_STRIP.sub(' ', s)
    if not keep_holding:
        s = _ISSUER_STRIP_HOLDING.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()


#  RETIRED 2026-08-05: `_XLIST_FUND_TOL` / `_fund_near_equal`, the old edge C
#  (EXACT weightedAverageShsOut + revenue/netIncome/totalAssets near-equal within 5%).
#  Not simplified away -- REFUTED. That edge gated a tolerance on an exact share match,
#  and shares are THE ONE LISTING-DEPENDENT FIELD in the fingerprint (register B-7), so
#  the gate is what made the conjunction fail. It is strictly subsumed by K1 union K2
#  below: on the 2026-01-09 panel, K1+K2+K3 reproduces every pair the A/B/C edge set
#  found (ZERO regressions) and adds 247 more. Nothing is left that needs a tolerance,
#  so there is no longer a threshold parameter anywhere in the grouping -- which is
#  worth more than the last one or two pairs a tolerance would buy.


def _latest_raw(cdx_df, cols):
    have = [c for c in cols if c in cdx_df.columns]
    df = cdx_df[['source', 'date'] + have].copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['source', 'date'])
    df[have] = df.groupby('source')[have].ffill()
    return df.groupby('source')[have].last()


def _latest_period(cdx_df):
    """Per-source date of the row `_latest_raw` lands on -- the SAME sort and the SAME
    groupby-last, so the date this returns is the date of the values that function
    returns. Deliberately a second small pass rather than a column threaded through
    `_latest_raw`: that function ffills its value columns, and ffilling a DATE would
    report a period the line has no statement for."""
    df = cdx_df[['source', 'date']].copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['source', 'date'])
    return df.groupby('source')['date'].last()


#  K1 -- STATEMENT IDENTITY. FMP serves the ISSUER's own statements to every line it
#  lists, so these three together are an issuer identity. This is the old edge A MINUS
#  the listing-dependent share count, and that subtraction is the entire fix: for every
#  cross-listing / depositary pair the old scheme missed, revenue / netIncome /
#  totalAssets are BYTE-IDENTICAL and it is marketCap and weightedAverageShsOut that
#  wobble (0EDE.L/NXPI, 0IJ2.L/ES, 0KXS.L/RGLD, 4PG.DE/OTIS: all three fields exactly
#  0.0000 apart; 0LF0.L/TXT identical on all three but 4.96% apart on shares).
#  netIncome is REQUIRED and EXACT deliberately -- it is the field that separates a
#  parent from a consolidating holding company (Heineken N.V. vs Heineken Holding N.V.,
#  which reports roughly half the netIncome once minority interest is out). Weaken or
#  drop netIncome and the one live MUST-NOT-MERGE in the register becomes reachable.
_K1_COLS = ('revenue', 'netIncome', 'totalAssets')

#  K2 -- ISSUER AGGREGATE. marketCap is an issuer-level, currency-normalised number, so
#  it groups the FX-shifted cross-listings K1 cannot: 0R4M.L / LUG.TO / LUG.ST carry
#  byte-identical marketCap while their totalAssets differ by 85%. The floor keeps
#  zero/near-zero caps -- of which the panel has many, and which would otherwise all
#  collide into one bogus mega-group -- out of the key.
_K2_MARKETCAP_FLOOR = 1e6

#  ...AND MARKETCAP ALONE IS NOT AN IDENTITY. CORROBORATION IS REQUIRED (2026-08-13,
#  register N-1). The docstring below used to claim ZERO chance collisions on the
#  2026-01-09 panel. That claim did not survive contact with the CUR3K panels: on BOTH
#  the 2026-08-11 and 2026-08-13 runs, `ASY.PA` / `0OA7.L` (Assystem S.A., ISIN
#  FR0000074148) and `ELIOR.PA` (Elior Group S.A., ISIN FR0011950732) both carry
#  marketCap 640,500,000.0 TO THE EURO and were merged into one issuer. Nothing else
#  about them matches -- revenue 330.2M vs 3,179M, totalAssets 629.5M vs 3,763M, shares
#  15.0M vs 262.5M, price EUR47.41 vs EUR2.23 (21.2x apart) -- and ELIOR.PA won the
#  survivor pick, so BOTH Assystem lines were dropped and a ~EUR700M French engineering
#  firm became UNSELECTABLE by the screen. That is the expensive error this whole
#  section is built to avoid: a false merge DELETES A REAL COMPANY, while a missed merge
#  only costs a duplicate slot.
#
#  WHY IT HAPPENS, AND WHY IT WILL HAPPEN MORE. The naive "two 64-bit floats will never
#  collide" intuition is wrong here because FMP's marketCap IS HEAVILY QUANTIZED: on the
#  2026-08-13 panel 204 of 2,618 capped lines end in 5 or more zeros (640,500,000 is one
#  of them), so the effective key space around any given size band is small and shrinks
#  further the rounder the number. Collision count grows with the PAIR count, so a
#  ~9,000-name production universe is order 8x this panel's -- roughly 8 false merges,
#  i.e. 8 deleted companies -- which is why this is fixed before the next fetch rather
#  than lived with at 1.
#
#  THE RULE: marketCap must be corroborated by ONE non-quote field. K2 fires only when
#  the caps are exactly equal AND at least one of these ALSO matches exactly:
#
#      the normalised issuer name  |  the latest statement date  |  revenue
#                       |  netIncome  |  totalAssets
#
#  Expressed as FIVE hash buckets rather than a pairwise test, so the architectural
#  invariant at the top of `_issuer_components` is untouched: every key is still a pure
#  function of ONE LINE'S OWN FIELDS, and two lines still meet only by landing in the
#  same bucket. Because each new bucket REFINES the old marketCap bucket, the change is
#  monotone -- K2 can now only SPLIT what it used to merge, never merge more.
#
#  THE DISJUNCTION IS MEASURED, NOT ASSUMED, AND A SINGLE CORROBORATOR WAS REFUTED.
#  Every K2-ONLY pair (a pair K1 and K3 do not also hold) on all three real panels was
#  enumerated and asked which non-quote fields its two lines agree on:
#      PANEL-JAN 2026-01-09  129 K2-only pairs -- ALL 129 agree on at least one
#      CUR3K     2026-08-13   62 K2-only pairs --  60 agree; the 2 that agree on
#                                                  NOTHING are the Assystem/Elior pair
#      CUR3K     2026-08-11   64 K2-only pairs --  62 agree; same 2 exceptions
#  So the disjunction loses NOTHING on 13,359 lines of real panel and separates exactly
#  the defect. Each single-field version does NOT, and both were tried:
#    * date alone breaks Analog Devices (ADI / 0HFN.L / ANL.DE), GSK / GSK.L and
#      QIPT / QIPT.TO -- the FX-shifted, fiscally-misaligned cross-listings are the very
#      pairs K1 cannot see and K2 exists to catch, so keying on the date is
#      anti-correlated with K2's purpose;
#    * name alone breaks 15 pairs on PANEL-JAN and 6 on CUR3K -- Comcast Holdings vs
#      Comcast Corporation, EIDP vs Corteva, Aegon Ltd. vs Aegon N.V., MDA Space Ltd vs
#      MDA Ltd., the Prudential and DTE preferred lines -- because a vendor happily
#      serves one issuer's lines under several different name strings.
#  `date` in particular EARNS its place and is not filler: Carnival Corporation & plc
#  (CCL / CCL.L / 0EV1.L / POH1.DE), Cintas (CTAS / CIT.DE / 0HYJ.L) and Restaurant
#  Brands (QSR.TO / QSP-UN.TO / 0VFA.L) agree on the date and on NOTHING ELSE.
#
#  WHAT THIS DOES NOT DO, STATED PLAINLY. It is a corroboration requirement, not a
#  collision-proof identity. Two unrelated lines never share a name or an exact
#  revenue/netIncome/totalAssets, so the operative disjunct for a CHANCE collision is
#  the date -- and two random lines on the 2026-08-13 panel share a latest statement
#  date with probability 0.39 (0.65 on PANEL-JAN, whose dates are more concentrated).
#  So this removes on the order of 60% of future chance collisions, NOT all of them.
#  Eliminating them outright needs an identity key (K4, below, or a common-fiscal-period
#  K1), not a stronger corroborator.
_K2_CORROBORATORS = ('name', 'date') + _K1_COLS

#  ...AND THE `date` DISJUNCT NEEDS A STATEMENT-CONSISTENCY CHECK ON TOP (2026-08-14).
#
#  THE LIVE DEFECT.  The paragraph above closes by saying the operative disjunct for a chance
#  collision is `date`, and that this removes "on the order of 60% of future chance collisions,
#  NOT all of them".  One of the survivors was already shipping: on the 2026-08-13 run
#  `ALTA.PA` (Altarea SCA, ISIN FR0000033219) and `AREIT.PA` (Altareit SCA, FR0000039216) were
#  merged and AREIT.PA was DELETED (`decided_by=shares`).  They are a PARENT and its listed
#  SUBSIDIARY -- revenue EUR875.9M vs 738.9M, totalAssets EUR7.69bn vs 2.99bn, shares 23.4M vs
#  1.75M -- and they agree on exactly two things: a marketCap of 1,022,830,380.0 to the euro,
#  and a latest statement date.  That is the Assystem/Elior failure again, one disjunct later.
#
#  TWO FIXES WERE TRIED FIRST AND BOTH ARE REFUTED BY MEASUREMENT.  Recorded because each is
#  the obvious idea, and the next reader will have the same one:
#
#   1. "THE DATE DISJUNCT SHOULD NOT APPLY WHEN BOTH LINES ARE ON THE SAME EXCHANGE" -- the
#      reasoning being that the date disjunct exists for FX-shifted cross-listings, which are
#      on different exchanges by definition.  IT CHANGES NOTHING: 0 groups move on all three
#      of the 2026-08-07 / 08-11 / 08-13 panels.  Altarea survives it by TRANSITIVITY through
#      the LSE line -- `0IRK.L` (.L) and `AREIT.PA` (.PA) are on DIFFERENT exchanges, so that
#      edge is untouched, and `ALTA.PA`/`0IRK.L` are held by K1, K3 AND K4.  Restricting a
#      union-find edge does nothing when a two-hop path around it survives.
#   2. "TWO KNOWN, DIFFERENT ISINs MAY NOT MERGE ON THE DATE ALONE".  It fixes Altarea but
#      BREAKS TWO DOCUMENTED WANTED MERGES to do it.  Enumerated on 2026-08-13, the pairs held
#      ONLY by the date disjunct are exactly FIVE, and every one has two known, different
#      ISINs: the two Altarea edges (FALSE), `QSR.TO`/`QSP-UN.TO` + `0VFA.L`/`QSP-UN.TO`
#      (Restaurant Brands Inc. vs its exchangeable LP units -- named in the block above as a
#      case the date disjunct EARNS its place on), and `AEG`/`AGN.AS` (Aegon Ltd. vs Aegon
#      N.V.).  Two real losses to fix one false merge is the wrong trade, and a different ISIN
#      is plainly not proof of a different issuer -- Robertet (`RBT.PA`/`CBE.PA`) and Value8
#      (`VALUE.AS`/`PREVA.AS`) both carry different ISINs and both merge correctly, on K1.
#
#  WHAT ACTUALLY SEPARATES THEM, and it is the property the register already names: A SHARE
#  CLASS OR A CROSS-LISTING SHARES ITS ISSUER'S STATEMENTS; A SUBSIDIARY DOES NOT.  K1 tests
#  that at EXACT equality, which is why it cannot see a pair reported in two currencies.  The
#  general form is that two lines of ONE issuer differ by a SINGLE multiplicative constant --
#  the FX rate, or 1.0 in a common currency -- so revenue, netIncome and totalAssets all scale
#  by the SAME factor.  A parent and a subsidiary do not: consolidation moves the three lines
#  by DIFFERENT factors.  So the discriminator is the SPREAD of the three ratios, which is
#  CURRENCY-FREE by construction -- and being currency-free is exactly what killed proposals
#  1 and 2, both of which used a proxy for currency instead of the thing itself.
#
#  THE STATISTIC IS `_statement_spread` -- READ ITS DOCSTRING FIRST.  It is NOT a plain
#  max/min of the three ratios; it DISCARDS THE SINGLE MOST DEVIANT one before taking max/min,
#  because a plain max/min was measured to refuse 2.90% / 3.64% / 3.22% of pairs whose
#  same-issuer identity is not in question (Unilever, OpenText, Broadcom...).  Everything below
#  is measured with the trimmed statistic.
#
#  WHERE 1.25 SITS, STATED WITHOUT THE HEADROOM CLAIM AN EARLIER VERSION MADE.  That version
#  said the threshold had "36x" margin over the worst true pair.  THAT WAS AN ARTIFACT OF
#  MEASURING ONLY THE FIVE PAIRS THEN HELD BY `date` ALONE.  Measured over every K2-`date`
#  bucket pair on all three panels, with the trimmed statistic, the two populations DO NOT
#  cleanly separate -- they leave a band about four percent wide:
#
#      largest KNOWN-TRUE spread   OGI / OGI.TO           1.3928   (PANEL-JAN)
#                                  0VGE.L / SSRM          1.3244   (both CUR3K panels)
#                                  SSRM / SSRM.TO         1.3135   (both CUR3K panels)
#      the FALSE pair              ALTA.PA / AREIT.PA     1.4478   (all three panels)
#
#  So there is no comfortable gap to sit in the middle of.  1.25 is placed BELOW the true tail
#  DELIBERATELY, on the refusing side: it accepts a measured 0.04% / 0.45% / 0.46% false
#  refusal (SSR Mining on the CUR3K panels, Open Farm on PANEL-JAN) in exchange for 16% of
#  margin against the false pair.  A threshold of 1.40 would refuse nothing true on any of the
#  three panels AND still split Altarea -- but it would sit 3.4% under the false pair and 0.5%
#  over the nearest true one, so the next vendor restatement moves it across.  The asymmetry
#  in the block above decides it: a refused true pair costs a duplicate slot, a false merge
#  DELETES A REAL COMPANY, so the threshold is placed where the recurring cost is a slot.
#
#  THE STATISTIC IS STILL NOISY AT THE BOTTOM and that is why it is not tighter.  netIncome is
#  the unstable field: over the 1,013 KNOWN-TRUE pairs on these three panels (same exact ISIN,
#  so identity is not in question) the UNTRIMMED spread has a median of 1.0000 but a maximum of
#  1,324 -- SSR Mining's two lines carry near-zero and slightly different netIncome, so their
#  ratio explodes while the companies are plainly identical.  The trim is what removes most of
#  that, and SSR Mining is precisely the pair it does not fully rescue.
#
#  THE FAILURE DIRECTION IS THE SAFE ONE, AND THAT IS WHY THIS IS ACCEPTABLE AT ALL.  This rule
#  can only ever REMOVE a date-disjunct union: it is monotone in the same way the corroborator
#  itself is, so it can SPLIT what is merged today and can NEVER merge more.  A refused true
#  pair costs a duplicate slot on the shortlist; a false merge DELETES A REAL COMPANY.  The
#  block above sets that asymmetry out, and this threshold is chosen on the same side of it.
#
#  NOT A COLLISION-PROOF IDENTITY EITHER, stated as plainly as the paragraph above states it:
#  a chance marketCap+date collision between two lines that HAPPEN to have proportional
#  statements would still merge.  That still wants K4/ISIN coverage or a common-fiscal-period
#  K1; this closes the case that is deleting a company today.
_K2_DATE_MAX_STATEMENT_SPREAD = 1.25


def _statement_spread(a, b, _val):
    """How far the two lines' (revenue, netIncome, totalAssets) ratios are from ONE constant,
    AFTER DISCARDING THE SINGLE MOST DEVIANT of the three.  `None` when it cannot be told.

    WHY THE MOST DEVIANT RATIO IS DISCARDED (review D-3, 2026-08-14).  The first version of
    this took `max/min` over all three ratios, and that is refused by measurement on our own
    panels.  Enumerating every K2-`date` bucket pair that is ALSO corroborated by another key
    -- so same-issuer identity is not in question -- and asking how many the plain `max/min`
    would refuse at 1.25:

        PANEL-JAN 2026-01-09   80 of 2,757 = 2.90%
        CUR3K     2026-08-11   16 of   440 = 3.64%
        CUR3K     2026-08-13   14 of   435 = 3.22%

    and the refused set is not exotic -- `UL`/`ULVR.L` (Unilever, the canonical cross-listing)
    at 2.00, `OTEX`/`OTEX.TO` at 4.13, `BATRA`/`BATRK` at 3.48, `1YD.DE`/`AVGO` at 1.64.
    THE TWO DISTRIBUTIONS OVERLAP OUTRIGHT: Altarea/Altareit, the FALSE pair this guard exists
    for, sits at 2.17 -- BELOW OpenText's 4.13 and ABOVE Unilever's 2.00.  So "two lines of one
    issuer differ by a single multiplicative constant" is NOT true in general, and a statistic
    resting on it is not a discriminator; the earlier version only ordered the five then-current
    date-only pairs correctly, which is a much weaker claim than it was written as.

    WHAT ACTUALLY GOES WRONG IS ALWAYS *ONE* LINE, AND RESTATEMENT LAG IS THE COUNTEREXAMPLE
    CLASS.  A vendor restatement, a minority interest, or a period-boundary difference moves
    ONE of the three fields while the other two still agree, and `max/min` over all three lets
    that single line carry the verdict.  The case that proves it is this guard's OWN wanted
    exemplar: `QSR.TO`/`QSP-UN.TO` (Restaurant Brands) reads rev 0.9996 / ni 0.9996 / ta 0.9996
    on the 2026-08-13 panel but rev 0.9996 / ni 0.7621 / ta 0.9996 on the 2026-08-11 one,
    because FMP served `QSP-UN.TO` netIncome 665,000,000 for period 2026-04-01 on the 08-11
    vintage and RESTATED it to 507,000,000 by 08-13 while `QSR.TO` carried 506,783,948
    throughout.  Under the old statistic that pair scored 1.3116 and was REFUSED on one of the
    three panels cited as the threshold's evidence.  (Its blast radius was zero -- the family
    survives via K1 and K4 through `QSR` -- but by luck of a second key, not by this guard.)

    SO: DROP THE ONE MOST DEVIANT RATIO, then take `max/min` of what is left.  Measured
    false-refusal on the same known-same-issuer populations:

        PANEL-JAN  2.90% -> 0.04%      08-11  3.64% -> 0.45%      08-13  3.22% -> 0.46%

    a 65-70x reduction, and Altarea is STILL refused (1.4478 against the 1.25 threshold).  It
    also fixes the RBI-08-11 case (-> 1.0000) and both Carnival DLC pairs (`CCL`/`CCL.L` inf ->
    1.0070, `CCL`/`POH1.DE` 1.9507 -> 1.0070), so the PANEL-JAN Carnival split now disappears
    on the statistic's own merits instead of needing the name-map argument.

    THE COST, STATED BECAUSE IT IS A REAL TRADE.  The Altarea margin narrows from 74% above the
    threshold (2.1705 vs 1.25) to 16% (1.4478 vs 1.25), and an ASSYSTEM/ELIOR-SHAPED pair would
    score 1.2378 -- INSIDE the threshold -- where the old statistic put it at 1.6106 outside.
    That is hypothetical rather than a live regression: Assystem/Elior agree on nothing but
    marketCap, are not in the `date` bucket at all, and are already separated by the K2
    corroborator on both CUR3K panels.  The trade is taken deliberately: a 3% standing
    false-refusal rate on real cross-listings is a certain, recurring cost, while the narrowed
    margin is a hypothetical one.

    NEGATIVE RESULT, RECORDED SO THE ROUTE STAYS CLOSED: using `revenue + totalAssets` only and
    dropping netIncome is WORSE -- 4.57% / 7.73% / 7.82% false refusal -- because two fields
    make the `len(ratios) < 2` branch fire on the zero-revenue LSE-depositary lines (22-53 pairs
    per panel), converting them from "merge" to "cannot tell" -> refuse.  Do not take it.

    RETURN CONTRACT:
      None : fewer than two ratios computable -- "cannot tell".  The caller treats that as NOT
             corroborated, which is the safe direction (see the block above).
      inf  : TWO OR MORE ratios have opposite signs.  No currency conversion or share-class
             relationship flips the sign of a statement line, so one sign disagreement is the
             deviant ratio to discard (that is exactly Carnival's negative `CCL.L` revenue),
             but two of them is positive evidence of different issuers rather than one bad line.
    """
    ratios = []
    for c in _K1_COLS:
        x, y = _val(a, c), _val(b, c)
        if x is None or y is None or x == 0 or y == 0:
            continue
        ratios.append(x / y)
    if len(ratios) < 2:
        return None
    #  A non-positive ratio has no log, so it cannot be ranked for deviance -- it IS the most
    #  deviant, by the sign argument above.  One is discarded; two mean the pair is not one
    #  issuer seen twice.
    bad = [r for r in ratios if r <= 0]
    if len(bad) >= 2:
        return float('inf')
    if bad:
        if len(ratios) < 3:
            #  Only one usable ratio would remain: a sign disagreement with nothing left to
            #  corroborate against is not evidence of agreement.
            return float('inf')
        ratios = [r for r in ratios if r > 0]
    elif len(ratios) >= 3:
        #  Discard the ratio furthest from the MEDIAN in log space -- the median is the
        #  robust centre, so the single restated / minority-interest line is what leaves.
        import math
        logs = sorted((math.log(r), r) for r in ratios)
        med = logs[len(logs) // 2][0]
        worst = max(logs, key=lambda t: abs(t[0] - med))[1]
        ratios = list(ratios)
        ratios.remove(worst)
    return max(ratios) / min(ratios)


def _issuer_components(syms, cdx_df, names, isin_map=None):
    """Union-find grouping of same-issuer lines. FOUR EXACT KEYS, NO TOLERANCE.

    Each key is a HASH BUCKET COMPUTED FROM ONE LINE'S OWN FIELDS -- no pairwise
    comparison, no tolerance, no threshold anywhere in the grouping:

      K1  exact (revenue, netIncome, totalAssets), all three present   [_K1_COLS]
      K2  exact marketCap (present, > _K2_MARKETCAP_FLOOR) CORROBORATED by an exact
          match on at least one of `_K2_CORROBORATORS` -- see the long note above that
          constant for the Assystem/Elior false merge that forced the corroboration and
          for why no SINGLE corroborator survives the panels.
      K3  exact (normalised name incl. `Holding`, weightedAverageShsOut), both present
          -- the old edge B, RETAINED UNCHANGED except for the `Holding` token. It is
          the only key that catches 32 real pairs (Manulife's six lines, Southern's
          five, Chimera's six, PennyMac's five, ACRI-A.ST/ACRI-B.ST), so dropping it
          would be a regression.
      K4  exact ISIN, from the profile map (`isin_map`; None -> `_isin_map_cached()`,
          `{}` -> the key is inert). Wired 2026-08-13, register N-2; see below.

    MEASURED on the 2026-01-09 panel (8,106 lines):
      grouping          components  multi-groups  lines dropped  new pairs  REGRESSIONS
      current A/B/C           6,437         1,236   1,669 (20.6%)         -            -
      K1 only                 6,449         1,234   1,657              +126          158
      K1+K2                   6,340         1,277   1,766              +241           32
      K1+K2+K3 (this)         6,328         1,282   1,778 (21.9%)      +247            0
    i.e. a STRICT SUPERSET of what the old edges found, +247 pairs, zero regressions.

    PRECISION -- the load-bearing number, and it is why the aggression is safe. Every
    candidate false positive on that panel was inspected by hand: of the 16 K2 groups
    whose members' totalAssets disagree by >25%, ALL 16 are the same issuer (FX: Lundin
    Gold, Lundin Mining, IPCO, Traton, Stora Enso; period alignment: RBC, GSK, Comcast;
    subsidiary bond lines: AFGB/C/D/E). Of the 55 K1 groups spanning >1 distinct
    normalised name, ALL 55 are the same issuer (MicroStrategy->Strategy, Barrick
    Gold->Barrick Mining, Bed Bath & Beyond->Beyond, the FMP-truncated "Babcock &
    Wilcox Enterprises, I").

    *** "ZERO CHANCE COLLISIONS IN EITHER" WAS TRUE OF THAT PANEL AND IS FALSE IN
    GENERAL -- REFUTED 2026-08-13, register N-1. *** The sentence used to end here and it
    was load-bearing: it is the reason K2 was allowed to key on marketCap ALONE. One
    chance collision has now occurred, on BOTH the 2026-08-11 and 2026-08-13 CUR3K runs
    -- Assystem S.A. merged into Elior Group S.A. on an exact 640,500,000.0 marketCap and
    was deleted from the universe. The claim is left standing above as the record of what
    was measured on 8,106 January lines (1 in 342 multi-line groups is genuinely below
    what one panel can resolve), NOT reconciled away: the inference drawn FROM it was the
    defect, and K2 now requires corroboration. Read `_K2_CORROBORATORS`.

    The shell-collision worry
    does not materialise: 42 K1 groups have revenue == 0 and all 42 are true
    LSE-depositary/common pairs -- a 4-decimal (netIncome, totalAssets) pair is
    effectively unique. And where names collide but the issuers DIFFER, the fundamentals
    keys correctly REFUSE to merge: FBNC/FBP/FNLC (three different First Bancorps),
    IBCP/INDB, GHC/GHM, DOM.L/DPZ, OBDC/OWL, ATER/ATN.L, SST/SYS1.L, TORO/TTC all stay
    separate. A fundamentals fingerprint separates issuers that name matching merges.

    RESIDUAL RECALL MISS: exactly one, 0SAY.L / DWS.DE (dmarketCap 0.26%, drevenue 4.64%
    -- a fiscal-period misalignment). One miss in 8,106 lines does not buy back a
    tolerance parameter; if it is ever worth fixing, fix it by aligning both lines on a
    COMMON FISCAL PERIOD before fingerprinting, not by loosening a threshold.

    SCOPE-INVARIANCE (this is what the old rule did NOT have). getData_gen's rule C is
    PAIRWISE -- it recognises an instrument line only relative to a shorter sibling -- so
    its completeness depended on which universe was active. Every key here is a bucket
    over a single line's own fields, so grouping is scope-invariant: run on the emitted
    87-row ranking alone it produces the same groups as running on all 8,106 lines and
    projecting down (verified both ways). The pairwise weakness dissolves for the DROP
    decision. It does NOT dissolve for the PICK decision -- a group with only one member
    in the pool has nothing better to choose -- which is why the share-class filter
    stays; see the note above filter_non_common_instruments.

    K4 (exact ISIN) IS NOW WIRED (2026-08-13, register N-2). The note here used to say it
    was not, because "profile ISIN is fetched but not plumbed into cdx_df" -- and that
    blocker was STALE: nothing needs to reach cdx_df. `_isin_map_cached()` is already
    loaded on this path and already consumed by `_isin_plurality_term`, so K4 is a
    union-find pass over a map dedup was ALREADY READING to order survivors. It was
    ordering with an identifier it refused to group with.

    WHAT IT BOUGHT, MEASURED on the real runs (pairs, incl. transitive closure):
      2026-08-13  +11 pairs, 0 lost   |   2026-08-11  +12 pairs, 0 lost
    and every one is a genuine same-issuer pair: the LSE depositary/common families
    (0HPW.L/BR, 0I0X.L/CDXS, 0QCV.L/ABBV, 0R18.L/BBY, 0J4V.L/HRTX, 0KCC.L/IMDX), the
    US/TSX cross-listings (AG/AG.TO, HUT/HUT.TO, KEEL/KEEL.TO) and Samsung.

    SAMSUNG IS THE CASE IT WAS WIRED FOR, and note it is a DIFFERENT defect from the one
    the old note described. The old note worried about the survivor PICK inside a group
    {SMSD.L, SMSN.L}. The actual live defect was in the GROUPING: `SMSN.L` was in a group
    OF ITS OWN, so the 2026-08-13 shortlist billed as 100 issuers was 98 -- Samsung
    Electronics occupied two slots. K2 missed it (SMSN.L's marketCap is
    1,129,161,624,800,000 against the Korean lines' 1,343,738,780,000,000, because its
    statements sit one fiscal quarter behind) and K3 missed it (the depositary line has
    its own share count, 270,134,000 vs 4,023,170,000). K4 catches it: SMSN.L and BC94.L
    both carry ISIN US7960508882.

    SK HYNIX IS NOT FIXED, AND THAT IS A DECISION RATHER THAN AN OVERSIGHT.
    `000660.KS` and `SKHY` are the same issuer and remain two groups, so the shortlist is
    99, not 100. K4 cannot reach it: SKHY's ADR ISIN `US78392B2060` is held by no other
    line on the panel. The ONLY key that reaches it is a NAME-ONLY key, and a name-only
    key was measured and REJECTED -- on the 2026-08-13 panel it buys SK hynix and merges
    THREE pairs of genuinely different companies:
        TORO (Toro Corp., shipping) + TTC (The Toro Company, mowers)
        TEAM.L (TEAM plc) + TISI (Team, Inc.)
        001800.KS (ORION Holdings Corp.) + ORN (Orion Group Holdings, Inc.)
    TORO/TTC is one of the must-not-merge pairs pinned four paragraphs above. Trading one
    recovered duplicate for three deleted companies is the wrong side of this section's
    whole asymmetry, so SK hynix stays split.
    THE KEY THAT WOULD FIX IT PROPERLY is the COMMON-FISCAL-PERIOD K1 already named under
    RESIDUAL RECALL MISS above -- bucket on (date, revenue, netIncome, totalAssets) for
    EVERY row rather than on the latest row only, which stays a pure per-line hash bucket.
    Measured, it catches SK hynix AND Samsung AND the 0SAY.L/DWS.DE miss; it also merged
    CRML (Critical Metals) with SZZL (Sizzle Acquisition Corp. II) on one shared row --
    a shell collision, i.e. it reopens exactly the risk the latest-row-only K1 avoids by
    accident. NOT taken in this pass: it is a wider change than the two defects being
    fixed here and deserves its own measurement and review.

    POINT-IN-TIME CAVEAT (a property to test, not a defect). K2 keys on `marketCap`, which
    is QUOTE-derived, so group membership is date-dependent in a way the statement keys are
    not. The backtest path must keep handing this function a `cdx_df` already truncated to
    date D or earlier (baseline_tools/stage2_pit); `_latest_raw` ffills and takes the last
    row, which is correct only on a pre-sliced frame. Flagged rather than redesigned.

    Returns (comps, latest, _val):
      comps  : dict root_symbol -> [member symbols]  (insertion order = order in syms)
      latest : per-source latest raw fundamentals (from _latest_raw)
      _val   : (symbol, col) -> rounded finite float or None
    Shared by dedup_to_issuers and dedup_ranked so both resolve issuer identity
    IDENTICALLY."""
    # `price` is NOT a grouping input -- it is carried only so the dropped-sibling audit
    # trail can show the CEO what each collapsed line was quoted at (dedup_to_issuers).
    latest = _latest_raw(cdx_df, list(_K1_COLS)
                         + ['weightedAverageShsOut', 'marketCap', 'price'])
    #  The date `latest`'s values came from -- a K2 corroborator, never a key of its own.
    period = _latest_period(cdx_df)
    #  `None` means "use the process-wide profile map", `{}` means "no ISIN data". SAME
    #  CONVENTION AS `_investability_key`, deliberately: the pick and the grouping must
    #  resolve ISIN from the same place or they can disagree about what one issuer is.
    imap = _isin_map_cached() if isin_map is None else isin_map

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

    # ONE pass over the pool, four keys, all in ONE bucket dict -- each key is tagged
    # ('K1'/'K2'/'K3'/'K4') so the namespaces cannot collide with each other, and the
    # shape makes the important property self-evident: a line's keys are functions of
    # THAT LINE ALONE. Two lines meet only by landing in the same bucket.
    buckets = {}
    for s in syms:
        nm = _norm_issuer_name(names.get(s, ''), keep_holding=True) if names else ''

        k1 = tuple(_val(s, c) for c in _K1_COLS)
        if all(v is not None for v in k1):
            buckets.setdefault(('K1',) + k1, []).append(s)

        mc = _val(s, 'marketCap')
        if mc is not None and mc > _K2_MARKETCAP_FLOOR:
            #  ONE BUCKET PER CORROBORATOR, which is how a DISJUNCTION ("cap plus ANY
            #  one of these") stays a set of per-line hash buckets instead of becoming a
            #  pairwise test. Two lines meet under K2 iff their caps match AND at least
            #  one corroborator matches -- exactly one shared bucket is enough.
            #  A corroborator the line LACKS emits no bucket, so a missing field can
            #  never merge two lines by both being absent.
            _corr = {'name': nm or None, 'date': period.get(s)}
            for c in _K1_COLS:
                _corr[c] = _val(s, c)
            for tag in _K2_CORROBORATORS:
                v = _corr.get(tag)
                #  `pd.isna` covers the one case `is None` does not: a NaT period on a
                #  line whose every statement row has an unparseable date.
                if v is not None and not pd.isna(v):
                    buckets.setdefault(('K2', mc, tag, v), []).append(s)

        sh = _val(s, 'weightedAverageShsOut')
        if nm and sh is not None:
            buckets.setdefault(('K3', nm, sh), []).append(s)

        #  K4 -- EXACT ISIN. `_clean_isin` returns '' for anything unusable, and '' is
        #  falsy, so a line without a usable ISIN emits no bucket and cannot group with
        #  another line that also lacks one.
        isin = _clean_isin(imap.get(s)) if imap else ''
        if isin:
            buckets.setdefault(('K4', isin), []).append(s)

    #  THE `date` BUCKET IS THE ONE KEY RESOLVED PAIRWISE, and the deviation is deliberate.
    #  Every other key stays a pure per-line hash: a line's key is a function of that line
    #  alone and two lines meet by landing in one bucket.  The date check CANNOT be expressed
    #  that way -- "these two lines' statements are proportional" is a relation between two
    #  lines, not a value either of them carries, and trying to hash it is what made the
    #  same-exchange proposal a no-op (see `_K2_DATE_MAX_STATEMENT_SPREAD`).
    #  IT IS CHEAP BECAUSE THE BUCKET IS ALREADY NARROW: membership requires an EXACT
    #  marketCap AND an EXACT latest statement date, so these buckets hold 2-3 lines on the
    #  real panels (338 of them with >1 member on 2026-08-13).  The quadratic is over that,
    #  not over the pool.
    for key, grp in buckets.items():
        if key[0] == 'K2' and key[2] == 'date' and len(grp) > 1:
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    sp = _statement_spread(grp[i], grp[j], _val)
                    #  None = cannot tell -> NOT corroborated.  A missing statement may never
                    #  be read as agreement; that asymmetry is the same one `_corr` uses when
                    #  it emits no bucket for a field the line lacks.
                    if sp is not None and sp <= _K2_DATE_MAX_STATEMENT_SPREAD:
                        union(grp[i], grp[j])
            continue
        for s in grp[1:]:
            union(grp[0], s)

    comps = {}
    for s in syms:
        comps.setdefault(find(s), []).append(s)
    return comps, latest, _val


# =========================================================================== #
#  CANONICITY MARKERS -- THE DETECTOR PILE, RELOCATED                           #
#                                                                               #
#  These are the same heuristics that used to REMOVE lines from the universe.     #
#  Here they only ORDER the members of an issuer group, so a false positive costs   #
#  a slightly worse surviving ticker instead of a deleted company. Read the         #
#  inversion note at the top of this section before adding or loosening one.       #
# =========================================================================== #

#  (a) LSE INTERNATIONAL ORDER BOOK. A digit-prefixed .L symbol (0HQ7.L, 0R4M.L,
#  0IJO.L) is an institutional grey-market depositary line the CEO cannot buy at the
#  quoted size. This is the single biggest source of untradeable picks: 19 of the 87
#  slots in the 2026-01-09 emitted ranking were 0*.L lines.
_IOB_LSE_RE = re.compile(r'^\d.*\.L$')

#  (b) PREFERRED-SERIES TICKER SUFFIX -- US -PA, TSX -PFJ, Nordic -PREF. Same pattern
#  as getData_gen._PREFERRED_SUFFIX_RE; duplicated rather than imported because carveOut
#  is deliberately credential-less and importing the ingestion module at module scope
#  would drag `requests` into the carve. Safe because the dual-class convention is
#  -A/-B/-C, never -P.
_PREF_SUFFIX_RE = re.compile(r'-P[A-Z]{0,3}$')

#  (d) SAME-ISSUER SYMBOL EXTENSION -- the candidate is a SHORTER GROUP MEMBER's symbol
#  plus a short tail, with no separator (IMPPP = IMPP + P, HNNAZ = HNNA + Z, CIMN =
#  CIM + N, WHLRD = WHLR + D, BWNB = BW + NB).
#
#  THIS TAIL IS DELIBERATELY MORE PERMISSIVE THAN getData_gen's, AND THAT DIFFERENCE IS
#  THE INVERSION CASHING IN -- it is the one place in this change where the architecture
#  actually buys something a detector could not have. getData_gen._INSTRUMENT_TAIL_RE is
#  `^(P[A-Z]?|[A-Z]?[RUWZ]|[PRUWZ][A-Z])$`, a hand-audited WHITELIST, because there a
#  false positive DELETES a common: share classes live in exactly this shape (GOOGL =
#  GOOG + L, UAA = UA + A, WLYB, LILAK, UONEK, METCB, FOXA, NWSA). Here a false positive
#  only picks the sibling, so the whitelist is not needed and its conservatism has a
#  measurable COST: the un-whitelisted single-letter tails (register entry
#  `unwhitelisted-single-letter-tail`) are invisible to it.
#
#  MEASURED on the 2026-01-09 panel -- permissive vs the whitelist, 1,282 groups, 4 picks
#  change and NOT ONE of them is a regression:
#    CIM/CIM-PA..PD/CIMN     CIMN (a NOTES line) -> CIM        <- the real fix. CIMN is
#                            un-whitelisted ("N"), so it sat in the CANONICAL tier and
#                            won its group on marketCap. A notes line as the CEO's
#                            ticker is the exact defect this whole change exists to kill.
#    WLY/WLYB                WLYB -> WLY        dual-class commons, either is fine
#    NWS/NWSA/0K7U.L         NWSA -> NWS        dual-class commons, either is fine
#    CCP.L/CCPA.L/CCPC.L     CCPA.L -> CCP.L    picks the base line
#  No group loses its last canonical member (6 all-non-canonical groups either way).
#  GOOGL / UAA / METCB are demoted -- and that is harmless BY CONSTRUCTION, because the
#  dual-class ruling merges them with GOOG / UA / METC, so the sibling common survives.
#  DO NOT copy this pattern back into getData_gen: there it would delete GOOGL.
_ORDERING_TAIL_RE = re.compile(r'^[A-Z0-9]{1,2}$')

#  (e) KOREAN PREFERRED. Korean symbols are a 5-character issuer root plus a
#  1-character line code: `...0` is the common, anything else is a preferred class.
#  This is the marker with no analogue in any existing rule -- the Korean convention is
#  a suffix ON THE ROOT, which is why the share-class filter caught 1 of 196 -- and it
#  is what makes Korea admissible at all.
#
#  IT IS A CHARACTER, NOT A DIGIT, AND THAT MATTERS. The spec says "6th digit"; measured
#  on the live 2026-08-04 list, the 6th character of the 196 symbols in the 91
#  multi-line families is 0 x91, 5 x78, 7 x9, 9 x1 -- and ALSO **K x15 and L x2**
#  (Korea's "new-type" preferred: Samsung C&T 02826K.KS, Hanjin Kal 18064K.KS, SK Inc
#  03473K.KS, Solus 33637K/33637L.KS). Every one of those 17 carries its common's name
#  VERBATIM, so `_non_common_name_tag` returns '' for all of them. Written as `\d` this
#  marker would have called 17 Korean preferreds canonical.
_KOREAN_LINE_RE = re.compile(r'^([0-9A-Z]{5})([0-9A-Z])\.(KS|KQ)$')

_NON_CANONICAL_TAGS = ('lse-iob', 'preferred-suffix', 'name-vocabulary',
                       'symbol-extension', 'korea-preferred')


def _sym_base(s):
    """Ticker without its exchange suffix ('ACRI-A.ST' -> 'ACRI-A')."""
    return s.rsplit('.', 1)[0] if '.' in s else s


def _sym_suffix(s):
    return s.rsplit('.', 1)[1] if '.' in s else ''


def _name_vocabulary_tag(name):
    """(c) FMP names the instrument as one ("... 6.125% Senior Notes due 2026",
    "... Warrants", "... Pfd Registered Shs Non-Voting").

    Delegates to getData_gen._non_common_name_tag so there is ONE instrument vocabulary
    in the repo -- that regex set was derived by reading every risky name in the
    universe and it must not be re-guessed here. Imported LAZILY (carveOut is
    credential-less; getData_gen pulls in `requests`), and a failed import DEGRADES this
    one marker to always-'' rather than raising: losing a marker costs a worse ticker
    inside a group, never a lost company. That asymmetry is the inversion, made
    operational.
    """
    if not isinstance(name, str) or not name:
        return ''
    try:
        import getData_gen as _gg
        return _gg._non_common_name_tag(name)
    except Exception:
        return ''


#  The liquidity gap a group must show before volume is allowed to decide anything: one
#  ORDER OF MAGNITUDE, as a RATIO between values.  A ratio, not a bucket index -- see the
#  long note in `_volavg_liquidity_term` for why the bucketed first cut was wrong.
_VOLAVG_DECIDING_RATIO = 10.0


def _volavg_comparable_values(group, volavg_map):
    """`{member: volAvg}` if the WHOLE group can be compared on volume honestly, else
    `None` meaning ABSTAIN.  Extracted 2026-08-08 so the two volume terms cannot drift.

    THE ABSTENTION RULE LIVES HERE, IN ONE PLACE, because there are now TWO terms that
    consume it -- `_volavg_liquidity_term` (the confident order-of-magnitude term) and
    `_volavg_raw_liquidity_term` (the weak last-resort term below ISIN).  Duplicating the
    guard would let a later edit relax it for one term and not the other, and the guard is
    the entire correctness argument for both.  Pure extraction: the conditions and their
    order are byte-for-byte what `_volavg_liquidity_term` applied before, so the decade
    term's behaviour is unchanged.

    Two conditions, each preventing a specific way of deciding on something other than
    liquidity (the third condition -- "never read `sym`" -- is a property of this
    function's SIGNATURE: it cannot see the member under test, so the abstain/discriminate
    choice provably cannot differ between two members of one group):

      1. EVERY member must have a usable reading.  Otherwise a mapped member would sort
         above an unmapped one and the survivor would be decided by DATA AVAILABILITY --
         the exact defect the reviewer caught in the first ISIN cut.  Note that this is a
         GROUP-level abstention and it has to be: a sort key is a total preorder, so there
         is no scalar value that means "no opinion about THIS member only" -- wherever an
         unreadable member is placed, that placement IS a decision about it.  Abstaining
         the group is the only realisation of "must not win on absence, must not be
         demoted for absence" that a single ordering term admits.
      2. EVERY member's reading must carry the SAME as-of date.  The map merges
         never-overwrite, so a symbol this run did not fetch keeps a stale reading;
         comparing it against a fresh one compares two market regimes and looks identical
         to a liquidity difference.  An UNDATED map (the pre-2026-08-06 shape) has None
         for every entry, which is self-consistently "all the same unknown date" -- so it
         is allowed, and the staleness caveat rides on the FILENAME as it did before.
    """
    if not volavg_map or len(group) < 2:
        return None
    vals, dates = {}, set()
    for m in group:
        v, asof = volavg_map.get(m, (None, None))
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None                   # (1) a member with no usable reading -> abstain
        if not math.isfinite(f) or f <= 0:
            return None
        vals[m] = f
        dates.add(asof)
    if len(dates) > 1:
        return None                       # (2) mixed as-of dates -> refuse the comparison
    return vals


def _volavg_liquidity_term(sym, group, volavg_map):
    """The ordering signal average volume DOES support: which line of an issuer is the
    LIQUID one, coarsened to ORDERS OF MAGNITUDE.  Lower (more negative) sorts first.

    Returns 0 for every member -- i.e. ABSTAINS -- unless the whole group can be compared
    honestly.  A constant added to every member of a group cannot change a sort, so
    abstention is exactly "no opinion".

    WHY VOLUME AND NOT SOMETHING ELSE (register K-1).  Three groups merge correctly and
    pick the WRONG line: Robertet + its certificat (CBE.PA), Value8 + its preference line
    (PREVA.AS), Samsung + its preferred GDR (SMSD.L).  All three are SAME-EXCHANGE with
    names identical verbatim and an IDENTICAL derived price (price = marketCap /
    weightedAverageShsOut, both issuer-level, so they cancel), so every canonicity marker
    (a)-(e) is ruled out BY CONSTRUCTION and the key falls to the alphabetical tail, where
    the non-common wins ('CBE' < 'RBT', 'PREVA' < 'VALUE', 'SMSD' < 'SMSN').  Volume reaches
    them, and unlike an ISIN it is DIRECTIONAL BY CONSTRUCTION -- the common IS the liquid
    line -- so no convention has to be invented to read it.

    *** WHERE THE "ISIN CANNOT REACH THEM" CLAIM WAS TOO STRONG, corrected 2026-08-06. ***
    It read: ISIN carries no security-type field, so `_isin_plurality_term` abstains on a
    2-member group with 2 distinct ISINs.  Each half is true, but the PREMISE -- that these
    are 2-member groups -- came from a SAMPLED universe that was SPLITTING them.  Robertet
    has THREE lines upstream (RBT.PA, CBE.PA and 0NZN.L on LSE); the sample was dropping the
    LSE line, and with it the second vote that lets plurality speak.  If 0NZN.L carries the
    common's ISIN -- the usual pattern for an LSE IOB line, and NOT MEASURED HERE, no
    isindic pickle exists -- then plurality is {common: 2, certificat: 1}, the term SPEAKS,
    and it demotes the certificat correctly WITHOUT volume.  So on the full pool volume is
    the SECOND of two terms that reach Robertet, not the only one; "volAvg is what fixes the
    three K-1 picks" holds for Value8 and Samsung on the same argument as before, but for
    Robertet it is unproven and probably over-attributed.  The two terms point the SAME way
    here, so nothing is at risk -- but the attribution should not be relied on until the
    first run with a profile build measures both maps.

    POSITION IN THE KEY.  This is term 6, BELOW every canonicity marker and ABOVE ISIN
    plurality, immediately above the alphabetical last resort.  Terms 1-5 are
    byte-unchanged, so any group decided today by canonicity, share count, market cap or
    symbol shape is decided IDENTICALLY.  The only groups volume can move are the ones that
    fall past those -- 381 of 1,282 (29.7%) reach the raw alphabet today -- which is
    precisely the failure surface.

    VOLUME OUTRANKS ISIN, BY CEO RULING (2026-08-06).  The first cut placed it BELOW ISIN
    as the conservative choice and flagged that as not obviously correct; the CEO chose
    volume above ISIN, and the reasoning is the one that was flagged: volume is
    DIRECTIONAL BY CONSTRUCTION -- the common IS the liquid line -- whereas ISIN plurality
    is an IDENTITY INFERENCE that can point the wrong way.  Three depositary lines sharing
    one ISIN against a common carrying its own would hand plurality to a depositary
    receipt.  Where both speak, volume now wins; ISIN still decides every group volume
    abstains on, which is most of them.  BOTH NO-DATA PATHS REMAIN BIT-IDENTICAL under the
    swap -- an absent map makes its term a constant 0 across the group, and a constant
    cannot move a sort whatever its position -- and that is the path every existing pickle
    takes.

    COMPARE VALUES, NOT THEIR FLOORS.  *** The first cut bucketed with
    `-int(floor(log10(v)))` and that was WRONG, caught in review. ***  A decade INDEX has a
    hard edge at every power of ten: volAvg 9,900 -> bucket 3 but 10,100 -> bucket 4, so a
    2% difference spoke with FULL FORCE -- the exact opposite of the stated rationale that
    near-ties tie.  And the direction mattered: a K-1 non-common line sitting just above a
    boundary with its common just below made the term ACTIVELY SELECT THE NON-COMMON, which
    is strictly worse than the abstention it was designed to fall back to, in the very
    groups it was added for.  volAvg is re-read every fetch, so a line drifting across a
    power of ten would also flip the survivor between runs.

    The rule is therefore a RATIO against the group's own maximum: a member is demoted only
    if it is at least `_VOLAVG_DECIDING_RATIO` (10x) LESS liquid than the most liquid line;
    anything inside that band ties.  For a 2-member group that is exactly "decide iff
    max/min >= 10".  For 3+ it generalises the way it should: {1e6, 9e5, 1e4} ties the two
    big lines (they fall through to alphabet, as today) and demotes only the tiny one.  This
    has no absolute edge anywhere -- scaling every value in a group by any factor leaves the
    term unchanged -- and it keeps the property that was wanted: only a genuine
    order-of-magnitude gap decides, and a 2% wobble never does.  A threshold still has a
    boundary AT 10x, but that boundary flips between "volume decides" and "alphabet
    decides", never between "picks A" and "picks B".

    The three K-1 groups are expected to clear 10x comfortably (a certificat / preference
    line / preferred GDR against its common), but that is an EXPECTATION, NOT A
    MEASUREMENT: no volavgdic pickle exists yet and none of the six symbols is in any saved
    panel, so this has never been evaluated on real values.  If a group turns out to sit
    inside one order of magnitude, this term abstains there and the next term decides --
    no worse than today, but not fixed either.

    ABSTAIN UNLESS THE GROUP IS COMPARABLE.  The conditions moved to
    `_volavg_comparable_values` on 2026-08-08 (unchanged) so this term and the weak raw
    term below ISIN share ONE guard: every member needs a usable reading, every reading
    needs the same as-of date, and the decision to speak never reads `sym`.
    """
    vals = _volavg_comparable_values(group, volavg_map)
    if vals is None:
        return 0
    top = max(vals.values())
    #  A RATIO against the group's own maximum, so there is no absolute edge and a near-tie
    #  ties. 0 = within an order of magnitude of the most liquid line (sorts first),
    #  1 = at least 10x less liquid (demoted).
    if all(top / v < _VOLAVG_DECIDING_RATIO for v in vals.values()):
        return 0                          # no member is a decade behind -> nothing to say
    mine = vals.get(sym)
    if mine is None:
        return 0                          # `sym` outside `group` -- no opinion about it
    return 0 if top / mine < _VOLAVG_DECIDING_RATIO else 1


def _volavg_raw_liquidity_term(sym, group, volavg_map):
    """RAW average volume, descending (more liquid sorts first), as the LAST tiebreak
    before the alphabet.  Returns `-volAvg`; 0 for every member when the group abstains.
    ADDED 2026-08-08, CEO ruling.

    WHY A SECOND, WEAKER VOLUME TERM.  `_volavg_liquidity_term` above only speaks on an
    ORDER-OF-MAGNITUDE gap and ties everything inside it; the groups it ties then fall
    past ISIN plurality to the RAW ALPHABET.  On the 2026-08-08 run that was still 7
    groups -- and alphabetical order correlates with NOTHING.  The trade this term makes
    is deliberate and is the CEO's:

        a SLIGHTLY UNSTABLE BUT INFORMATIVE tiebreak beats a PERFECTLY STABLE ARBITRARY
        one.

    The decade rule was chosen for run-to-run stability -- volAvg is re-read every fetch,
    so a 3% wobble can flip a survivor when raw values decide.  That cost is real and is
    ACCEPTED here, because the thing being traded away is not accuracy, it is alphabetical
    order: the fallback this displaces is `'CBE' < 'RBT'`, which carries no information
    about which line is the common at all.  And in the K-1 shape it is plausibly WORSE than
    a coin, because a derived line's ticker is often a mangled variant that sorts before
    its common ('CBE'/'RBT', 'PREVA'/'VALUE', 'SMSD'/'SMSN' -- three for three).

    *** THE OBJECTION, RECORDED RATHER THAN DISMISSED. ***  A near-tie in volume genuinely
    IS weak evidence about which line is the common, and a wrong-but-confident tiebreak can
    be worse than an admittedly-arbitrary one because it LOOKS principled in the artifact.
    That objection is about READING the result, not about the ordering -- the ordering is
    better-than-chance either way -- so it is answered by REPORTING, and it is, three times:
      * this term has its OWN name in `_KEY_TERM_NAMES` (`volavg_raw`), so `decided_by`
        never lets a 1.4x margin masquerade as the confident decade term;
      * the dedup report carries the RAW volumes and their as-of dates for the dropped and
        the surviving line, so a reader sees the MARGIN and can judge its weight; and
      * that report is WRITTEN TO DISK as `DedupSurvivorReport_<date>.csv` (repo root since 2026-08-10) and ships
        in the transfer, and `partition_universe` prints `n_decided_volavg_raw` beside
        `n_decided_alphabetical`.
    *** THE THIRD BULLET IS LOAD-BEARING AND WAS MISSING UNTIL 2026-08-08 (reviewer F1). ***
    The first two were true of a frame that lived only in memory: no caller read it, no CSV
    carried it, no transfer pattern matched it. An answer to "the artifact shows the margin"
    is worthless while there is no artifact -- so if a future edit stops writing that CSV,
    this term's justification lapses with it. A group decided on a 1.03x volume difference
    must be visible AS a 1.03x difference, or the false confidence is real.

    POSITION: term 8 -- BELOW the decade term, BELOW ISIN plurality, and IMMEDIATELY ABOVE
    the alphabetical last resort.  So it reaches EXACTLY the groups that today fall to raw
    alphabet and nothing else: every canonicity marker, share count, market cap, the
    symbol-shape tail, the decade term and ISIN all still outrank it byte-unchanged.

    SAME ABSTENTION GUARD AS THE DECADE TERM, deliberately, and it matters MORE here.  The
    guard is `_volavg_comparable_values`, shared:
      * a member with NO reading abstains the WHOLE GROUP, so an absent/zero/null reading
        can neither win nor be demoted -- absence is never read as a volume of zero;
      * DISAGREEING as-of dates abstain the group.  The decade term refuses that
        comparison because a stale reading against a fresh one compares two market
        regimes; a RAW comparison is strictly more exposed to it, since it acts on
        differences far smaller than the drift a stale reading can accumulate.
    When the guard abstains, every member gets a literal 0 -- a constant, which cannot
    move a sort -- and the group falls to the alphabet exactly as it does today.

    NO-MAP IS BIT-IDENTICAL.  `_volavg_comparable_values` returns None on an empty map, so
    with no volavgdic pickle this term is a constant 0 for the whole group.  Every pickle
    written before 2026-08-08 is in that state.

    *** THE ABSTAIN SENTINEL IS IN-BAND, AND WHAT KEEPS THAT SAFE IS THE GROUP-WIDE GUARD.
    READ THIS BEFORE CHANGING EITHER. ***  Every speaking value is `-volAvg`, i.e. strictly
    NEGATIVE, so the abstain value 0 is not an out-of-band marker -- it is the WORST
    (least-liquid) value in the term's own range.  That is harmless today for exactly one
    reason: `_volavg_comparable_values` abstains the WHOLE GROUP at once, so when 0 appears
    it appears for every member, and a constant cannot move a sort.  The decade term above
    shares the sentinel but is immune by construction (its range is {0, 1} and 0 is its
    BEST value, so an abstaining member sorts first, not last).
    THE COUPLING: if anyone ever makes abstention PER-MEMBER here -- e.g. "skip the members
    with no reading and compare the rest", which reads like a harmless robustness tweak --
    then an absent, null or zero reading silently becomes "least liquid" and LOSES its
    group, and the survivor is decided by DATA AVAILABILITY.  That is the precise defect the
    reviewer caught in the first ISIN cut, and the group-wide guard is the only thing
    preventing it.  An out-of-band sentinel is not available cheaply (the term must be
    order-comparable against real `-volAvg` values, and any finite constant is in-band
    somewhere), so the guard IS the mitigation -- do not weaken it, and do not treat a
    per-member abstention as an equivalent refactor.
    """
    vals = _volavg_comparable_values(group, volavg_map)
    if vals is None:
        return 0
    mine = vals.get(sym)
    if mine is None:
        return 0                          # `sym` outside `group` -- no opinion about it
    #  NEGATED so that MORE liquid sorts FIRST, matching -shares / -marketCap above.  The
    #  values are only ever compared WITHIN one group (the sort is per-group), so a raw
    #  magnitude is meaningful here in exactly the way it is for those two terms.
    return -mine


#  The literal `volAvg_asof` markers used when there is no DATE to state, so an empty cell
#  never has to be interpreted.  Exported as names because the tests and any reader that
#  filters on them must not re-spell the strings.
VOLAVG_STATUS_NOT_CAPTURED = 'not-captured'      # symbol absent from the map (or no map)
VOLAVG_STATUS_NO_READING = 'no-reading'          # in the map, but null / 0 / non-finite
VOLAVG_STATUS_UNDATED = 'undated-capture'        # a real value from the pre-dating pickle


def _volavg_reading(sym, vmap):
    """`(value_or_NaN, asof_date_or_STATUS_MARKER)` for ONE symbol -- the single
    implementation of the three-way absence semantics, extracted 2026-08-08.

    Two readers now need identical absence semantics: `volavg_report_frame` (the per-name
    review column) and the DEDUP REPORT (the raw volumes beside each dropped/surviving
    line).  If they were written twice they could drift, and the whole value of the dedup
    columns is that `no-reading` (an abstention trigger) stays distinguishable from a real
    zero and from `not-captured`.  Same three markers, same order of tests, as before.
    """
    vmap = vmap or {}
    if sym not in vmap:
        return float('nan'), VOLAVG_STATUS_NOT_CAPTURED
    v, asof = vmap.get(sym, (None, None))
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = float('nan')
    if not math.isfinite(f) or f <= 0:
        return float('nan'), VOLAVG_STATUS_NO_READING
    return f, (asof if asof else VOLAVG_STATUS_UNDATED)


def volavg_report_frame(symbols, volavg_map=None):
    """REPORT-ONLY average volume for a list of symbols: a 2-column frame
    (`volAvg_report`, `volAvg_asof`) aligned to `symbols`, for the human-review artifacts
    (register J-1, CEO 2026-08-06).

    REPORT, NOT SCREEN.  The CEO's ruling is that average volume is SURFACED beside each
    name and NOTHING is excluded on it: no liquidity floor, no threshold, no re-ordering.
    Same shape as the loss-distribution decision -- show the number, let his own judgement
    use it -- because any floor the house could pick would silently drop names on a bar
    nobody can justify.  So this function is a pure lookup: it returns values and never
    filters, sorts, or scores.  Its other readers in the pipeline are the TWO dedup survivor
    tiebreaks -- `_volavg_liquidity_term` (the order-of-magnitude term) and, from
    2026-08-08, `_volavg_raw_liquidity_term` (the weak raw term below ISIN) -- both of which
    are UNTOUCHED by this; nothing here feeds either of them.  What IS shared, deliberately,
    is `_volavg_reading`: this frame and the dedup report's `*_volAvg` columns resolve
    absence through one function so the three markers cannot drift apart between artifacts.

    ABSENCE IS NOT ZERO, AND THE THREE KINDS OF ABSENCE ARE DISTINGUISHED.  Every existing
    pickle predates the volAvg capture, so "no number" is the NORMAL state and must not read
    as a genuinely illiquid name -- a 0 in a volume column is a finding, an empty one is a
    gap, and they must not look alike.  Same reasoning as the NaN-not-0 rule already applied
    to the dedup diagnostic columns.  `volAvg_report` is therefore NaN whenever no usable
    reading exists, and `volAvg_asof` states WHICH absence it is:
      * `not-captured`   -- the symbol is not in the map at all (or no map exists);
      * `no-reading`     -- present but null / 0 / non-finite, which is FMP's usual answer
                            for a thin line and is an abstention trigger in the dedup term
                            too, NOT a liquidity reading of zero;
      * `undated-capture`-- a real value from the pre-2026-08-06 UNDATED pickle shape.

    THE DATE TRAVELS WITH THE NUMBER, always.  Average volume is time-varying and
    findAllSectors merges MERGE-NEVER-OVERWRITE, so a symbol a run did not fetch keeps a
    stale reading indefinitely.  A dated column lets the reader see that two names are being
    compared across two market regimes; an undated one invites exactly the false comparison
    `_volavg_liquidity_term` refuses to make (its condition 2).  That is why `volAvg_asof` is
    not optional and is emitted even when it can only carry a status marker.
    """
    vmap = _volavg_map_cached() if volavg_map is None else volavg_map
    vals, dates = [], []
    for sym in symbols:
        f, asof = _volavg_reading(sym, vmap)
        vals.append(f)
        dates.append(asof)
    return pd.DataFrame({'volAvg_report': vals, 'volAvg_asof': dates})


# --------------------------------------------------------------------------------- #
#  TRADED VALUE PER DAY (CEO, 2026-08-13) -- the SECOND consumer of the volavgdic     #
#  entry, and the FIRST one that needs the profile fields the loader above drops.     #
# --------------------------------------------------------------------------------- #
#  `volAvg` is a SHARE count, so it is not comparable across listings: 45.7M shares of a
#  $154 line and 45.7M shares of a KRW 1.5M line are four orders of magnitude apart in
#  money.  The CEO's own named example is traded VALUE, which is the comparable quantity:
#
#      dollarVolume_usd = volAvg  x  profile price  x  FX(profile currency -> USD)
#
#  ALL THREE FACTORS COME FROM THE SAME volavgdic ENTRY, which is what makes the product
#  meaningful: findAllSectors folds `price` and `currency` into the entry that carries
#  `volAvg` and they share the ONE `asof`, so this is one instant's reading rather than
#  three snapshots multiplied together.  The AggScore CSV's own `price` column is a
#  SEPARATE, LATER profile call (measured on the 2026-08-13 run: 000660.KS is 1,504,000 in
#  the 00:17 capture and 1,616,000 in the 03:36 CSV), so it is deliberately NOT used here.
#
#  ############# WHICH CURRENCY CONVERTS THIS ONE -- AND WHY IT IS THE OTHER ONE #######
#  ## Read the two-currency block at the top of this file first.  `marketcap_usd_series` ##
#  ## converts with `reportedCurrency` and MUST NOT be changed to the profile currency.  ##
#  ## THIS function converts with the PROFILE (trading) currency, and that is not an     ##
#  ## inconsistency -- it is the same rule applied to a different quantity.  `volAvg` and ##
#  ## the profile `price` are properties of a LISTING LINE and are denominated in the    ##
#  ## line's TRADING currency (SHEL.L quotes in pence and reports USD; its traded value  ##
#  ## is a pence quantity, not a USD-statement quantity).  Converting a trading quantity  ##
#  ## with the STATEMENT currency would be the same unit mismatch, in the mirror.        ##
#  ##                                                                                    ##
#  ## THIS IS THE FIRST LIVE USE OF THE PENCE MINOR-UNIT PATH (the 2026-08-09 note at the ##
#  ## top of this file predicted exactly this): zero sources REPORT in GBp, so `GBp`/`GBX` ##
#  ## have never been looked up in production before.  VERIFIED on the 2026-08-13 run --  ##
#  ## SHEL.L 9,288,640 sh x 3322.5 GBp x 0.0134987 = $416.6M/day, which is the right      ##
#  ## order for Shell's London line; the GBP (not GBp) rate would have given $41.6bn.     ##
#  #####################################################################################
#
#  REPORT, NEVER SCREEN -- the same standing ruling as `volavg_report_frame` (register
#  J-1, CEO 2026-08-06).  Nothing here filters, sorts or scores; it is appended to
#  already-selected, already-ordered frames.  A liquidity FLOOR remains a decision nobody
#  has taken, and this function must not become one by the back door.
#
#  ABSENCE IS NOT ZERO, and the KIND of absence is named, exactly as `volAvg_asof` does
#  it -- a name we could not price and a name that trades $0 must not look alike.
DOLLARVOL_STATUS_NO_PRICE = 'no-price'                 # entry has no usable profile price
DOLLARVOL_STATUS_NO_CURRENCY = 'no-currency'           # entry predates the currency capture
DOLLARVOL_STATUS_FX_UNRESOLVED = 'fx-unresolved:%s'    # currency known, rate refused/absent


def _load_volavg_profile_map(volavg_pickle=None):
    """{sym: {'volAvg', 'asof', 'price', 'currency'}} from the same pickle `_load_volavg_map`
    reads -- a SECOND, WIDER view of the same file.

    DELIBERATELY NOT A WIDENING OF `_load_volavg_map`.  That loader documents itself as the
    single seam that makes "capture changes nothing" a PROPERTY: it drops every profile field
    so the dedup survivor tie-breaks cannot start reading one by accident.  Adding keys to it
    would retire that guarantee for every existing consumer in order to serve one new report
    column.  A separate accessor keeps the guarantee and makes the new dependency explicit.

    The UNDATED (pre-2026-08-06) entry shape carries no profile fields at all, so it yields
    price/currency None -> the reading is refused, which is the correct answer for a pickle
    that predates the capture.
    """
    import glob
    path = volavg_pickle
    if not path:
        #  Globs the REPO ROOT first, then the CWD -- the same order and the same reason as
        #  `_load_volavg_map`.
        cands = sorted(glob.glob(os.path.join(_MODULE_DIR, 'volavgdic_fmp_*.pickle')))
        if not cands:
            cands = sorted(glob.glob('volavgdic_fmp_*.pickle'))
        path = cands[-1] if cands else None
    else:
        path = _resolve_repo_data(path)
    if not path or not os.path.exists(path):
        return {}
    d = pd.read_pickle(path)
    if not isinstance(d, dict):
        return {}
    out = {}
    for sym, v in d.items():
        if isinstance(v, dict):
            out[sym] = {'volAvg': v.get('volAvg'), 'asof': v.get('asof'),
                        'price': v.get('price'), 'currency': v.get('currency')}
        else:
            out[sym] = {'volAvg': v, 'asof': None, 'price': None, 'currency': None}
    return out


def profile_map_for_run(run_dir, run_date):
    """The profile map belonging to ONE named run, or {} if that run has no capture.

    THE FILENAME CONVENTION STAYS INSIDE THIS MODULE, and that is the point rather than a
    tidiness preference.  `test_universes.test_the_volavg_pickle_still_has_exactly_ONE_reading_seam`
    pins the set of modules that so much as NAME this artifact to {findAllSectors (writes),
    carveOut (loads), Sbocker (ships)} -- because the moment a fourth module knows the
    filename, it is one step from doing its own `read_pickle` and reading raw entries, and the
    single-seam argument lapses silently.  The presentation generator needs THIS RUN's capture
    (it is handed an arbitrary `--run-dir`, so the newest-in-repo-root default would pair a
    2026-08-13 shortlist with an 08-11 capture -- observed).  It asks for it by run, here,
    instead of constructing the path itself.

    NO CROSS-RUN FALLBACK BY CONSTRUCTION: an absent capture returns {}, never the newest
    file lying around.
    """
    try:
        path = os.path.join(str(run_dir), 'volavgdic_fmp_%s.pickle' % run_date)
        if not os.path.exists(path):
            return {}
        return _load_volavg_profile_map(path)
    except Exception:
        return {}


_VOLAVG_PROFILE_CACHE = None


def _volavg_profile_map_cached():
    global _VOLAVG_PROFILE_CACHE
    if _VOLAVG_PROFILE_CACHE is None:
        try:
            _VOLAVG_PROFILE_CACHE = _load_volavg_profile_map()
        except Exception:
            #  Same degradation as `_volavg_map_cached`: an unreadable pickle costs the
            #  report column, never the run.
            _VOLAVG_PROFILE_CACHE = {}
    return _VOLAVG_PROFILE_CACHE


def trading_currency(sym, profile_map=None):
    """The TRADING currency of a listing line, or None.

    Exposed separately from the dollar-volume frame because the AggScore CSV needs to LABEL
    its `price` column and that label must never be guessed from the exchange suffix (see
    `SUFFIX_TO_CURRENCY`: the suffix does not determine the currency -- SHEL.L reports USD
    and quotes GBp, and the LSE IOB lines are foreign issuers wearing a `.L`).
    """
    pmap = _volavg_profile_map_cached() if profile_map is None else profile_map
    e = pmap.get(sym)
    c = e.get('currency') if isinstance(e, dict) else None
    return c if isinstance(c, str) and c.strip() else None


def dollar_volume_frame(symbols, profile_map=None, fx=None, clone_map=None, fx_label=None):
    """REPORT-ONLY traded value per day in USD: a 2-column frame (`dollarVolume_usd`,
    `dollarVolume_basis`) aligned to `symbols`.

    `clone_map`  {source: marker} from `vendor_contamination.clone_counterparts`.  Appended to
                 the basis of any name the run's own contamination check paired with another
                 line -- see the KNOWN LIMIT below, which used to be documented in prose only.
    `fx_label`   appended to the basis of every COMPUTED row, for a caller converting with
                 something other than the run's live table.  The deck passes it when it has to
                 fall back to the sanity anchors, because otherwise an anchor-converted value
                 and a live-converted one produce byte-identical basis strings.

    `dollarVolume_basis` is `'<asof>|<CCY>'` when the value computed -- so the reader sees
    BOTH how old the reading is and which currency it came out of -- and one of the
    `DOLLARVOL_STATUS_*` markers otherwise.  `volAvg`'s own three absence markers
    (`not-captured` / `no-reading` / `undated-capture`) are reused verbatim via
    `_volavg_reading`, so the two volume columns cannot disagree about whether a name has a
    volume reading at all.

    KNOWN LIMIT, and it is the vendor's, not ours: on a CLONE LINE (a GDR/ADR of a foreign
    issuer) FMP can report the HOME line's share volume against the DEPOSITARY line's price.
    Measured on 2026-08-13: SKHY (SK Hynix's Nasdaq line) computes $7.06bn/day, essentially
    the same as 000660.KS's $6.48bn -- a Nasdaq OTC line does not trade that, and it puts SKHY
    THIRD in the top-100 by traded value, above TSM.  The number is then wrong by the
    depositary ratio.  It is NOT silently corrected: we have no depositary ratio, and a
    guessed one would be an invented number wearing a computed label.

    BUT IT IS NO LONGER FLAGGED IN PROSE ONLY (reviewer H-5).  A docstring and a tooltip do
    not travel with a CSV cell, and the cell said `2026-08-13|USD` -- unqualified -- for the
    third-largest traded value in the shortlist.  The run ALREADY computes the pairing
    (`VendorContaminationFlags_<date>.csv` pairs 000660.KS<->SKHY and 005930.KS<->SMSN.L), so
    `clone_map` puts it in the basis string.  This invents nothing and adds no fetch: it
    reuses an artifact of the same run, and it covers SMSN.L for free.
    """
    pmap = _volavg_profile_map_cached() if profile_map is None else profile_map
    #  The volume half goes through the SHARED reading helper, so `not-captured` /
    #  `no-reading` / `undated-capture` mean here exactly what they mean in the volAvg
    #  column beside it.
    vmap = {s: (e.get('volAvg'), e.get('asof')) for s, e in pmap.items()
            if isinstance(e, dict)}
    vals, basis = [], []
    for sym in symbols:
        vol, asof = _volavg_reading(sym, vmap)
        if not (isinstance(vol, float) and math.isfinite(vol)):
            vals.append(float('nan'))
            basis.append(asof)                     # one of the three volAvg absence markers
            continue
        e = pmap.get(sym) or {}
        try:
            px = float(e.get('price'))
        except (TypeError, ValueError):
            px = float('nan')
        if not math.isfinite(px) or px <= 0:
            vals.append(float('nan'))
            basis.append(DOLLARVOL_STATUS_NO_PRICE)
            continue
        cur = e.get('currency')
        if not isinstance(cur, str) or not cur.strip():
            vals.append(float('nan'))
            basis.append(DOLLARVOL_STATUS_NO_CURRENCY)
            continue
        rate = _fx_to_usd(cur.strip(), fx=fx)
        if rate is None:
            #  A refused/stale live rate lands here, and that is deliberate: it must be
            #  indistinguishable from a currency we never knew (the `_fx_to_usd` contract).
            vals.append(float('nan'))
            basis.append(DOLLARVOL_STATUS_FX_UNRESOLVED % cur.strip())
            continue
        vals.append(vol * px * float(rate))
        #  The QUALIFIERS ride with the number, in the field a CSV reader actually sees.
        #  Order is fixed (asof|currency, then FX source, then the clone marker) so the
        #  string stays greppable.
        _b = '%s|%s' % (asof, cur.strip())
        if fx_label:
            _b += '|%s' % fx_label
        _cm = (clone_map or {}).get(sym)
        if _cm:
            _b += '|%s' % _cm
        basis.append(_b)
    return pd.DataFrame({'dollarVolume_usd': vals, 'dollarVolume_basis': basis})


def _non_canonical_tag(sym, name='', group=()):
    """The first non-common marker `sym` shows, or '' if it looks like the common.

    Markers are tested in the order the spec fixes (a)-(e); the tag is returned rather
    than a bool so the dedup audit trail can say WHY a line was demoted.
    """
    if not isinstance(sym, str) or not sym:
        return ''
    if _IOB_LSE_RE.match(sym):
        return 'lse-iob'
    if _PREF_SUFFIX_RE.search(_sym_base(sym)):
        return 'preferred-suffix'
    nt = _name_vocabulary_tag(name)
    if nt:
        return 'name-vocabulary:' + nt
    # (d) needs the GROUP -- it is the one marker that is relative, and it is relative to
    # the issuer group rather than to the whole pool, which is what makes it
    # scope-invariant in a way getData_gen's rule C is not. Same exchange suffix is
    # required, as in rule C, so a genuine foreign listing is never read as a tail.
    base, suf = _sym_base(sym), _sym_suffix(sym)
    for other in group:
        if other == sym or _sym_suffix(other) != suf:
            continue
        ob = _sym_base(other)
        if len(ob) < len(base) and base.startswith(ob) \
                and _ORDERING_TAIL_RE.match(base[len(ob):]):
            return 'symbol-extension'
    m = _KOREAN_LINE_RE.match(sym)
    if m and m.group(2) != '0':
        return 'korea-preferred'
    return ''


#  REMOVED 2026-08-05: `_TIE_DECIMALS`, the rounding at which two AggScores counted as
#  TIED for the old survivor tie-break. Canonicity now overrides rank outright, so there
#  is no tie to detect: a "tie" is simply the case where every ordering term below
#  canonicity was already equal, which the sort handles without a tolerance. Deleted
#  rather than left as a dead constant, because a rounding knob sitting beside a survivor
#  rule invites a reader to think the rule still consults the score.


def _investability_key(sym, val_fn, sector_map=None, names=None, group=(), isin_map=None,
                       volavg_map=None):
    """Deterministic "most investable line" ordering key for an issuer's listings.

    CANONICITY CLASS FIRST, then the CEO's share count. Lowest sorts first:
      1. `_non_canonical_tag(...)` -> 0 if the line shows no non-common marker, 1 if it
         does. THE dominant term.
      2. -weightedAverageShsOut  -- the CEO's size discriminator, INSIDE a canonicity
         tier, which is the only place it works.
      3. -marketCap.
      4. digit-prefix, punctuation count, length -- the previous tail, unchanged.
      5. volAvg liquidity, as a ratio against the group's most liquid line
         (`_volavg_liquidity_term`) -- ADDED 2026-08-06. ABOVE ISIN by CEO ruling
         (2026-08-06): volume is directional by construction, ISIN plurality is an
         identity inference that can point the wrong way. See the note on that function.
      6. ISIN plurality within the group (`_isin_plurality_term`) -- ADDED 2026-08-05,
         see below.
      7. RAW volAvg, descending (`_volavg_raw_liquidity_term`) -- ADDED 2026-08-08 by CEO
         ruling, as a weak tiebreak BELOW everything above and ABOVE the alphabet only.
         Term 5 ties everything inside one order of magnitude; those groups used to land on
         the raw alphabet, which correlates with nothing. Same abstention guard as term 5.
      8. alphabetical -- the last resort, unchanged.

    Terms 5, 6 and 7 all return a CONSTANT 0 for the whole group when their map is
    absent, and a constant cannot move a sort -- so every pre-2026-08-05 artifact and every
    existing pickle resolves through this key bit-identically, and the 5/6/7 ORDER is
    unobservable on any of them.

    WHY NOT SHARE COUNT FIRST, WHICH IS WHAT THE BRIEF ASSUMED. Because it was measured
    and it is a bad rule. FMP serves the ISSUER's own filed share count to every one of
    its lines, so weightedAverageShsOut is IDENTICAL across all members in 1,188 of
    1,282 multi-line groups (92.7%) -- "largest share count" is a TIE in 93% of cases
    and degenerates into whatever follows it. Ranking groups by largest share count
    picks a structurally non-canonical line in 700 of 1,282 groups (54.6%); largest
    marketCap fails identically (54.9%) for the same reason. Even restricted to the 94
    groups where shares genuinely differ it still picks wrong 30 times (32%), because
    that difference is a 0.1-1.6% filing-vintage wobble, not a depositary ratio (0KGE.L
    reads 0.1% MORE shares than PAYX; 0KV3.L 1.3% more than RF). A depositary programme
    does not report a fraction of the shares -- it reports the issuer's shares. Register
    B-7 is real, but it is about `shareCountChange`, a TIME-SERIES DELTA; the share-count
    LEVEL is not a listing-size proxy.

    MEASURED failure rate, same 1,282 groups:
      largest share count (as briefed)      693 / 1,282  (54.1%)
      largest marketCap                     701 / 1,282  (54.7%)
      previous _investability_key            40 / 1,282   (3.1%)
      canonicity-first, then shares (this)    6 / 1,282   (0.47%)
    All six residual failures are groups with NO canonical member at all (TRTN-PA..PE,
    GLOP-PA..PC, SEAL-PA/PB, SNV-PD/PE, TD-PFA/PFJ.TO, TRINI/TRINZ) -- nothing better
    exists, so keeping one is correct.

    *** 0.47% IS A LOWER BOUND ON THE TRUE FAILURE RATE, FOR TWO INDEPENDENT REASONS.
    Do not quote it as the failure rate. ***
      1. THE LABELLER IS THE RULE. "Structurally non-canonical" is scored with
         `_non_canonical_tag` -- the same function the ordering uses -- so the metric
         CANNOT COUNT a wrong pick that no marker recognises. That is not hypothetical:
         it is exactly how CIMN (a Chimera NOTES line) sat in the canonical tier and won
         its group on marketCap while scoring as a success.
      2. THE POPULATION EXCLUDES THE KNOWN FAILURE CLASS. All three groups where the
         shipped rule is KNOWN to pick the non-common -- CBE.PA/RBT.PA (Robertet
         certificat), PREVA.AS/VALUE.AS (Value8 preference), SMSD.L/SMSN.L (Samsung
         preferred GDR) -- are ABSENT from the panel: none of the six symbols appears in
         it, in `moatdf` or in `cdx_df`. The panel carries no Amsterdam/Paris
         fundamentals and Samsung's GDRs were never fetched, so the measurement is taken
         on a population that structurally cannot contain them.
    In all three, every marker (a)-(e) is ruled out BY CONSTRUCTION -- FMP gives the
    non-common the common's name verbatim, there is no -P suffix, the symbols share no
    prefix, and neither is a .KS line -- so the key falls through to the alphabetical
    tail and the NON-COMMON wins ('CBE' < 'RBT', 'PREVA' < 'VALUE', 'SMSD' < 'SMSN').
    ISIN is the only discriminator available for them; see the K4 note in
    _issuer_components and design/dedup-policy.md section 10.

    ISIN WIRED 2026-08-05 (CEO: "on the K-1, just pick one that makes the most sense"),
    AS TERM 5 -- IMMEDIATELY ABOVE THE ALPHABETICAL LAST RESORT AND NOWHERE HIGHER.
    That position is the whole safety argument, and it is chosen, not incidental:
      *  NEVER WORSE THAN TODAY, PROVABLY. Terms 1-4 are byte-unchanged and sit ABOVE it,
         so any group that today is decided by canonicity, share count, market cap or the
         symbol-shape tail is decided IDENTICALLY. The only groups ISIN can move are the
         ones that today fall through to raw alphabet -- 381 of 1,282 (29.7%) -- which is
         precisely the failure surface. The measured 0.47% canonicity-first failure rate
         cannot regress, because nothing ISIN does can outrank a canonicity marker.
      *  BIT-IDENTICAL WITH NO ISIN DATA. `_isin_plurality_term` returns a constant 0 for
         every member when the map is empty / NO member is mapped / the group has one
         distinct usable ISIN, and a constant added to every member of a group cannot
         change a sort. Every existing pickle is in that state (no isindic_fmp_*.pickle
         exists yet), so this change is a NO-OP until the next profile build lands.
         Asserted in test_dedup_issuer (test_isin_absent_is_bit_identical).
      *  PARTIAL ISIN DATA CANNOT REORDER A GROUP EITHER (reviewer, 2026-08-05). The
         first cut penalised an UNMAPPED member relative to a mapped one, so a
         half-populated map moved survivors on data availability alone. Fixed in
         `_isin_plurality_term`: absence now ties with a singleton, and the
         abstain decision is taken before this line's ISIN is read. See that note.
      *  WHAT IT DOES NOT DO. It does NOT resolve the three known-wrong groups. All three
         are 2-member with (almost certainly) 2 distinct ISINs, where plurality ties. An
         ISIN carries no security-type field, so no honest rule reads "common" out of one;
         see the long note in `_isin_plurality_term`. UNVERIFIED, and flagged as such: no
         isindic pickle exists yet and the six symbols are absent from the panel, so the
         actual ISIN values for those groups have NOT been read -- the claim that they
         differ rests on issuer structure, not on this repo's data.

    `sector_map` IS ACCEPTED AND IGNORED, deliberately and not by accident. It used to
    be criterion 1 ("prefer a line the sector map already tags"), which meant FMP's
    sector TAGGING decided which ticker the CEO saw. Sector propagation happens
    independently in dedup_to_issuers and does not need the survivor to be the tagged
    line. The parameter stays in the signature so no caller breaks.
    """
    nm = (names or {}).get(sym, '') if names else ''
    noncanon = 1 if _non_canonical_tag(sym, nm, group) else 0
    sh = val_fn(sym, 'weightedAverageShsOut')
    sh = sh if sh is not None else -1.0
    mc = val_fn(sym, 'marketCap')
    mc = mc if mc is not None else -1.0
    digitpfx = 1 if sym[:1].isdigit() else 0
    punct = sum(ch in '-.' for ch in sym)
    imap = _isin_map_cached() if isin_map is None else isin_map
    isin_t = _isin_plurality_term(sym, group, imap)
    vmap = _volavg_map_cached() if volavg_map is None else volavg_map
    vol_t = _volavg_liquidity_term(sym, group, vmap)
    volraw_t = _volavg_raw_liquidity_term(sym, group, vmap)
    #  vol_t BEFORE isin_t -- CEO ruling 2026-08-06; see the term list above.
    #  volraw_t AFTER isin_t and immediately before `sym` -- CEO ruling 2026-08-08.
    return (noncanon, -sh, -mc, digitpfx, punct, len(sym), vol_t, isin_t, volraw_t, sym)


#  THE SURVIVOR-KEY TERM NAMES, IN KEY ORDER.  One name per element of the tuple
#  `_investability_key` returns, so `dedup_to_issuers` can report WHICH TERM decided a
#  group instead of leaving an operator to reconstruct it from the values.  Kept adjacent
#  to the key itself: adding a term without adding its name here is a bug the assertion in
#  `_deciding_term` turns into an exception rather than a silently mislabelled column.
_KEY_TERM_NAMES = ('canonicity', 'shares', 'marketCap', 'digit_prefix', 'punctuation',
                   'symbol_length', 'volavg', 'isin_plurality', 'volavg_raw',
                   'alphabetical')
_VOL_TERM_IX = _KEY_TERM_NAMES.index('volavg')
_ISIN_TERM_IX = _KEY_TERM_NAMES.index('isin_plurality')
#  `volavg_raw` is NAMED SEPARATELY from `volavg` on purpose, not merged into it: the two
#  terms carry very different evidential weight (a >=10x liquidity gap vs possibly a 1.03x
#  one), and `decided_by` is the column an operator uses to judge whether to trust a pick.
#  Collapsing them would let the weak term borrow the confident one's name.
_VOLRAW_TERM_IX = _KEY_TERM_NAMES.index('volavg_raw')


def _same_key_term(a, b):
    """Equality for ONE key element, with NaN == NaN.

    `val_fn` serves whatever the panel holds, and a missing share count / market cap can
    arrive as NaN rather than None. NaN != NaN, so a naive comparison would name `shares`
    as THE DECIDING TERM for two lines that are both merely unmeasured -- a fabricated
    decision in the one column added so an operator does not have to guess.
    """
    if a is b:
        return True
    try:
        if a == b:
            return True
    except Exception:
        return False
    try:
        return math.isnan(a) and math.isnan(b)
    except (TypeError, ValueError):
        return False


def _deciding_term(surv_key, other_key):
    """The name of the FIRST key element on which the survivor beat `other_key`.

    Purely observational: it reads the two keys the sort already produced and reports
    where they first diverge. It cannot move a survivor because it runs after the sort and
    returns a string nothing consumes.

    `''` means the two keys are element-for-element equal, which is only reachable for a
    symbol against itself (the last term is the symbol, which is unique within a group) --
    so an empty `decided_by` on a real dropped row is a signal that the grouping produced
    a duplicate member, not a normal state.
    """
    assert len(surv_key) == len(_KEY_TERM_NAMES), (
        '_investability_key returns %d terms but _KEY_TERM_NAMES has %d -- the audit '
        'column would mislabel the deciding term' % (len(surv_key), len(_KEY_TERM_NAMES)))
    for nm, a, b in zip(_KEY_TERM_NAMES, surv_key, other_key):
        if not _same_key_term(a, b):
            return nm
    return ''


def dedup_ranked(ranked_sources, cdx_df, names, scores=None, sector_map=None,
                 isin_map=None):
    """Collapse same-issuer lines in a RANK-ORDERED source list to ONE line per issuer.

    The issuer is represented at its BEST RANK POSITION; the surviving TICKER is the
    group's most CANONICAL line (`_investability_key`), which is not necessarily the
    best-ranked one. Order-preserving.

    SURVIVOR RULE CHANGED 2026-08-05: CANONICITY OVERRIDES RANK. This is the design
    decision the previous version explicitly deferred ("needs a design decision, audit
    M-5"), and it is now made. Previously the highest-RANKED line survived and
    investability broke only EXACT-score ties, which left the cited case standing: HNNAZ
    (Hennessy Advisors 4.875% NOTES) survived over HNNA at AggScore 0.1159 vs 0.1026 --
    not a tie, so the score really did prefer the notes line.

    WHY OVERRIDING THE SCORE IS RIGHT, AND WHY IT IS NOT THROWING INFORMATION AWAY.
    A score computed on a notes / preferred / depositary line is a score of THE ISSUER's
    fundamentals attached to an instrument the CEO is not buying, so the rank difference
    is an artifact rather than information -- and it is measurably an artifact:

      *  THE DERIVED PRICE IS IDENTICAL ACROSS AN ISSUER'S LINES. price =
         marketCap / weightedAverageShsOut (getData_fmp), and both are issuer-level, so
         they cancel: 0HQ7.L and BKE both 54.80; 0R2J.L, AEM and AEM.TO all 168.56;
         HEN.DE and HEN3.DE both 66.66 (which is itself a finding -- Henkel's ordinary
         and preferred do NOT trade at the same price, so FMP is serving one
         issuer-level price to both lines). 9 of the 19 duplicate groups in the emitted
         2026-01-09 ranking therefore carry a BYTE-IDENTICAL AggScore.
      *  THE OTHER 10 GROUPS DIFFER ON SCORE -- AEM.TO -0.3320 vs AEM -0.3605 vs 0R2J.L
         -0.3635 -- and the reason is NOT what the design spec says it is. The spec
         attributes it to "history depth and reporting-date alignment". MEASURED, that is
         wrong: of those 10 groups, ROW COUNT differs in only 1 and DATE SPAN in only 2,
         so 8 of 10 differ on score with the SAME number of rows over the SAME date range.
         What actually differs is the ROW VALUES. Aligning AEM against AEM.TO on their 22
         common dates, `netIncome` differs on 20 of 22, `operatingIncome` and
         `interestExpense` on 22 of 22, and `price` on 20 of 22 -- with no NaN mismatch
         anywhere, so this is not a coverage gap. "Identical price, identical
         fundamentals" is true only of the LATEST row, which is the only row K1
         fingerprints; the HISTORY the score integrates over is materially different per
         listing.
         THE POINT FOR THIS FUNCTION IS UNCHANGED AND IF ANYTHING STRONGER: the rank
         difference is an artifact of which listing's history FMP happened to serve, so
         the old rank-based survivor rule was systematically keeping whichever line that
         artifact favoured. Canonicity-first stops the pipeline selecting on it.
         WHAT IS NOT FIXED HERE, and is a finding in its own right (adjacent to register
         C-7): an issuer's listings carry DIFFERENT HISTORICAL STATEMENTS AND DIFFERENT
         HISTORICAL PRICES, so every history-integrating metric is listing-dependent.
         That also refutes "prefer the line with the most history" as a tie-break -- see
         design/dedup-policy.md section 10, where it is costed and demoted.

    So the damage from a wrong pick was never price error. It is (1) UNTRADEABILITY --
    an LSE IOB 0*.L line is not a mildly worse pick, it is not a pick at all (register
    J-1), and raw rank picked the IOB or preferred line in 7 of the top 20; (2) score
    NOISE selection, above; (3) silent DOUBLE WEIGHT -- before dedup one bet occupies two
    slots at two different scores, so a 20-name list is really 13 companies.

    THE ISSUER KEEPS ITS RANK POSITION; ONLY THE TICKER CHANGES -- IN `kept`. Callers
    that consume the returned `kept` LIST get that property for free (it is built in rank
    order with the survivor placed at the group's best position). A caller that instead
    filters its own frame by `source in set(kept)` -- postBoRank._dedup_issuers_in_ranking
    does exactly that -- leaves the survivor at ITS OWN row, so an issuer whose canonical
    line ranked lower is represented at the LOWER position. That was invisible before,
    because the survivor WAS the best-ranked line. It is a real consequence of this
    change and it is called out here rather than silently absorbed.

    `scores` and `sector_map` ARE ACCEPTED AND UNUSED. `scores` drove the old exact-tie
    tie-break, which canonicity-first subsumes entirely (a tie is just the case where
    every ordering term below canonicity was already equal); `sector_map` fed a criterion
    that has been deliberately dropped (see _investability_key). Both stay in the
    signature so existing call sites -- postBoRank, stage2_pit, baseline_tools -- keep
    working untouched.

    CEO standing principle: NO duplicate issuers in the emitted top-N -- a dual-listing
    or share-class must not occupy two slots (the TFPM / TFPM.TO case). This is the
    SELECTION-TIME dedup: apply it to the full ranked list BEFORE taking head(N), so the
    emitted top-N contains N DISTINCT issuers. It reuses the EXACT issuer grouping
    (_issuer_components, keys K1/K2/K3/K4) the carve-out uses, so "same issuer" means the
    same thing across the pipeline -- and both sites now also share the SAME survivor
    rule, which they did not before.

    `isin_map` -- None (the default) means the process-wide profile map, so production
    behaviour is unchanged and this site groups on the SAME ISINs `dedup_to_issuers` does.
    Pass `{}` to run with K4 inert.  IT IS AN ARGUMENT RATHER THAN A GLOBAL FOR A REASON
    THAT COST A TEST: since K4 was wired, the DEFAULT reads whichever
    `isindic_fmp_*.pickle` is sitting in the repo root, so any caller that builds a
    synthetic frame out of REAL ticker symbols silently inherits those symbols' real
    ISINs.  `baseline_tools/test_live_dedup` did exactly that and its TFPM / TFPM.TO pair
    started merging -- correctly, they are one issuer -- on a run artifact that had
    nothing to do with the fixture.  A caller that wants a closed world must now say so.

    Returns (kept, dropped):
      kept    : deduped rank-ordered source list (>= 1 line per distinct issuer)
      dropped : list of (dropped_symbol, kept_survivor) in rank order (audit trail)
    """
    ranked = list(ranked_sources)
    imap = _isin_map_cached() if isin_map is None else isin_map
    comps, _latest, _val = _issuer_components(ranked, cdx_df, names, imap)
    root_of = {s: r for r, members in comps.items() for s in members}

    # --- pick the CANONICAL line in each multi-line group ------------------------
    # Unconditional, not tie-gated: canonicity overrides rank (see above).
    #  The SAME `imap` reaches the sort key, so the grouping and the pick cannot resolve
    #  one issuer's ISINs from two different maps.
    chosen = {}
    for r, members in comps.items():
        if len(members) < 2:
            continue
        chosen[r] = sorted(
            members,
            key=lambda m: _investability_key(m, _val, None, names, members, imap))[0]

    kept, dropped, done = [], [], {}
    for s in ranked:
        r = root_of.get(s, s)
        if r in done:
            if s != done[r]:        # never record the survivor as dropped-onto-itself
                dropped.append((s, done[r]))
        else:
            # The issuer is represented at its BEST rank position; the surviving TICKER
            # is the group's most CANONICAL line, which may be a lower-ranked one.
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
      (1) a line showing NO non-common marker (canonicity class -- see
          _non_canonical_tag), then
      (2) largest weightedAverageShsOut, then (3) largest latest market cap, then
      (4) not digit-prefixed, fewest punctuation, shortest, alphabetical.
    This is `_investability_key` verbatim -- dedup_ranked now uses the SAME rule, so the
    carve-out and the emitted ranking can no longer disagree about which line of an
    issuer is the real one (they could before: this site preferred the sector-tagged /
    biggest-cap line while dedup_ranked preferred the best-RANKED line).
    SECTOR PROPAGATION: the survivor inherits the group's known sector (majority of
    tagged members) -- this is what plugs the cross-listing leak regardless of which
    line survives. A conflict (two DIFFERENT known sectors in one group) is flagged.
    NOTE the survivor no longer has to be a sector-TAGGED line, which is the point:
    propagation happens below and does not depend on it, so FMP's sector tagging no
    longer decides which ticker the CEO sees.

    THE DROPPED-SIBLING AUDIT TRAIL IS THE MITIGATION FOR MERGING DUAL-CLASS COMMONS.
    A dual-class pair is one issuer, one set of statements, one economic bet, so it
    merges -- but the two classes trade at DIFFERENT PRICES, so which one you buy is a
    real 5-15% decision that merging would otherwise hide. The `report` frame therefore
    carries every dropped line's PRICE and SHARE COUNT alongside its symbol, so the
    consumer can show "AEM -- also listed as AEM.TO, 0R2J.L" and the CEO can pick the
    cheaper / more liquid line at execution. One slot in the ranking, all lines visible
    for the trade. (Wiring that into the emitted CSV is a separate change in the
    emission layer; the data is produced here.)

    THE RUN ANSWERS ITS OWN QUESTION ABOUT THE TIEBREAK (added 2026-08-06, reviewer).
    Before this, the frame recorded neither term value, so on the three K-1 groups
    (Robertet, Value8, Samsung) it was IMPOSSIBLE to tell from the run's own output whether
    volAvg SPOKE AND THE DROPPED LINE LOST or whether volAvg ABSTAINED and something else
    decided -- and group closure enlarged exactly those groups with the IOB lines most
    likely to report volAvg 0 or null, which is an abstention trigger (condition 1 of
    `_volavg_liquidity_term`). It was recoverable offline from the persisted volavgdic
    pickle, but that is a second pass over a ~5-hour artifact to answer a question the run
    already had the values for. Five OBSERVATIONAL columns, none of which the sort reads:

      decided_by       the name of the first `_investability_key` term on which the
                       survivor beat THIS dropped line (see `_KEY_TERM_NAMES`).
      dropped_vol_t    this dropped line's volAvg term value
      survivor_vol_t   the survivor's volAvg term value
      dropped_isin_t   this dropped line's ISIN-plurality term value
      survivor_isin_t  the survivor's ISIN-plurality term value

    THE RAW VOLUMES, ADDED 2026-08-08 (CEO).  The five columns above are TERM values, not
    readings, so when a group fell to the alphabet the artifact could not say WHETHER
    volume was close, absent or stale -- three states that demand three different responses
    and that a column of zeros renders identical.  That gap was hit for real: the 7 groups
    still reaching the alphabet on the 2026-08-08 run could not be diagnosed from the run's
    own output.  Four more OBSERVATIONAL columns close it, and they are what makes the weak
    `volavg_raw` tiebreak honest -- a pick made on a 1.03x margin is visible AS a 1.03x
    margin:

      dropped_volAvg        this dropped line's RAW average volume (NaN if none)
      dropped_volAvg_asof   its as-of date, or WHICH KIND of absence
      survivor_volAvg       the survivor's RAW average volume (NaN if none)
      survivor_volAvg_asof  the survivor's as-of date, or which kind of absence

    Same three-way absence semantics as the per-name CSV columns, from the SAME function
    (`_volavg_reading`), so they cannot drift: `not-captured` (absent from the map, or no
    map) / `no-reading` (present but null/0/non-finite -- an ABSTENTION TRIGGER, not a
    liquidity reading of zero) / `undated-capture` (a real value from the pre-dating pickle
    shape).  Two DIFFERENT as-of dates on one group is the date-disagreement abstention
    (condition 2) made visible -- that is the state that looks like a liquidity difference
    and is not one.  The raw term value is not emitted separately because it is exactly
    `-dropped_volAvg`; the reading IS the term.

    HOW TO READ ABSTAIN vs SPOKE-AND-LOST, which is the whole point. There is one row per
    DROPPED member and none for the survivor, so the survivor's value is repeated on every
    row of its group -- together those give EVERY member of the group, which is what the
    distinction needs (the winner's value alone cannot make it):
      * both columns NaN                -> the map does not exist, so the term was
                                           STRUCTURALLY SILENT and could not have decided
                                           anything. See below.
      * 0 for the survivor AND 0 for every dropped row of the group -> the term ABSTAINED
        (no reading, mixed as-of dates, or no member a decade behind). `_volavg_liquidity_term`
        emits abstention as a LITERAL 0 for every member, which is why this is readable as a
        value and not merely as a tie that cancels.
      * 0 for the survivor and 1 on a dropped row -> volAvg SPOKE and that line lost on
        liquidity. Whether it was DECISIVE is `decided_by == 'volavg'`; a term can speak
        and still be overridden by canonicity / shares / cap above it.
      * `decided_by` naming a term ABOVE volavg -> the group never reached the tiebreak.

    NO-DATA IS REPORTED AS NaN, NOT AS 0, DELIBERATELY. Every pickle in existence today has
    neither map, so both terms are a constant 0 for every member -- and a column of zeros
    reads exactly like "the term looked and found every line comparable", which is a
    DIFFERENT and much stronger statement than "there was nothing to look at". The absent-map
    case is therefore blanked, and `diagnostics['volavg_map_n']` / `['isin_map_n']` carry the
    entry counts so the state is a number an operator can see rather than an inference from a
    column of zeros.
    """
    from collections import Counter

    #  Loaded ONCE here and passed EXPLICITLY, where `_investability_key` would otherwise
    #  fetch the same memoised objects itself -- same values, so the pick is untouched; the
    #  reason to hoist them is that THIS site has to report whether each term had any data.
    #  HOISTED ABOVE THE GROUPING since 2026-08-13: K4 groups on ISIN and the survivor key
    #  orders on ISIN, so both now read the SAME map object rather than two lookups that
    #  could in principle disagree.
    imap, vmap = _isin_map_cached(), _volavg_map_cached()

    syms = list(BoScore_df['source'])
    comps, latest, _val = _issuer_components(syms, cdx_df, names, imap)
    _vol_col = (lambda t: t) if vmap else (lambda t: None)
    _isin_col = (lambda t: t) if imap else (lambda t: None)

    survivors, member_to_survivor, sector_override = set(), {}, {}
    rows, conflicts = [], []
    for members in comps.values():
        # THE shared "most investable line" key (module-level _investability_key), so
        # dedup_ranked and this survivor rule can never diverge. `sector_map` is passed as
        # None because the key ignores it -- kept explicit so a reader does not think this
        # site still prefers the tagged line.
        # Computed ONCE PER MEMBER and reused for the sort AND the audit columns below.
        # The key is a pure function of (sym, val_fn, names, members, imap, vmap) and the
        # last term is the unique symbol, so the ordering is total: memoising it cannot
        # change the survivor, it only stops recomputing it O(n log n) times per group.
        keys = {s: _investability_key(s, _val, None, names, members, imap, vmap)
                for s in members}
        surv = sorted(members, key=keys.__getitem__)[0]
        survivors.add(surv)
        secs = [x for x in (sector_map.get(m) for m in members) if _is_known_sector(x)]
        prop = None
        if secs:
            # THE OLD RULE, AND WHY IT WAS OVERRULED (CEO, 2026-08-10) -- kept, not deleted.
            # On a known-vs-known conflict this used to PREFER a cohort-relevant sector
            # (REIT / Mining / Financial) OUTRIGHT, so that a conflicting non-cohort sibling
            # tag (e.g. a baby-bond line mistagged 'Industrials') could not demote a REIT /
            # miner / BDC issuer out of its cohort. That intent is CORRECT and is preserved
            # below by the PLURALITY, which is the evidence the intent was really appealing
            # to. What the old rule additionally did -- and what the CEO overruled -- is
            # resolve a DEAD TIE in the cohort's favour, i.e. decide on no evidence at all.
            #
            # THE COST, MEASURED: `MAS` (Masco, ~$12bn building products) arrived tagged
            # `{'Consumer Cyclical': 1, 'Basic Materials': 1}` -- a 1-1 tie -- was routed to
            # Basic Materials by this preference, landed in the Mining cohort, and was ejected
            # there by `equityPositive`. Its negative book equity is buybacks, not distress;
            # the flag is a miner's balance-sheet floor and Masco is not a miner. A tie is not
            # evidence for a cohort, and a cohort carries SPECIALIST SOLVENCY FLAGS, so
            # guessing into one is strictly more dangerous than guessing out of one: the
            # general pool's flag set is the one designed for a company we cannot classify.
            #
            # THE RULE NOW: plurality on the FULL tag list, cohort-agnostic. A unique winner
            # is taken (so 2 REIT tags still beat 1 mistagged Industrials -- the protective
            # case survives). A TIE that would otherwise reach a cohort is AMBIGUOUS and the
            # issuer routes to GENERAL. A tie between two non-cohort sectors already routed to
            # general and still does; it is marked ambiguous anyway, because "we could not
            # tell" is worth recording even when it changes nothing.
            #
            # MEASURED against the map the 2026-08-10 run ACTUALLY USED (see the taxonomy
            # warning above -- the repo's copy answers a different question): 23 known-vs-known
            # conflicts, of which the OLD rule put 9 into a cohort that the plurality alone
            # would move out. The SHIPPED rule moves 7:
            #   FRU.TO, CAR, KEEL, HIVE.TO -> general;  DML.TO, AQMS -> general;
            #   LADR -> REIT (not general -- its own primary reads Real Estate, and only the
            #                 LSE sibling said Financial Services; a commercial-mortgage REIT
            #                 landing in REIT is a BETTER answer than either previous rule)
            # and two names the all-lines plurality would have moved STAY PUT, correctly:
            #   MAS  stays Mining -- its own primary is tagged Basic Materials and only the
            #        LSE IOB sibling said Consumer Cyclical. The vendor genuinely classifies a
            #        building-products company as a materials name; that is not an ambiguity
            #        for this rule to resolve.
            #   WY   stays REIT, same shape.
            #
            # NOT DONE HERE, AND EXPLICITLY PARKED BY THE CEO: deciding the cohort from the
            # BUSINESS MODEL rather than from the vendor's sector tag. That is the real fix --
            # and `MAS` is precisely the case that needs it, since no tiebreak can improve on a
            # correct-but-unhelpful vendor tag on the only line that has one.
            # ONLY THE ISSUER'S OWN EQUITY LINES GET A VOTE (reviewer S2, 2026-08-10).
            # A baby bond, a preferred or an IOB line carries the vendor's classification of
            # THE INSTRUMENT, not of the ISSUER, so counting it is counting the wrong thing --
            # and because there can be SEVERAL of them against ONE equity line, they can carry
            # the plurality outright.
            #
            # THE CASE THAT FORCED IT: `AFG` (American Financial Group) is tagged Financial
            # Services x1 -- the equity -- against Industrials x2, which are `AFGB` and `AFGE`,
            # its subordinated debentures.  Plurality alone therefore moved a P&C insurer out
            # of BalanceSheetFin into the general pool, and NOT as an ambiguity: it was a
            # confident wrong answer, unflagged.  Plurality protects the issuer only when the
            # correct tag is in the majority, which is exactly what a stack of bond lines
            # denies it.
            #
            # THE FILTER IS `_non_canonical_tag`, NOT A NEW HEURISTIC -- the same function the
            # survivor pick already trusts to tell an instrument from an issuer's common line
            # (it tags `AFGB`/`AFGE` `name-vocabulary:coupon-rate`, preferreds
            # `preferred-suffix`, `.L` IOB lines `lse-iob`).  Reusing it means the vote and the
            # pick cannot disagree about what a non-equity line is.
            #
            # THE CONFLICT REPORT STILL COUNTS EVERY MEMBER, deliberately: a reader must be
            # able to see that a bond line disagreed and was NOT counted, which a report
            # restricted to the voters would hide.
            voters = [x for m, x in zip(members, (sector_map.get(m) for m in members))
                      if _is_known_sector(x) and not _non_canonical_tag(
                          m, names.get(m, ''), members)]
            # A group of ONLY non-equity lines keeps the old behaviour rather than losing its
            # sector: no equity line exists to be overruled, so there is nothing to protect.
            pool = voters if voters else secs
            counts = Counter(pool)
            ranked = counts.most_common()
            tied = len(ranked) > 1 and ranked[0][1] == ranked[1][1]
            if tied:
                prop = AMBIGUOUS_SECTOR
            else:
                prop = ranked[0][0]
            if len(set(secs)) > 1:
                #  Reported with the FULL member counts and, when they differ, the counts that
                #  actually VOTED -- so "why did this resolve that way" is answerable from the
                #  diagnostic alone.
                _entry = dict(counts) if voters == secs else {
                    'all_lines': dict(Counter(secs)), 'equity_voters': dict(counts)}
                conflicts.append((surv, _entry, prop))
            sector_override[surv] = prop
        #  Read ONCE per group, not once per emitted cell. `_volavg_reading` is pure, so
        #  this cannot change a value; it stops the survivor's reading being recomputed for
        #  every dropped row of its group (and each member's being computed twice).
        surv_v, surv_asof = _volavg_reading(surv, vmap)
        for m in members:
            member_to_survivor[m] = surv
            if m != surv:
                m_v, m_asof = _volavg_reading(m, vmap)
                rows.append((m, surv, names.get(m, ''), sector_map.get(m, ''), prop,
                             '|'.join(sorted(members)),
                             _non_canonical_tag(m, names.get(m, ''), members),
                             _val(m, 'price'), _val(m, 'weightedAverageShsOut'),
                             _val(m, 'marketCap'),
                             #  OBSERVATIONAL ONLY -- read off the keys the sort has
                             #  already produced, appended AFTER `surv` is fixed.
                             _deciding_term(keys[surv], keys[m]),
                             _vol_col(keys[m][_VOL_TERM_IX]),
                             _vol_col(keys[surv][_VOL_TERM_IX]),
                             _isin_col(keys[m][_ISIN_TERM_IX]),
                             _isin_col(keys[surv][_ISIN_TERM_IX]),
                             #  The RAW readings behind those term values, so a reader can
                             #  see the MARGIN and whether it was close / absent / stale.
                             m_v, m_asof, surv_v, surv_asof))
    report = pd.DataFrame(rows, columns=['dropped', 'survivor', 'name',
                                         'orig_sector', 'propagated_sector', 'issuer_group',
                                         'non_canonical_tag', 'dropped_price',
                                         'dropped_shares', 'dropped_marketCap',
                                         'decided_by', 'dropped_vol_t', 'survivor_vol_t',
                                         'dropped_isin_t', 'survivor_isin_t',
                                         'dropped_volAvg', 'dropped_volAvg_asof',
                                         'survivor_volAvg', 'survivor_volAvg_asof'])
    #  WHICH TERM ACTUALLY DECIDED, AS COUNTS (added 2026-08-08, reviewer F4).  The
    #  `decided_by` column already carries this per row, but only inside a frame nobody was
    #  reading -- so the ONE question the weak raw tiebreak has to answer about itself ("how
    #  many groups did it take, and how many still fell to the raw alphabet?") could not be
    #  answered from a run at all.  It is a per-DROPPED-ROW count, not per group: a 3-member
    #  group contributes two rows, which is the right denominator for "how often did this
    #  term have to break a tie".  `n_decided_volavg_raw` is broken out by name because it
    #  is the number that closes the open question, and `n_decided_alphabetical` beside it
    #  is what the raw term is supposed to be driving DOWN.
    _decided = Counter(report['decided_by']) if len(report) else Counter()
    diagnostics = {'n_lines_in': len(syms), 'n_issuers_out': len(comps),
                   'n_collapsed': len(syms) - len(comps),
                   #  The state behind the blanked term columns, as a NUMBER: 0 = no map on
                   #  disk, so `*_vol_t` / `*_isin_t` are NaN and neither term participated.
                   'volavg_map_n': len(vmap), 'isin_map_n': len(imap),
                   'decided_by_counts': {t: int(_decided.get(t, 0))
                                         for t in _KEY_TERM_NAMES},
                   'n_decided_volavg': int(_decided.get('volavg', 0)),
                   'n_decided_volavg_raw': int(_decided.get('volavg_raw', 0)),
                   'n_decided_isin_plurality': int(_decided.get('isin_plurality', 0)),
                   'n_decided_alphabetical': int(_decided.get('alphabetical', 0)),
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

    # COVERAGE GUARD (2026-08-02).  "Non-empty" is NOT the property this carve needs.
    # The abort above only catches a map that is entirely absent, so a map that is
    # merely WRONG-SIZED -- e.g. built from a 142-name subset universe and then applied
    # to a 10,693-name pool -- passed straight through it while REIT and Mining leaked
    # wholesale, which is the exact failure the abort exists to prevent, reached by a
    # different route. So check what the map actually COVERS of the pool in hand.
    #
    # THRESHOLDS ARE GROUNDED IN MEASUREMENT, NOT PICKED (2026-08-02, live maps vs the
    # 2026-01-08 panel): a HEALTHY run covers 87.1% of its pool by sector (13% of names
    # legitimately have no FMP profile and fall to 'general'), and 98.9% by industry.
    # The poisoned-subset case measures 1.4%. Those are two orders of magnitude apart,
    # so any cut between them separates them cleanly:
    #   < 50% -> ABORT. Not a degradation, a wrong artifact. Same reasoning as the
    #            empty-map abort: refuse to ship a pool that only LOOKS carved.
    #   < 75% -> WARN loudly and proceed. Well below the measured 87.1% norm, so this
    #            does not fire on a healthy run, but a real erosion becomes visible.
    _pool = [s for s in symbols if isinstance(s, str)]
    _covered = sum(1 for s in _pool if s in sector_map)
    _frac = (_covered / len(_pool)) if _pool else 1.0
    if _pool and _frac < SECTOR_COVERAGE_ABORT_BELOW:
        msg = ("carveOut: sector map covers only %d of %d pool sources (%.1f%%) -- far "
               "below the ~87%% a healthy run shows. The map on disk was almost "
               "certainly built from a DIFFERENT (smaller) universe than the one being "
               "scored; rebuild it from a full universe." % (_covered, len(_pool), 100 * _frac))
        bang = "!" * 78
        banner = "\n".join([
            "", bang,
            "!!! CARVE-OUT ABORTED -- SECTOR MAP DOES NOT COVER THIS POOL !!!",
            "!!!   pool sources : %d" % len(_pool),
            "!!!   with a sector: %d  (%.1f%%)   healthy runs measure ~%.0f%%"
            % (_covered, 100 * _frac, 100 * SECTOR_COVERAGE_HEALTHY_REF),
            "!!! A non-empty but WRONG-SIZED map is the empty-map failure wearing a",
            "!!! disguise: REIT & Mining would be carved from a fraction of the pool",
            "!!! and the rest would leak into general, while the output still LOOKS",
            "!!! carved. Refusing to proceed.",
            "!!! LIKELY CAUSE: the maps were authored by a SUBSET run (e.g.",
            "!!! -tickerfilter stock_TEST1). Delete them and rebuild from a full",
            "!!! exchange-defined universe, or call findAllSectorsViaProfile.",
            "!!! " + msg, bang, ""])
        print(banner, file=sys.stderr, flush=True)
        print(banner, flush=True)
        raise RuntimeError(msg)
    if _pool and _frac < SECTOR_COVERAGE_WARN_BELOW:
        bang = "!" * 78
        wbanner = "\n".join([
            "", bang,
            "!!! CARVE-OUT WARNING -- THIN SECTOR-MAP COVERAGE !!!",
            "!!!   %d of %d pool sources have a sector (%.1f%%); healthy ~%.0f%%."
            % (_covered, len(_pool), 100 * _frac, 100 * SECTOR_COVERAGE_HEALTHY_REF),
            "!!! Uncovered names cannot be carved and fall to the GENERAL pool, so",
            "!!! some miners/REITs are probably leaking. Run PROCEEDS -- but treat",
            "!!! the cohort counts as understated and consider rebuilding the map.",
            bang, ""])
        print(wbanner, file=sys.stderr, flush=True)
        print(wbanner, flush=True)
    else:
        print('CARVE-OUT sector-map coverage: %d of %d pool sources (%.1f%%).'
              % (_covered, len(_pool), 100 * _frac), flush=True)

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
        #  *** THE SURVIVOR REPORT LEAVES MEMORY (2026-08-08, reviewer F1). ***
        #  Until now this frame existed ONLY at
        #  partition_universe(...)['diagnostics']['dedup']['report'] -- no caller read it,
        #  no CSV carried it, and no transfer pattern matched it. So the map that decides
        #  WHICH TICKER SURVIVES could decide a group on a 1.03x volume margin and NOTHING
        #  anywhere said so. That is not a documentation gap: the raw-volume tiebreak's
        #  entire answer to "a weak signal LOOKS principled in the artifact" is that the
        #  artifact shows the margin, and there was no artifact. The same diff that widened
        #  the transfer manifest to close evidence gaps had missed the record of what those
        #  maps decided, which is the exact failure mode 84abd40 exists to prevent.
        #  IT GOES AT THE REPO ROOT SINCE 2026-08-10 (CEO), NOT IN `output/`.  This block
        #  used to say it went in `output/` "which Sbocker's allowlist_dirs already ships
        #  WHOLE, so the evidence travels without adding a pattern that could silently stop
        #  matching".  The 2026-08-10 run is the counter-example that retired that argument:
        #  `output/` did not travel AT ALL, so the 08-10 survivor report never left the
        #  machine, while every root-level artifact from the same run did.  The dedup
        #  breakdown for that date was recoverable only because a copy happened to be inside
        #  a pickle -- which is luck, not evidence design.  It now has its own top-level
        #  manifest pattern in `Sbocker.allowlist_patterns`.  Directory:
        #  `transfer_utils.EVIDENCE_DIR`.
        try:
            _dd_rep = dedup_diag.get('report')
            if _dd_rep is not None and len(_dd_rep):
                os.makedirs(_tu.EVIDENCE_DIR, exist_ok=True)
                _dd_fn = os.path.join(
                    _tu.EVIDENCE_DIR, 'DedupSurvivorReport_%s.csv'
                    % pd.Timestamp.today().strftime('%Y-%m-%d'))
                _dd_rep.to_csv(_dd_fn, index=False)
                print('  dedup survivor report written to: %s' % _dd_fn, flush=True)
        except Exception as _e:
            print('  WARNING: could not write dedup survivor report (%s)' % _e, flush=True)
        #  WHICH TERM DECIDED, as a line an operator reads without opening the CSV. The two
        #  that matter are the weak tiebreak (what it took) and the alphabet (what is left).
        print("carveOut dedup: decided by -- %s"
              % ', '.join('%s=%d' % (t, n)
                          for t, n in dedup_diag['decided_by_counts'].items() if n),
              flush=True)
        if dedup_diag['n_decided_volavg_raw']:
            print("  NOTE: %d dropped line(s) were decided by the WEAK raw-volume tiebreak "
                  "(volavg_raw) -- a near-tie in volume is weak evidence; the margin is in "
                  "the *_volAvg columns of the CSV above. %d still fell to the raw alphabet."
                  % (dedup_diag['n_decided_volavg_raw'],
                     dedup_diag['n_decided_alphabetical']), flush=True)

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
    # NO GUESS -- the floor is applied ONLY where the reporting currency is really known
    # (MD + senior-dev joint decision, 2026-08-06; CEO delegated: "make a decision and
    # make it clear in reporting how it is done so we can change it if it is weird and we
    # know why"). Until now the floor converted with the coarse exchange-suffix guess ON
    # and then fell back to the RAW mixed-currency column, so EVERY exclusion on saved
    # (pre-reportedCurrency) data rested on an assumption. $25M is the most ABSOLUTE edge
    # in the pipeline and absolute edges do not cancel a systematic currency error -- the
    # same reasoning that removed the guess from the size tilt (register D-5) applies here
    # a fortiori, because this edge does not MIS-SCORE a name, it DELETES it: an excluded
    # name is never fetched, never scored, cannot appear, and no output says it was
    # dropped. That error is invisible and unrecoverable; a wrong INCLUSION is visible and
    # still has to survive the Stage-1 criteria, the veto and the top-100 cut. The feared
    # failure mode of keeping junk microcaps is already closed: an unresolvable currency
    # scores NEUTRAL on the size metric (stage2_metrics.MCAP_QUANT_MISSING), so a tiny
    # name cannot collect the small-cap reward, and every other market-cap metric in the
    # score is a same-currency RATIO (bookToPrice, salesToMarketCap, CFOtoMarketCap,
    # freeCashFlowToMarketCap), i.e. currency-neutral by construction.
    #
    # Measured on the 9,012-name 2026-01-08 panel (reportedCurrency absent, as on every
    # pickle saved before the next full fetch): the guessing floor excluded 1,092 names
    # (12.1% of the universe) -- 547 on a non-1.0 suffix rate, 545 on the raw-as-USD
    # rate-1.0 assumption -- and the gated floor excludes 0, because NOTHING resolves.
    # So on legacy panels this floor is now a NO-OP, which is a real change and is why the
    # degradation is announced LOUDLY below and flagged in the diagnostics rather than
    # left to be inferred from a count. It self-heals on the next full fetch, after which
    # the floor is correct for every name whose reportedCurrency resolves.
    #
    # A name with NO market cap at all was always KEPT (never dropped on missing); an
    # UNRESOLVABLE CURRENCY is now the same case -- unknown, therefore kept.
    mcap_raw = bs['source'].map(fund['marketCap'])
    _mcu = marketcap_usd_by_source(cdx_df)
    mcap = bs['source'].map(_mcu)
    below = mcap.notna() & (mcap < mcap_floor)
    n_below = int(below.sum())
    n_unknown_mcap = int(mcap.isna().sum())
    below_sources = set(bs.loc[below, 'source'])
    bs_floored = bs[~below].reset_index(drop=True)

    # --- announce the gating, and NAME the names whose exclusion depended on the guess --
    # Two integers is not "loud, never silent" for a floor that stopped firing. The set
    # that matters is: raw market cap below the floor, but currency unknown -> KEPT. Those
    # are exactly the names the old guessing floor would (or might) have deleted, and the
    # ones to look at if this decision turns out to be wrong.
    _kept_unknown_raw_below = bs.loc[mcap.isna() & mcap_raw.notna()
                                     & (mcap_raw < mcap_floor), 'source']
    _floor_pending = not currency_data_present(cdx_df)
    n_kept_unknown_raw_below = int(len(set(_kept_unknown_raw_below)))
    if _floor_pending:
        bang = "!" * 78
        wbanner = "\n".join([
            "", bang,
            "!!! CARVE-OUT WARNING -- $%.0fM UNIVERSE FLOOR NOT ENFORCED !!!"
            % (mcap_floor / 1e6),
            "!!!   reportedCurrency has not flowed on this data, so NO name has a known",
            "!!!   USD market cap and the floor excluded 0 of %d names. This is the"
            % len(bs),
            "!!!   DELIBERATE choice (2026-08-06): an absolute USD floor is not applied to",
            "!!!   a market cap in an UNKNOWN currency, because a wrong exclusion is",
            "!!!   invisible and unrecoverable. %d name(s) with a RAW market cap below the"
            % n_kept_unknown_raw_below,
            "!!!   floor are therefore IN the universe -- some are genuinely sub-floor.",
            "!!!   Run PROCEEDS; the size tilt scores them NEUTRAL. Self-heals on the",
            "!!!   next full fetch. Do NOT read this run's universe as floor-filtered.",
            bang, ""])
        print(wbanner, file=sys.stderr, flush=True)
        print(wbanner, flush=True)
    else:
        print("carveOut floor: applied in USD from reportedCurrency ONLY (no suffix "
              "guess) -- %d excluded, %d kept with an unknown/unresolvable currency "
              "(%d of those have a RAW market cap below the floor)"
              % (n_below, n_unknown_mcap, n_kept_unknown_raw_below), flush=True)
        # --- PARTIAL COVERAGE IS NOT "ENFORCED" (F-2, reviewer 2026-08-08) -------------
        # `currency_data_present` is a BOOLEAN: it says the floor ran, not how much of the
        # universe it reached. A feed covering only a couple of currencies still lands
        # here, and downstream stamps `floor_enforced: True` over a universe where most
        # names were never floored at all. State the FRACTION every time, and banner it
        # when it is low -- the run proceeds (a partial floor is not a wrong floor, and
        # every unfloored name is KEPT), but the label must not outrun the fact.
        try:
            import fx_rates as _fxr
            _min_cov = _fxr.FX_MIN_PANEL_COVERAGE
        except Exception:
            _min_cov = 0.90
        try:
            _cov_n, _cov_tot, _cov_frac = currency_coverage(cdx_df)
        except Exception as _cove:
            # A COVERAGE READOUT MUST NOT COST THE RUN. Report the miss, don't fake a
            # number: a silent 100% here would be worse than the defect it reports on.
            print("carveOut floor: WARNING -- coverage not computed (%s: %s); the floor "
                  "itself is UNAFFECTED, but this run carries no coverage figure."
                  % (type(_cove).__name__, _cove), flush=True)
            _cov_n, _cov_tot, _cov_frac = 0, 0, 1.0
        print("carveOut floor: COVERAGE -- %d of %d source(s) (%.1f%%) have a USD market "
              "cap, so the floor was applied to that fraction of the universe."
              % (_cov_n, _cov_tot, 100.0 * _cov_frac), flush=True)
        if _cov_tot and _cov_frac < _min_cov:
            _cbang = "!" * 78
            _cbanner = "\n".join([
                "", _cbang,
                "!!! $%.0fM FLOOR ONLY PARTIALLY APPLIED -- %.1f%% COVERAGE !!!"
                % (mcap_floor / 1e6, 100.0 * _cov_frac),
                "!!!   %d of %d name(s) have NO resolvable USD market cap, so the floor"
                % (_cov_tot - _cov_n, _cov_tot),
                "!!!   never applied to them. They are KEPT and UNBANDED -- nothing is",
                "!!!   wrongly deleted -- but this universe is NOT floor-filtered end to",
                "!!!   end, and any 'floor_enforced' label on it means 'the floor ran',",
                "!!!   NOT 'every name passed it'. Usual cause: an FX feed that installed",
                "!!!   LIVE while resolving only part of the supported currency set --",
                "!!!   check the fx_rates block of RunProvenance and FxRates_*.csv.",
                _cbang, ""])
            print(_cbanner, file=sys.stderr, flush=True)
            print(_cbanner, flush=True)
    # The floor moves names IN and OUT of the universe, so NAME THEM -- two integers is
    # not "loud, never silent" for a universe change of this size. Same dated-CSV treatment
    # the share-class filter gets. THREE populations, because the CEO's condition for being
    # able to reverse the gating decision is knowing exactly which names it kept:
    #   kept_currency_unknown  raw cap < floor, currency unknown -> KEPT BY THE GATING.
    #                          This is the set the old guessing floor would have deleted;
    #                          it is the whole cost of the decision, by name.
    #   newly_excluded         raw cap >= floor but the KNOWN USD cap is below it (a real
    #                          currency correction, e.g. a GBP/EUR reporter just over the
    #                          raw cutoff).
    #   newly_kept             raw cap < floor but the KNOWN USD cap is at or above it.
    _flip_out = bs.loc[below & mcap_raw.notna() & (mcap_raw >= mcap_floor), 'source']
    _flip_in = bs.loc[~below & mcap.notna() & mcap_raw.notna()
                      & (mcap_raw < mcap_floor), 'source']
    print("carveOut floor: %d name(s) excluded that the raw-currency floor kept, "
          "%d name(s) kept that it wrongly excluded, %d name(s) kept because the "
          "reporting currency is UNKNOWN (gating, 2026-08-06)"
          % (len(_flip_out), len(_flip_in), n_kept_unknown_raw_below), flush=True)
    if len(_flip_out):
        print("  newly EXCLUDED: %s" % ', '.join(sorted(set(_flip_out))), flush=True)
    if len(_flip_in):
        print("  newly KEPT: %s" % ', '.join(sorted(set(_flip_in))), flush=True)
    if n_kept_unknown_raw_below:
        _ku = sorted(set(_kept_unknown_raw_below))
        print("  KEPT ON UNKNOWN CURRENCY (%d): %s%s"
              % (len(_ku), ', '.join(_ku[:40]),
                 '' if len(_ku) <= 40 else ' ... (full list in the CSV)'), flush=True)
    try:
        _ku_all = sorted(set(_kept_unknown_raw_below))
        _fx_rows = pd.DataFrame({
            'source': list(set(_flip_out)) + list(set(_flip_in)) + _ku_all,
            'direction': (['newly_excluded'] * len(set(_flip_out))
                          + ['newly_kept'] * len(set(_flip_in))
                          + ['kept_currency_unknown'] * len(_ku_all))})
        _fx_rows['marketCap_raw'] = _fx_rows['source'].map(fund['marketCap'])
        _fx_rows['marketCap_usd'] = _fx_rows['source'].map(_mcu)
        # The rate the OLD guessing floor would have used, for reference only -- it is no
        # longer applied to anything. Kept in the CSV so the reversal is a one-line change
        # with the evidence already in hand.
        _fx_rows['retired_suffix_fx_rate'] = _fx_rows['source'].map(_suffix_fx_to_usd)
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
        # --- floor provenance (gating, 2026-08-06). `mcap_floor` alone says what the floor
        # WAS ASKED to be; these three say what it ACTUALLY DID, so no consumer can present
        # a "$25M-floored universe" that was never floored. generate_presentation reads
        # `floor_enforced` before it claims the floor in the deck.
        'floor_enforced': bool(currency_data_present(cdx_df)),
        'floor_currency_pending': bool(_floor_pending),
        'n_kept_currency_unknown_raw_below_floor': n_kept_unknown_raw_below,
        'industry_coverage': (_ind_cov, len(symbols)),   # (names with a real industry, universe)
        'vehicle_caught': veh,             # FIN-1 pre-floor; `below_floor` reconciles to post-floor
        'finmanager_caught': _caught(FIN2_MANAGER),
        'balancesheet_caught': _caught(FIN3_BALSHEET),
        'dedup': dedup_diag,     # None if dedup disabled; else {n_lines_in,n_issuers_out,n_collapsed,report,...}
    }
    return {'general': general, 'cohorts': cohorts, 'labels': bs.set_index('source')['carve_label'],
            'diagnostics': diagnostics}
