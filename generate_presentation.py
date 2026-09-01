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

#  A SECOND COPY of reviewReference.PLAYBOOK_METRICS, and the duplication is the hazard:
#  `shareCountChange` / `longTermDebtChange` (E-2, 2026-08-04) had to be added in BOTH places,
#  and the coverage guard lives in reviewReference
#  (`assert_allow_list_covers_the_weighted_metrics`).  `test_e2_weight_vector` asserts the two
#  lists agree, so they cannot drift apart silently again.  `grahamNumberToPrice` (D-10,
#  2026-08-05) is the second metric to be added in both places; see reviewReference for why.
PLAYBOOK_METRICS = [
    'returnOnCapitalEmployed', 'returnOnEquity', 'RoA', 'grossProfitMargin',
    'freeCashFlowYield', 'currentRatio', 'earnYield', 'revenueGrowth',
    'incomeQuality', 'Altman-Z', 'Piotroski', 'bVpRatio', 'tbVpRatio',
    'grahamNumberToPrice',
    'freeCashFlowPerShareGrowth', 'moatScore', 'CycleHeat',
    'shareCountChange', 'longTermDebtChange',
    #  2026-08-06: `interestCoverage` (S Tier 1, general + Mining/REIT/FIN-2) and
    #  `navPerShareGrowth` (FIN-1's R Tier 1).  See reviewReference for the proxy caveat that
    #  must travel with the second one.
    'interestCoverage', 'navPerShareGrowth',
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
# Metrics whose MARKER must come from the pool's own (period-corrected) basis, NOT the raw
# cdx latest row -- reviewer finding H2 (marker-vs-pool basis). These four are flow/stock
# ratios (a flow
# numerator over a balance-sheet or price denominator), so the pipeline's per-quarter
# annualization correction (reviewReference._pool_raw_fast -> stage2_metrics.postbm_metric
# with rpy=reporting_period.rows_per_year) SCALES them. The pool distribution is built on the
# corrected basis; taking the marker from the raw cdx latest row left a semi-annual company's
# dot plotting at 2.1-3.1x its true position against that pool (measured: GFRD.L ROCE 2.98x,
# ITV.L 3.13x; 41 of the shipped top-100 are non-quarterly reporters). Sourcing the marker
# from `raw_all` -- the SAME _pool_raw_fast output the pool is built from -- makes the two
# sides identical BY CONSTRUCTION rather than by a duplicated correction factor that could
# drift. Quarterly reporters have rpy=4 -> factor 1 -> bit-identical marker (verified).
# NOTE the same-period ratios (currentRatio, grossProfitMargin, incomeQuality) are NOT here:
# numerator and denominator share a period, so annualization cancels and they are unaffected.
POOL_BASIS_MARKERS = {'RoA', 'earnYield', 'returnOnEquity', 'returnOnCapitalEmployed'}

# DISPLAY-LAYER ANNUALIZATION (coordinator ruling, on top of the pipeline's 2026-07-25
# period-aware / annualize work -- NOT part of the earlier 2026-07-19 soundness-audit wave).
# The pipeline puts every flow/stock metric on a common PER-QUARTER basis
# (stage2_metrics: STAGE2_FLOW_OVER_STOCK via postbm_metric, plus free_cash_flow_yield --
# both multiply by reporting_period.per_quarter_factor). That is right for SCORING (z-scored,
# so only the cross-sectional ratio matters) but wrong for DISPLAY: the thresholds we render
# against (ROIC >15%/<10%, R1/R4's ROE>15%) are ANNUAL rules of thumb, and a human reading
# "ROIC: 20%" means twenty percent per YEAR. So we annualize these five for display.
#
# The factor is a CONSTANT x4: the pool basis is already per-quarter FOR EVERY SOURCE by
# construction (semi-annual rows were halved upstream), so no per-source rpy is needed here --
# per-quarter -> annual is x4 regardless of filing frequency (= rp.annualize_factor semantics
# applied to an already-per-quarter quantity).
#
# Applied ONCE, at the source, to BOTH the pool distributions and raw_all -- so the displayed
# value, the dot's marker, the percentile, the p10/p50/p90 spread, the verdict/flag rule inputs
# and Section C all inherit the SAME scale. A uniform scale leaves percentile positions and dot
# positions EXACTLY invariant, so marker == pool basis (1.00x) still holds by construction.
ANNUALIZED_DISPLAY_METRICS = {'RoA', 'earnYield', 'returnOnEquity',
                              'returnOnCapitalEmployed', 'freeCashFlowYield'}
ANNUALIZE_DISPLAY_FACTOR = 4.0

SECTION_G_METRICS = [
    # '(ann.)' marks the five metrics annualized for display (ANNUALIZED_DISPLAY_METRICS):
    # the pool basis is per-quarter, so these are x4'd to the annual convention a reader
    # (and the annual rule-of-thumb thresholds) assume.
    ('ROIC (ann.)',         'returnOnCapitalEmployed', 'returnOnCapitalEmployed',    'ratio'),
    # ROE/ROA formatted as % here to MATCH Section C's pct_format (reviewer F6: the same
    # metric must not render twice with inconsistent formatting on a cohort page).
    ('ROE (ann.)',          'returnOnEquity',          'returnOnEquity',             'pct'),
    ('ROA (ann.)',          'returnOnAssets',          'RoA',                        'pct'),
    ('Gross Margin',        'grossProfitMargin',       'grossProfitMargin',          'pct'),
    ('FCF Yield (ann.)',    None,                      'freeCashFlowYield',          'pct'),
    ('Current Ratio',       'currentRatio',            'currentRatio',               'ratio'),
    ('Earnings Yield (ann.)', 'earningsYield',         'earnYield',                  'pct'),
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


# =========================================================================== #
#  REPORTING EXPANSION -- CURRENCY, TRADED VALUE, DATA COVERAGE (CEO 2026-08-13) #
# =========================================================================== #
#  THE PRESENTATION RULE THIS BLOCK OBEYS, in the CEO's own words: "it needs to be
#  correctly presented, i.e. the correct suggestive nature of how it is presented (putting
#  a red flag next to high quality earning indicator is, for example, the exact opposite of
#  what I mean)."  Read as a design rule with teeth: A MARKER'S VISUAL WEIGHT AND VALENCE
#  MUST MATCH THE DIRECTION OF THE THING IT MARKS.  A marker pointing the wrong way is
#  WORSE than no marker, because the reader trusts the marker over the number.
#
#  So every field added here was assigned a valence BEFORE it was styled, and three of the
#  four turned out NOT to be simple good/bad axes:
#
#   FIELD                 VALENCE           HOW IT IS PRESENTED, AND WHY
#   -------------------   ---------------   ------------------------------------------------
#   priceCurrency         NONE (a unit)     Plain text beside the price, same weight as the
#                                           price.  A currency is neither good nor bad.  The
#                                           thing being FIXED is that the page printed
#                                           `${price}` unconditionally, which is not a
#                                           neutral rendering -- it is a false assertion for
#                                           every non-USD line.
#   market cap            NONE (a size)     Rendered from `marketCap_usd` and LABELLED USD.
#                                           `marketCap` is a reportedCurrency quantity, so
#                                           the page was rendering 000660.KS as
#                                           "$1,890,724.65B".  No marker: size is not merit.
#   dollarVolume_usd      NOT MONOTONE      Number shown NEUTRALLY, with a ONE-SIDED amber
#                                           caution at the THIN end only.  High traded value
#                                           earns NO positive marker: liquidity says nothing
#                                           about value, and this filter's thesis is
#                                           NEGLECTED small caps -- rewarding liquidity here
#                                           would suggest the opposite of the strategy.  The
#                                           thin-end caution is about the READER'S ability to
#                                           build and exit a position, and the wording says
#                                           so; it is not a claim about the business.
#   imputed_weight_share  MONOTONE, BAD     Amber >=20%, red >=50%, and worded as EPISTEMIC
#                                           ("how much of this score is measured") rather
#                                           than as quality.  It must read "we know little
#                                           about this name", never "this is a bad company".
#                                           It sits beside `forensicTag`, which can say
#                                           `clean` on a 93%-imputed name -- the tag is not
#                                           lying (it describes the FORENSIC checks) but the
#                                           two are orthogonal and the label says so.
#   veto ejection         HARD EXCLUSION    A run-level NEUTRAL card, not a per-name warning.
#                                           Ejected names are not on any page (they were
#                                           removed before ranking), so it is a statement
#                                           about the SCOPE of what the reader is looking at
#                                           -- the filter working as designed -- and it is
#                                           styled like the industry counter, not like the
#                                           red `.basis-warning`.
#
#  EVERY FIELD DEGRADES TO "—" WHEN ITS COLUMN IS ABSENT, and absence is never rendered as
#  zero.  Runs before 2026-08-13 carry no `priceCurrency` / `dollarVolume_usd` /
#  `imputed_weight_share`, and a 0 in any of them would be a confident false statement.

#  ~$25k/day is the CEO's own reference line (2026-08-13).  On the shipped 2026-08-13 run it
#  separates 529 of 3,145 universe names (16.8%) from the rest, and reaches only 3 of the
#  top-100 -- so it marks a genuine tail rather than colouring the page.
THIN_LIQUIDITY_USD_PER_DAY = 25_000.0
#  Imputation bands.  0.20 is `missing_data_fill_report`'s OWN "scored >=20% on fills"
#  reporting threshold, reused rather than re-invented so the run log and the deck agree.
#  0.50 is where the majority of the score is guesswork.
IMPUTED_SHARE_AMBER = 0.20
IMPUTED_SHARE_RED = 0.50


def money_format(v, decimals=2):
    """Compact magnitude for a USD amount: $1.43T / $1.23B / $45.6M / $12.3k / $987.

    `T` is not decoration: the top of this universe is a trillion-dollar name and the deck's
    old `f"${mktcap/1e9:.2f}B"` rendered META as "$1432.45B", which a reader has to count
    digits to size.  The whole point of the column is a magnitude read at a glance.
    """
    v = safe_float(v)
    if np.isnan(v):
        return "—"
    a = abs(v)
    for div, suf in ((1e12, 'T'), (1e9, 'B'), (1e6, 'M'), (1e3, 'k')):
        if a >= div:
            return f"${v/div:,.{decimals}f}{suf}"
    return f"${v:,.0f}"


def price_with_currency(price, currency):
    """The quoted price LABELLED with the currency it is quoted in.

    THE DEFECT THIS REPLACES: the page rendered `f"${price:.2f}"` unconditionally.  A `$` in
    front of a GBp or KRW number is not a neutral formatting choice -- it is a false
    statement, and it is the same class of defect as the AggScore CSV's FX-mixed
    `GrahamNumberToPrice` (register N-3).  An unknown currency yields a BARE number with an
    explicit "currency unknown" note rather than a guessed symbol: the exchange suffix does
    NOT determine the currency (SHEL.L quotes GBp and reports USD), so guessing is exactly
    the reasoning this pipeline has already been burned by.
    """
    p = safe_float(price)
    if np.isnan(p):
        return "—"
    cur = currency if isinstance(currency, str) and currency.strip() else None
    num = f"{p:,.2f}" if abs(p) < 1e4 else f"{p:,.0f}"
    if cur is None:
        return (f'{num} <span class="pctile" title="This run did not record the trading '
                f'currency of this listing. It is deliberately NOT guessed from the exchange '
                f'suffix -- SHEL.L quotes in pence and reports USD.">(currency unknown)</span>')
    return f'{escape(cur)}&nbsp;{num}'


def dollar_volume_cell(v):
    """Traded value per day, NEUTRAL, with a one-sided thin-liquidity caution.

    NO POSITIVE MARKER AT THE HIGH END, deliberately -- see the valence table above.  The
    caution fires only below ~$25k/day and is worded as a constraint on the READER (position
    sizing and exit), because that is the only thing the number supports; it is not a
    statement about the company.  Amber, not red: it does not disqualify a name, and this
    pipeline's standing ruling on volume is REPORT, NEVER SCREEN (register J-1).

    KNOWN LIMIT, carried into the tooltip rather than silently corrected: on a depositary
    (GDR/ADR) line FMP can report the HOME line's share volume against the DEPOSITARY line's
    price, which inflates the product by the depositary ratio.
    """
    f = safe_float(v)
    if np.isnan(f):
        return '—'
    title = ('Average daily traded VALUE = average share volume x quoted price x the run\'s '
             'live FX. Shown for comparability -- average VOLUME alone is a share count and '
             'is not comparable across listings. REPORT ONLY: nothing is screened, ranked or '
             'excluded on it. Caveat: on a GDR/ADR line the vendor can report the home '
             'line\'s volume against the depositary price, which overstates this.')
    cell = f'<span title="{escape(title)}">{money_format(f)}/day</span>'
    if f < THIN_LIQUIDITY_USD_PER_DAY:
        thin = ('Thin: below ~$25k of value traded per day. This constrains YOUR ability to '
                'build and exit a position at a sane price -- it is NOT a judgement on the '
                'business, and nothing was excluded on it.')
        cell += f' <span class="flag AMBER" title="{escape(thin)}">thin market</span>'
    return cell


def data_coverage_cell(imputed_share):
    """How much of this name's score was MEASURED rather than imputed.

    STATED AS COVERAGE, NOT AS A VERDICT.  The underlying number is
    `imputed_weight_share` -- the fraction of the scoring WEIGHT (not the count of columns;
    the weights span ~20x) that `normalizeAndDropNA` filled in at the pool median because the
    metric was missing.  A name at 0.93 was ranked almost entirely on fills.
    Rendered as the complement so the reader sees a coverage percentage falling, which is the
    direction the eye reads as worse, and worded epistemically: the risk is that we know
    little about this name, NOT that the name is bad.

    IT DOES NOT CONTRADICT `forensicTag`, and the tooltip says so explicitly -- on the shipped
    2026-08-13 top-100, STRT (93.2% imputed) and PET.TO (90.0%) are both tagged `clean`.  That
    tag describes the FORENSIC checks and is orthogonal to coverage; a reader who takes
    `clean` as "well-understood" is making an inference the tag never offered.
    """
    s = safe_float(imputed_share)
    if np.isnan(s):
        return '—'
    measured = max(0.0, min(1.0, 1.0 - s))
    title = ('Share of this name\'s SCORING WEIGHT that came from real measured metrics. The '
             'remainder was imputed at the pool median by the missing-data fill, so the rank '
             'rests on it. Independent of `forensicTag`: a name can be forensically `clean` '
             'and still be ~90% imputed -- the tag describes the forensic checks only. '
             'Source: MissingDataFillReport (imputed_weight_share).')
    cell = (f'<span title="{escape(title)}">{measured*100:.1f}% measured</span>')
    if s >= IMPUTED_SHARE_RED:
        cell += (f' <span class="flag RED" title="{escape(title)}">'
                 f'{s*100:.0f}% imputed — thin evidence</span>')
    elif s >= IMPUTED_SHARE_AMBER:
        cell += (f' <span class="flag AMBER" title="{escape(title)}">'
                 f'{s*100:.0f}% imputed</span>')
    return cell


def veto_scope_banner(veto_reports):
    """Run-level card naming what the Stage-1 veto removed BEFORE ranking; '' when unusable.

    A SCOPE STATEMENT, NOT A WARNING, and styled accordingly (the neutral `.industry-counter`
    card, not the red `.basis-warning`).  The veto is the filter working as designed; what
    the reader needs is that the shortlist below was drawn from a pool that had already lost
    a large fraction of its members, and from which one.  Measured on the shipped 2026-08-13
    run: 573 of 1,404 general-pool names (40.8%) were ejected before the head(100).

    NAMES ARE NOT LISTED HERE -- there can be hundreds and they are not on any page.  The
    card points at `Stage1VetoEjections_<date>.csv`, which the run now ships (register N-5).

    "EJECTED NOBODY" AND "DID NOT RUN" ARE KEPT DISTINCT, because that distinction is the
    whole reason `stage1_veto` reports per pool at all.
    """
    if not isinstance(veto_reports, dict):
        return ''
    if not veto_reports:
        return ('<div class="industry-counter"><span class="ic-head">Stage-1 veto</span> '
                'DID NOT RUN this run — every pool below is UN-VETOED. This is not the same '
                'as a veto that ran and ejected nobody.</div>')
    gen = veto_reports.get('general') or {}
    if not gen.get('enabled', True):
        return ('<div class="industry-counter"><span class="ic-head">Stage-1 veto</span> '
                'was OFF this run — no name was removed by it.</div>')
    parts = []
    for lab, r in sorted(veto_reports.items()):
        r = r or {}
        n_in, n_ej = r.get('n_in'), r.get('n_ejected')
        if not r.get('applies', True):
            parts.append(f'{escape(str(lab))}: not applicable')
        elif isinstance(n_in, int) and n_in > 0:
            parts.append(f'{escape(str(lab))}: {n_ej} of {n_in} '
                         f'({100.0*(n_ej or 0)/n_in:.1f}%)')
    if not parts:
        return ''
    return ('<div class="industry-counter">'
            '<span class="ic-head">Stage-1 veto</span> removed these names from each pool '
            '<em>before</em> ranking, so the shortlist below is drawn from what remained. '
            'Ejection is a hard exclusion on persistent solvency / earnings-reality flags — '
            'not a warning to weigh. '
            '<div class="ic-list">' + ' · '.join(parts) + '</div>'
            '<div class="leg-note">The ejected names themselves are in '
            '<code>Stage1VetoEjections_&lt;date&gt;.csv</code> beside this run\'s other '
            'artifacts.</div></div>')


# --------------------------------------------------------------------------- #
#  Reporting-frequency awareness for the TTM helpers  (reviewer R-N2)           #
# --------------------------------------------------------------------------- #
# A "trailing twelve months" sum is ONE YEAR of rows = rows_per_year rows: 4 for a
# quarterly filer, 2 for a semi-annual one. These helpers used to hard-code head(4),
# which for a semi-annual filer summed TWENTY-FOUR months -- then compared it against
# 12-month bars (Sloan's 0.10) and printed it as "TTM". Classification comes from the
# pipeline's own `reporting_period` module (period column when present, date-cadence
# fallback on saved data) so the presentation and the scorer agree on who is semi-annual.
# Unknown -> 4 (quarterly), so quarterly filers and any un-mapped source are BIT-IDENTICAL.
_RPY_BY_SOURCE = {}
_RPY_MAP_STATUS = 'not-built'   # 'ok' | 'not-built' | 'degraded: <reason>'


def build_rpy_map(cdx_df):
    """Populate the module-level {source: rows_per_year} map from a cdx panel. Idempotent.

    NEVER fails silently (domain N9): the callers sit inside broad try/except blocks, so a
    degraded map would quietly return every semi-annual filer to the 24-month TTM basis this
    layer exists to fix -- with nothing visible on the page. On degradation we record it in
    _RPY_MAP_STATUS, banner on stderr+stdout, and `rpy_basis_banner()` renders a page-level
    warning so the CEO can never read a page whose basis silently regressed."""
    global _RPY_BY_SOURCE, _RPY_MAP_STATUS
    try:
        import reporting_period as rp
        m = rp.rows_per_year_by_source(cdx_df) or {}
        if not m:
            raise RuntimeError('classification returned no sources')
        _RPY_BY_SOURCE, _RPY_MAP_STATUS = m, 'ok'
    except Exception as e:
        _RPY_BY_SOURCE = {}
        _RPY_MAP_STATUS = f'degraded: {type(e).__name__}: {e}'
        banner = ("\n" + "!" * 78 + "\n"
                  "!!! REPORTING-FREQUENCY MAP UNAVAILABLE -- TTM BASIS DEGRADED !!!\n"
                  "!!! Every source is being treated as QUARTERLY, so a SEMI-ANNUAL filer's  !!!\n"
                  "!!! 'TTM' sums TWENTY-FOUR months again (Sloan/FFO/P-FFO affected) and is !!!\n"
                  "!!! compared against 12-month bars. The page carries a visible banner.    !!!\n"
                  f"!!! Cause: {_RPY_MAP_STATUS}\n"
                  + "!" * 78 + "\n")
        print(banner, file=sys.stderr, flush=True)
        print(banner, flush=True)
    return _RPY_BY_SOURCE


def reporting_frequency(source, cdx_df=None):
    """(label, is_inferred) for one source's REPORTING FREQUENCY -- CEO-facing.

    PREFERS the pipeline's stamped ingest verdict (reporting_period.FREQ_COLUMN, written once at
    ingest = the single source of truth). Falls back to the SAME cadence classification the
    classifier uses when that column is absent (e.g. the saved pickles, which predate the
    stamp), and reports is_inferred=True so the page can say so rather than assert it."""
    label_of = {2: 'semi-annual (H1/H2)', 4: 'quarterly'}
    try:
        import reporting_period as rp
        col = rp.FREQ_COLUMN
        if cdx_df is not None and col in getattr(cdx_df, 'columns', []):
            vals = {str(v).strip().lower() for v in cdx_df.loc[cdx_df['source'] == source, col]
                    if str(v).strip() and str(v).strip().lower() != 'nan'}
            if len(vals) == 1:
                v = vals.pop()
                if v == rp.SEMIANNUAL:
                    return 'semi-annual (H1/H2)', False
                if v == rp.QUARTERLY:
                    return 'quarterly', False
                return f'{v} (as stamped)', False
    except Exception:
        pass
    rpy = rpy_for_source(source)                     # cadence classification (same logic)
    return label_of.get(rpy, 'quarterly'), True


def rpy_basis_banner():
    """Page-level HTML banner when the filing-frequency basis is degraded/unbuilt; '' when ok."""
    if _RPY_MAP_STATUS == 'ok':
        return ''
    return ('<div class="basis-warning"><strong>⚠ REPORTING-BASIS WARNING —</strong> the '
            'filing-frequency classification is unavailable, so every company is treated as a '
            'quarterly filer. For any SEMI-ANNUAL filer the "TTM" figures on this page sum '
            'TWENTY-FOUR months (Sloan accruals, FFO/share, P/FFO) and are compared against '
            'twelve-month bars, so those numbers and their 🚩/verdict icons are NOT reliable '
            f'for such names. Status: {escape(_RPY_MAP_STATUS)}.</div>')


def industry_counter_banner(postrank_df, general_top20, industrydic, cdx_df=None):
    """The INDUSTRY COUNTER block for the top-100 and top-20 (CEO, 2026-08-04); '' if unusable.

    The 07-17 corrected top-100 held 11 Marine Shipping (7 of the top-20) and nothing on this
    page said so.  The CEO reads the deck by hand, so composition has to be VISIBLE here --
    which also means it must be visible BEFORE the first name page, since an 11-of-100
    concentration changes how every deep-dive below is read.

    INFORMATIONAL ONLY -- no name is dropped, reordered or re-scored by anything here, and the
    counts are not read back by any other section.  The counting itself is delegated to the
    repo-root `industry_concentration` module, the SAME one the pipeline's run log uses, so the
    deck and the run log can never disagree about how many shipping names are in the list.

    `general_top20` is passed in rather than recomputed so the count is of the twenty names this
    deck actually renders (banded General head-20, or the flat fallback).
    """
    try:
        import industry_concentration as ic
    except Exception as _e:                                     # pragma: no cover
        log.info(f"industry counter skipped: {type(_e).__name__}: {_e}")
        return ''
    if postrank_df is None or getattr(postrank_df, 'empty', True):
        return ''
    ind = dict(industrydic) if isinstance(industrydic, dict) else {}
    if not ind:
        return ('<div class="industry-counter"><strong>INDUSTRY COUNTER —</strong> '
                'unavailable: no industry map was loaded for this run, so no name on this page '
                'can be industry-labelled.</div>')
    uni = None
    if cdx_df is not None and 'source' in getattr(cdx_df, 'columns', []):
        uni = sorted(set(cdx_df['source']))

    lists = [('Top-100', list(postrank_df['source'].head(100))),
             ('Top-20', list(general_top20))]
    blocks = []
    for label, srcs in lists:
        named, n_unc, n_tot = ic.industry_counts(srcs, ind)
        if not n_tot:
            continue
        cells = ''.join(
            '<span class="ic-cell%s">%s <b>%d</b> <span class="ic-pct">(%.0f%%)</span></span>'
            % (' ic-hot' if k / n_tot >= 0.10 else '', escape(str(name)), k,
               100.0 * k / n_tot)
            for name, k in named)
        blocks.append(
            '<div class="ic-list"><span class="ic-head">%s (%d names)</span>%s'
            '<span class="ic-cell ic-unc">unclassified <b>%d</b></span></div>'
            % (escape(label), n_tot, cells, n_unc))
    if not blocks:
        return ''
    detail = ''
    if uni and len(list(general_top20)):
        # The interpretive view: a count means nothing without the universe base rate (3 of 20
        # in a 15%-of-universe industry is nothing; 3 of 20 in a 0.8% one is 19x).  Same
        # function the run log prints, rendered as preformatted text so the numbers line up.
        try:
            detail = ('<pre class="ic-detail">'
                      + escape(ic.concentration_line(list(general_top20), uni, ind=ind))
                      + '</pre>')
        except Exception as _e:                                 # pragma: no cover
            log.info(f"industry concentration detail skipped: {type(_e).__name__}: {_e}")
    return ('<div class="industry-counter"><strong>INDUSTRY COUNTER —</strong> '
            'count of names per FMP <em>industry</em>, most-concentrated first. '
            'Informational only: nothing on this page is filtered, ranked or scored by it. '
            '<span class="ic-pct">(≥10% of a list is highlighted; unclassified is a data gap, '
            'not an industry.)</span>'
            + ''.join(blocks) + detail + '</div>')


def schema_note_banner(aggscore_df):
    """Page-level banner when the run's AggScore CSV declares a REDUCED schema; '' otherwise.

    Added 2026-07-30.  `baseline_tools/emit_deck_inputs.py` can build the deck's inputs fully
    offline, which necessarily omits the API-sourced columns, and it stamps a `_SCHEMA_NOTE` on
    every row to say so.  That note reached the CSV but NEVER the HTML, so a reader of the deck
    could not tell an offline-reduced run from a full pipeline run -- the gaps looked like facts
    about the companies rather than absences in the input.  Same treatment as the
    reporting-basis warning, because it is the same class of problem: the reader must be told
    what the page cannot know.
    """
    if aggscore_df is None or getattr(aggscore_df, 'empty', True):
        return ''
    if '_SCHEMA_NOTE' not in getattr(aggscore_df, 'columns', []):
        return ''
    notes = [str(x) for x in aggscore_df['_SCHEMA_NOTE'].dropna().unique() if str(x).strip()]
    if not notes:
        return ''
    return ('<div class="basis-warning"><strong>⚠ REDUCED-SCHEMA RUN —</strong> this deck was '
            'built from OFFLINE inputs, not from a full pipeline run. ' + escape(notes[0])
            + '</div>')


def rpy_for_source(source):
    """rows_per_year for a source name; 4 (quarterly) when unknown."""
    v = _RPY_BY_SOURCE.get(source, 4)
    try:
        v = int(v)
    except (TypeError, ValueError):
        return 4
    return v if v in (2, 4) else 4


_RPY_BLIND_DEFAULTS = 0     # calls that had to guess quarterly (see below)


def _rpy_from_frame(df):
    """rows_per_year inferred from a per-source frame's own 'source' column.

    NEVER SILENT when it cannot tell (the window-anchoring class: the moat window, the H2
    markers and the extended-pool reducer were each wrong once because a frequency default
    was applied invisibly). A frame with no usable 'source' column -- e.g. a column-selected
    groupby frame -- cannot be classified, so we must fall back to quarterly; that fallback is
    COUNTED and warned about once, so a future call site that reintroduces the bug shows up in
    the log instead of quietly summing 24 months for semi-annual filers. Call sites that cannot
    supply a 'source' column MUST pass `rpy` explicitly."""
    global _RPY_BLIND_DEFAULTS
    if df is None or not hasattr(df, 'columns') or 'source' not in df.columns or not len(df):
        _RPY_BLIND_DEFAULTS += 1
        if _RPY_BLIND_DEFAULTS == 1:
            log.warning("reporting-frequency BLIND DEFAULT: a TTM helper received a frame with "
                        "no 'source' column and no explicit rpy -- assuming quarterly. A "
                        "semi-annual filer would silently get a 24-month window here. Pass rpy "
                        "explicitly at that call site.")
        return 4
    s = df['source'].dropna()
    if not len(s):
        _RPY_BLIND_DEFAULTS += 1
        return 4
    return rpy_for_source(s.iloc[0])


# =========================================================================== #
#  THE TTM WINDOW IS POSITIONAL, AND IT ABSTAINS  (fix 2026-08-14)             #
# =========================================================================== #
#  WHAT WAS WRONG.  Both helpers below were `df[col].dropna().head(rpy)`, which is a window
#  over the newest rpy PRESENT VALUES rather than over the newest rpy PERIODS.  That makes
#  "trailing twelve months" a claim the number does not own, in two independent ways:
#
#    LIMB 1 -- FEWER THAN rpy VALUES.  The only guard was `len(values) == 0`, so a column
#      with one populated period returned that ONE period's figure, labelled TTM, with
#      nothing in the return value to say so.  A quarter presented as a year.
#    LIMB 2 -- REACHING PAST THE WINDOW.  With a hole inside the newest rpy rows, `dropna`
#      walks OLDER to find its rpy-th value, so the sum spans more than twelve months while
#      still being divided into, and compared against, twelve-month quantities.
#
#  MEASURED ON THE SHIPPED 2026-08-13 CUR3K PANEL, and the measurement corrects the reason
#  this fix was requested rather than confirming it:
#    * over all 15,774 (source, column) pairs these helpers read, LIMB 1 fires **0 times** --
#      every column carries either at least rpy present periods or none at all.  So the limb
#      described as the live defect is LATENT on this data, not active.  It is still fixed:
#      "latent" is a property of one panel, not of the function.
#    * LIMB 2 fires on 21 of 15,774 pairs (0.133%), median span exactly 1.00 years and worst
#      1.50 -- and on 0 of the 600 shortlist pairs.
#    * driving the REAL presentation build for 2026-08-13 through all 15 exercised call
#      sites (22,516 invocations), this change alters 10 of them, at four sites (the FCF
#      margin, the Sloan recomputation and the two REIT reducers), every one from a number
#      to an abstention, and every one of the 10 is a LIMB-2 reach.  None is a top-100 name;
#      they are extended-pool members.
#
#  WHY ABSTAIN RATHER THAN SUM WHAT IS THERE.  These figures feed ratios and per-share
#  quantities that are compared against annual benchmarks (a "> 4x" leverage bar, a P/FFO, an
#  AFFO payout).  A sum over a different span is not a smaller measurement of the same thing,
#  it is a different quantity wearing the same units -- and the consumer has no way to tell.
#  NaN is a state every consumer here already handles.
def ttm_sum(df, col, rpy=None):
    """Trailing-TWELVE-MONTH sum: the newest `rpy` rows (4 quarterly / 2 semi-annual) after a
    newest-first sort, or NaN.

    THE WINDOW IS POSITIONAL. `rpy` ROWS, EVERY ONE PRESENT -- see the block above. Anything
    less returns NaN rather than a sum over a different span. `rpy=None` derives it from the
    frame's own 'source' column; pass it explicitly where the frame carries no 'source'
    (e.g. the extended-pool reducer)."""
    if df is None or df.empty or col not in df.columns:
        return np.nan
    if rpy is None:
        rpy = _rpy_from_frame(df)
    rpy = int(rpy)
    window = pd.to_numeric(df[col], errors='coerce').head(rpy)
    #  Two conditions, not one: the frame may be SHORTER than the window (a newly-listed
    #  name), and the window may have a HOLE in it.  Both mean "there is no trailing year
    #  here", and neither may be answered with a partial sum.
    if len(window) < rpy or window.isna().any():
        return np.nan
    return float(np.sum(window.values))


def ttm_aligned_sums(df, cols, rpy=None):
    """Trailing-12-month sums for several columns over the SAME `rpy` newest PERIODS, or NaNs.

    Unlike calling ttm_sum per column -- which would window each column independently -- this
    keeps a paired sum (e.g. R2's netIncome vs operating cash flow) on one consistent period
    set. Returns a tuple of floats (np.nan where unavailable), in `cols` order.

    THE ALIGNMENT IS NOW GUARANTEED BY CONSTRUCTION rather than by construction-and-hope: the
    window is the newest `rpy` ROWS and EVERY listed column must be present in ALL of them, so
    the returned sums cover the same twelve months as each other AND as `ttm_sum`'s. The old
    `df[cols].dropna().head(rpy)` kept the columns aligned WITH EACH OTHER but let the shared
    window slide older -- so a paired ratio could be internally consistent and still not be a
    year. `df` is assumed newest-first (get_cdx_for_ticker's order); `rpy=None` -> derived from
    the frame's 'source' column (4 = quarterly when unknown)."""
    if df is None or df.empty or any(c not in df.columns for c in cols):
        return tuple(np.nan for _ in cols)
    if rpy is None:
        rpy = _rpy_from_frame(df)
    rpy = int(rpy)
    sub = df[cols].apply(pd.to_numeric, errors='coerce').head(rpy)
    if len(sub) < rpy or bool(sub.isna().to_numpy().any()):
        return tuple(np.nan for _ in cols)
    return tuple(float(sub[c].sum()) for c in cols)


def net_debt_to_ebitda_annual(df, rpy=None):
    """Net debt / ANNUAL EBITDA -- the standard multiple a "> 4x" bar is written for.

    cdx_df's netDebtToEBITDA divides a point-in-time net-debt STOCK by a single period's
    EBITDA, so the raw field reads ~4x high for a quarterly filer and ~2x for a semi-annual
    one. Stage-1 already corrects it (reporting_period.STAGE1_FLOW_CORRECTION marks it
    'flow_den'/'annualize'); the presentation did not, so it printed a per-quarter multiple
    as "x" and fired R4's "> 4x" leverage limb on ~2.4x as many names as intended
    (measured on the saved panel: raw median 3.348 / 47.1% over 4, vs annualized 0.837 /
    20.0%). We reuse the pipeline's own factor so both agree by construction."""
    raw = latest_row_value(df, 'netDebtToEBITDA')
    if np.isnan(raw):
        return np.nan
    if rpy is None:
        rpy = _rpy_from_frame(df)
    try:
        import reporting_period as rp
        f = rp.stage1_flow_factor('netDebtToEBITDA', rpy)      # = 1 / annualize_factor(rpy)
    except Exception:
        f = 1.0 / float(int(rpy) or 4)
    return raw * f


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


def compute_fcf_margin_ttm(df, rpy=None):
    """Compute TTM FCF margin (TTM FCF / TTM revenue). RATIO-CANCELLING w.r.t. the window
    (both sums span the same periods), but the window is still made a true 12 months."""
    if df is None or df.empty:
        return np.nan
    fcf_ttm = ttm_sum(df, 'freeCashFlow', rpy)
    rev_ttm = ttm_sum(df, 'revenue', rpy)
    if np.isnan(fcf_ttm) or np.isnan(rev_ttm) or rev_ttm == 0:
        return np.nan
    return fcf_ttm / rev_ttm


def compute_cash_conversion(df, rpy=None):
    """Compute cash conversion (TTM FCF / TTM netIncome). RATIO-CANCELLING w.r.t. the window."""
    if df is None or df.empty:
        return np.nan
    fcf_ttm = ttm_sum(df, 'freeCashFlow', rpy)
    ni_ttm = ttm_sum(df, 'netIncome', rpy)
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


def compute_ffo_per_share(df, rpy=None):
    """REIT FFO per share (proxy): (netIncome + D&A) / weightedAverageShsOut (TTM).
    A LEVEL (not ratio-cancelling): it flows into P/FFO against a point-in-time market cap,
    so a 24-month sum would understate P/FFO for a semi-annual filer."""
    if df is None or df.empty:
        return np.nan
    ni_ttm = ttm_sum(df, 'netIncome', rpy)
    da_ttm = ttm_sum(df, 'depreciationAndAmortization', rpy)
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
    # RANGE-FREE on purpose (domain S7): the old 'emp -0.47...0.40' band was measured on the
    # GENERAL pool only, while cohort pages score on per-cohort weight vectors whose scale
    # differs (cohort AggScores ran ~+5 pre-normalisation), so any fixed band asserts a false
    # ceiling on one pool or the other. Comparable WITHIN a pool; not across pools.
    'AggScore':   ('unbounded · pool-relative', '↑ better',
                   'Stage-2 weighted sum of z-scored metrics. Scale depends on the weight '
                   'vector of the pool, so compare AggScore only against names in the SAME pool '
                   '(general vs a carve cohort), never across pools or against a fixed band.'),
    'moatScore':  ('0–11 (emp 1–7 typical, median 3)', '↑ better',
                   'count of 11 quality boxes ticked; range is the POST-ANNUALIZATION scale '
                   '(5 of the 11 components compared a quarterly ratio against an annual bar '
                   'and were structurally near-unpassable until the 2026-07-25 fix, so scores '
                   'from runs before that fetch sit lower — max now reaches 10–11)'),
    # -- Forensic (Sections F / G) --
    'M-Score':      ('stored Beneish+1.78', '↓ better (>0 flags)',
                     'earnings-manipulation index'),
    'C-Score':      ('0–5', '↓ better (≥4 of 5 flags, 0 clean)',
                     'count of 5 Montier accounting red flags. The sixth flag -- the '
                     'depreciation-rate test -- was REMOVED from the scored set entirely: it '
                     'needs GROSS property/plant/equipment as its denominator, a field FMP does '
                     'not carry on the standard balance sheet, so it fired by construction. '
                     'The ≥4 cutoff is UNCHANGED and is therefore '
                     'now ≥4 of 5 -- deliberately stricter than published Montier, because on '
                     'an advisory flag a false positive costs more than a false negative.'),
    'sloanAccruals':('ratio', '↓ better (low/neg = cash-backed)',
                     'balance-sheet accruals; pipeline flag = worst quintile in-run, not absolute'),
    # BASIS CHANGED 2026-08-01 (audit D2): this is no longer FMP's CFO/NI ratio.  It is
    # (CFO − netIncome) / totalAssets, averaged over the scoring window — an ACCRUALS
    # measure on an assets scale, so typical values are ~±0.05, NOT ~1.  The old
    # "≈1 neutral (emp −13.8…8.0)" text described the ratio and would now be actively
    # misleading.  Measured on the shipped top-100 under the new basis: −0.050…+0.043.
    'incomeQuality':('accruals/assets (emp −0.05…+0.04)', '↑ better (cash > earnings)',
                     '(CFO−NI)/totalAssets; sign-safe replacement for the CFO/NI ratio, '
                     'which inverted for loss-makers and exploded as NI→0'),
    # Soundness audit (2026-07-19, re-derived 2026-07-26 on the CURRENT code): the computed
    # variant is NOT published Altman-Z. The quarter-scale flow half of the original finding is
    # FIXED; what remains is that the statistic is ~ONE TERM -- 0.6*x4 (market cap / total
    # liabilities) is 66.5% of the score with corr 0.997 to Z -- plus the wrong x2 variable.
    # A near-single-ratio leverage measure, so the published 1.8/3.0 cutoffs do not apply.
    'Altman-Z':     ('relative-only (uncalibrated)', '↑ better',
                     'computed variant is NOT published Altman-Z: the 0.6*x4 term '
                     '(market cap / total liabilities) alone is ~66.5% of the score and '
                     'correlates 0.997 with it, so this behaves as essentially ONE '
                     'leverage term, not a five-factor Z. The 1.8/3.0 cutoffs do not '
                     'apply to it — treat as relative-only pending a fix'),
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
    # 'Altman-Z' DELIBERATELY ABSENT (soundness audit; re-derived 2026-07-26): the computed
    # variant is dominated by ONE term -- 0.6*x4 (market cap / total liabilities) is 66.5% of
    # the score, corr 0.997 -- plus the wrong x2 variable, so it is not the published
    # five-factor statistic and 1.8/3.0 are not calibrated for it. No tick is drawn on its peer
    # bar and no verdict is derived from it -- the bar shows the PEER DISTRIBUTION ONLY
    # (relative-only). Do not restore a tick/threshold here without a recalibration decision.
    'Piotroski': dict(tick=3, warn=('<=', 3), good=('>=', 7),
                         note='rule-of-thumb: ≤3 weak · ≥7 strong'),
    # No peer bar for these -> inline verdict chip only (tick=None).
    'M-Score': dict(tick=None, warn=('>', 0.0),
                         note='rule-of-thumb: stored >0 flags (Beneish>−1.78); invalid financials',
                         suppress=_FIN),
    'C-Score': dict(tick=None, warn=('>=', 4),
                         note='rule-of-thumb: ≥4 of 5 flags fired · 0 clean; stricter than published Montier by design; invalid financials',
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
    # Altman-Z DELIBERATELY ABSENT (soundness audit; re-derived 2026-07-26): the computed
    # variant is ~one term (0.6*x4 = mcap/total-liabilities, 66.5% of the score, corr 0.997),
    # not the published five-factor Z, so 1.8/3.0 are not calibrated for it -> it is listed
    # in VERDICT_GRAY (⚪, relative-only) until the pipeline metric is fixed. Do not restore
    # a threshold verdict here without a recalibration decision.
    'Piotroski':               dict(good='high', green=7,    red=4,    note='rule-of-thumb: ≥7 strong · ≤3 weak · 4–6 middling'),
    'M-Score':                 dict(good='low',  green=-0.5, red=0.5,  note='rule-of-thumb: stored ≤0 clean · >0 flag · −0.5…+0.5 borderline (invalid financials)', suppress=_FIN),
    'C-Score':                 dict(good='low',  green=0.5,  red=3.5,  note='rule-of-thumb (out of 5 flags): 0 clean · ≥4 of 5 flag · 1–3 borderline; ≥4-of-5 is stricter than published Montier by design (invalid financials)', suppress=_FIN),
    # 'incomeQuality': GRAY BY AUDIT DECISION (D2, 2026-08-01) -- deliberately no verdict.
    # It carried green=0.9 / red=0.7 ("≥0.9 cash-backed · <0.7 flag"), which are RATIO-scale
    # thresholds for FMP's CFO/NI.  The metric is now (CFO−NI)/totalAssets, whose values run
    # ~±0.05, so EVERY name would fall below red=0.7 and the whole shortlist would render as a
    # red flag.  Re-deriving calibrated bands for the new scale is a fitting exercise and is
    # NOT authorised, so the honest move is the one Altman-Z already takes: publish the number,
    # withhold the verdict, and say why -- rather than invent thresholds or leave miscalibrated
    # ones firing.  The orientation chip (↑ better) still applies and is unaffected.
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
    # 'incomeQuality': gray by the SAME audit decision (D2, 2026-08-01).  Its old green=0.9 /
    # red=0.7 bands were for FMP's CFO/NI ratio; the metric is now (CFO−NI)/totalAssets with
    # values ~±0.05, so those bands would flag every name.  Listed HERE rather than merely
    # deleted from VERDICT_RULES, so it renders as an explicit ⚪ "no honest standalone rule"
    # instead of silently falling through to whatever the default is.
    'incomeQuality',
    'peRatio', 'pbRatio', 'bVpRatio', 'tbVpRatio', 'grossProfitMargin', 'op_margin',
    'effective_tax', 'dso', 'inv_days', 'returnOnEquity', 'RoA', 'revenueGrowth',
    'freeCashFlowPerShareGrowth', 'net_debt_ebitda', 'CycleHeat', 'AggScore', 'moatScore',
    'p_ffo', 'ffo_per_share', 'ltv', 'nav',
}
# Moat comparators are stored as (value − threshold) [or (threshold − value) for the four
# "lower is better" ones], so 0 IS the bar. The 🟡 "sitting on its bar" band must therefore be
# PROPORTIONATE TO EACH COMPARATOR'S OWN THRESHOLD -- a flat ±0.02 meant "within 20% of the
# bar" for RoA (bar 0.10) but "within 2.5%" for TLtoEquity (bar 0.8), so amber did not mean the
# same thing across the eleven (measured amber share: RoA 23.3% vs TLtoEquity 2.2%). Thresholds
# below are the pipeline's own (postBo.moatIdentifier): value − thr, or thr − value.
MOAT_THRESHOLDS = {
    'FCFyield': 0.10, 'GrossMargin': 0.30, 'RevtoASS': 0.75, 'RoE': 0.15, 'RoA': 0.10,
    'ROIC': 0.15, 'SGAtoGP': 0.15, 'DeptoGP': 0.10, 'NetMargin': 0.20,
    'CapExtoEarnings': 0.20, 'TLtoEquity': 0.80,
}
# "on its bar" = within this FRACTION of the comparator's own threshold.
MOAT_NEAR_ZERO_FRAC = 0.05
_MOAT_NEAR_ZERO = 0.02   # legacy flat fallback (only for a comparator with no known threshold)


def moat_near_zero_band(component):
    """Half-width of the 🟡 'on its bar' band for one moat comparator: a fixed FRACTION of
    that comparator's own threshold, so amber means the same thing on all eleven."""
    thr = MOAT_THRESHOLDS.get(component)
    if thr is None or not np.isfinite(thr) or thr == 0:
        return _MOAT_NEAR_ZERO
    return abs(thr) * MOAT_NEAR_ZERO_FRAC

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


#  THE LOW-CONFIDENCE NOTE ASSERTED A VALUE THAT MAY NOT EXIST (P-2 sweep, 2026-08-17).
#  `low_conf` was tested BEFORE the value, so a metric that is both low-confidence and ABSENT
#  rendered the sentence "value present but flagged low-confidence" beside an empty cell --
#  the same "explanation of a thing that is not there" class as `M_drivers` on a row with no
#  M-score.  It reaches two live combinations: a forensically-invalid name whose M/C/Sloan is
#  NaN (`low_conf_forensic`) and a weak-denominator name whose incomeQuality or
#  cash-conversion is NaN (`denom_weak`).  Only the SENTENCE was fixed then.
#
#  ABSENCE NOW OUTRANKS LOW CONFIDENCE (P-5, 2026-08-29).  The icon precedence was left in
#  place in 2026-08-17 as a documented spec decision ("spec Part C") rather than a bug, so the
#  🟡 kept rendering over the ⚪ and a MISSING metric read as a merely UNCERTAIN one.
#  That is the weaker of the two facts, and the wrong one: "we could not measure this" is
#  stronger, more actionable, and cannot be mistaken for a soft reading of a number -- while
#  a 🟡 on an empty cell qualifies a reading nobody made.  Spec Part C is amended with the
#  code, not behind it: the docstring below, the on-page legend (`_icon_legend`) and the
#  pinning test in `test_reporting_expansion.py` all state the new order.
#  THE 🚩 IS UNAFFECTED -- it is independent of the verdict icon by design (the legend says so),
#  so a low-confidence name still carries whatever suspicion flag its rule fired; what changes
#  is only which of two ⚪/🟡 facts the verdict glyph reports when BOTH are true.
#  `_low_conf_note`'s absent-value branch is now unreachable through `compute_verdict` and is
#  kept deliberately as the belt to this brace (a direct assertion in the test exercises it),
#  so a future re-ordering cannot silently restore the false sentence as well as the icon.
def _low_conf_note(value, reason=None):
    """The Y2 low-confidence note, told apart from the case where there is no value.

    `reason` names WHICH low-confidence determination fired.  It exists because the guard now
    has two genuinely different triggers and a reader cannot act on the icon without knowing
    which one it is: "the forensic models do not apply to a bank" is a statement about the
    business, "we could not determine whether they apply" is a statement about our own
    coverage (Q-44).  An unnamed 🟡 collapses them into one shrug."""
    base = ('value unavailable, and this metric is also flagged low-confidence'
            if np.isnan(safe_float(value))
            else 'value present but flagged low-confidence')
    why = str(reason).strip() if reason and not isinstance(reason, bool) else ''
    return f'{base} — {why} (see 🚩/forensic)' if why         else f'{base} (see 🚩/forensic)'


def compute_verdict(metric_key, value, cohort_label=None, low_conf=False):
    """4-state verdict for one metric. Precedence (spec Part C, as amended by P-5 2026-08-29):
    (1) no rule / suppressed for cohort -> ⚪ 'gray'; (2) else NO VALUE -> ⚪ 'gray, value
    unavailable'; (3) else Y2 low-confidence -> 🟡 'neutral'; (4) else value vs rule.
    Returns (state, note) where state in {good,neutral,warn,gray} or (None, None) when the
    metric takes NO icon. `low_conf` carries the caller's Y2 determination: falsy for none,
    or truthy to fire -- and when it is a non-empty STRING that string is the REASON, rendered
    into the note (Q-44).  A bare `True` still works and reads exactly as before.

    (2) BEFORE (3) is the P-5 ruling: absence outranks low confidence.  A missing value and an
    uncertain value are different facts, and rendering the missing one as the uncertain one
    tells the reader the weaker, less actionable of the two -- and does it in the direction
    that reads as a number we merely distrust rather than a number we do not have."""
    if metric_key in VERDICT_RULES:
        r = VERDICT_RULES[metric_key]
        if cohort_label is not None and cohort_label in r.get('suppress', set()):
            return 'gray', 'no universal rule for this cohort'
        st = _verdict_state_band(r, value)
        if st is None:
            return 'gray', 'value unavailable'
        if low_conf:
            return 'neutral', _low_conf_note(value, low_conf)
        return st, r['note']
    if metric_key in VERDICT_FLOORS:
        f = VERDICT_FLOORS[metric_key]
        if cohort_label is not None and cohort_label in f.get('suppress', set()):
            return 'gray', 'no universal rule for this cohort'
        v = safe_float(value)
        if np.isnan(v):
            return 'gray', 'value unavailable'
        if low_conf:
            return 'neutral', _low_conf_note(value, low_conf)
        bad = (v < f['floor']) if f['bad'] == 'below' else (v > f['floor'])
        return ('warn', f['note']) if bad else ('gray', 'no positive standalone rule (peer-relative)')
    if metric_key in VERDICT_GRAY:
        return 'gray', 'no honest standalone rule — read against the peer bar'
    if metric_key.startswith('moat:'):
        v = safe_float(value)
        if np.isnan(v):
            return 'gray', 'value unavailable'
        comp = metric_key.split(':', 1)[1]
        band = moat_near_zero_band(comp)                      # proportionate to its own bar
        thr = MOAT_THRESHOLDS.get(comp)
        margin = (f' (bar {thr:g}; amber within ±{MOAT_NEAR_ZERO_FRAC:.0%} of it)'
                  if thr is not None else '')
        if v > band:
            return 'good', f'moat component passes its bar (stored value−threshold > 0){margin}'
        if v < -band:
            return 'warn', f'moat component fails its bar (stored value−threshold < 0){margin}'
        return 'neutral', f'moat component sits ON its bar (stored value−threshold ≈ 0){margin}'
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


def _sloan_recomputed(g, rpy=None):
    """Sloan accruals = (NI_ttm − CFO_ttm) / AVERAGE total assets over the same 12 months.

    ONE DEFINITION (code S5): this now matches the pipeline's canonical
    forensicFlags.computeSloanAccruals exactly -- TTM flows over the AVERAGE of beginning- and
    end-of-window total assets (Sloan's original form). It previously divided by the LATEST
    total assets, so the number under the bar, the peer bar itself, the 🚩/verdict and the
    'shortlist CSV' row shown beside it were TWO different quantities sharing one name and one
    0.10 bar. Denominator averaging matters for a grower: assets rise over the window, so
    latest-TA understates accruals versus the canonical measure.

    NOT ratio-cancelling: a 12-month FLOW difference over a STOCK, compared to an absolute
    12-month bar (0.10) -- so the window must be a true year (reviewer R-N2): `rpy` rows.
    Falls back to latest total assets when there is no one-year-earlier asset level (keeps the
    name in the peer pool rather than dropping it; the pipeline returns NaN there instead)."""
    ni = ttm_sum(g, 'netIncome', rpy)
    cfo = ttm_sum(g, 'netCashProvidedByOperatingActivities', rpy)
    if np.isnan(ni) or np.isnan(cfo) or 'totalAssets' not in getattr(g, 'columns', []):
        return np.nan
    if rpy is None:
        rpy = _rpy_from_frame(g)
    ta_series = pd.to_numeric(g['totalAssets'], errors='coerce').dropna()   # newest-first
    if ta_series.empty:
        return np.nan
    ta_end = float(ta_series.iloc[0])
    ta_begin = float(ta_series.iloc[int(rpy)]) if len(ta_series) > int(rpy) else np.nan
    avg_ta = (ta_end + ta_begin) / 2.0 if not np.isnan(ta_begin) else ta_end
    if np.isnan(avg_ta) or avg_ta == 0:
        return np.nan
    return (ni - cfo) / avg_ta


def _p_ffo_reducer(g, rpy=None):
    ffo_ps = compute_ffo_per_share(g, rpy)
    shares = latest_row_value(g, 'weightedAverageShsOut')
    mktcap = latest_row_value(g, 'marketCap')
    if np.isnan(ffo_ps) or np.isnan(shares) or np.isnan(mktcap) or ffo_ps * shares == 0:
        return np.nan
    return mktcap / (ffo_ps * shares)


def _affo_payout_reducer(g, rpy=None):
    # RATIO-CANCELLING (12-month dividends over a 12-month AFFO), but windowed correctly too.
    div = abs(ttm_sum(g, 'dividendsPaid', rpy))
    ffo = ttm_sum(g, 'netIncome', rpy) + ttm_sum(g, 'depreciationAndAmortization', rpy)
    capex = ttm_sum(g, 'netCashProvidedByOperatingActivities', rpy) - ttm_sum(g, 'freeCashFlow', rpy)
    denom = ffo - capex
    if np.isnan(div) or np.isnan(denom) or denom <= 0:
        return np.nan
    return div / denom


def _ext_reducer(g, rpy=None):
    """Per-source extended metrics, using the SAME helpers as the per-name markers so the
    pool value and the displayed marker are identical.

    `rpy` MUST be passed here: the group frame this receives is column-selected and carries
    NO 'source' column, so the frame-derived fallback cannot see the filing frequency and
    would silently treat a semi-annual filer as quarterly (reviewer R-N2)."""
    return pd.Series({
        'op_margin': compute_operating_margin(g),
        'fcf_margin': compute_fcf_margin_ttm(g, rpy),
        'interest_coverage': compute_interest_coverage(g),
        # ANNUAL multiple (see net_debt_to_ebitda_annual): the pool, the marker, the displayed
        # number and the "> 4x" triggers must all be the same annual quantity.
        'net_debt_ebitda': net_debt_to_ebitda_annual(g, rpy),
        'effective_tax': _clip01(latest_row_value(g, 'effectiveTaxRate')),
        'dso': latest_row_value(g, 'daysSalesOutstanding'),
        'inv_days': latest_row_value(g, 'daysOfInventoryOutstanding'),
        'sloan': _sloan_recomputed(g, rpy),
        'p_ffo': _p_ffo_reducer(g, rpy),
        'affo_payout': _affo_payout_reducer(g, rpy),
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
    #  Reconstruct the floor THE WAY THE PIPELINE APPLIES IT, or not at all.
    #  This used to threshold the RAW `marketCap` column, which is in each company's
    #  REPORTING currency, mixed across the universe -- so the deck's reconstructed
    #  membership never matched carveOut's (a SEK reporter cleared a $25M floor at ~$2.4M).
    #  carveOut now applies the floor only where reportedCurrency really resolves and KEEPS
    #  every unknown-currency name (gating, 2026-08-06). Mirror both halves: use the same
    #  single conversion path, and when the currency has not flowed (`floor_enforced` False)
    #  apply NO floor here either, because the pipeline applied none.
    floor_enforced = diag.get('floor_enforced')
    if floor_enforced is None:                      # artifact predates the flag
        try:
            import carveOut as _co
            floor_enforced = _co.currency_data_present(cdx)
        except Exception:
            floor_enforced = False
    mcap_usd = {}
    if floor_enforced and floor:
        try:
            import carveOut as _co
            mcap_usd = _co.marketcap_usd_by_source(cdx)
        except Exception:
            floor_enforced = False
    mem = {}
    for coh in ['general'] + COHORTS:
        names = labels[labels == coh].index.tolist()
        if not (floor_enforced and floor):
            mem[coh] = list(names)
            continue
        #  unknown USD cap -> KEPT, exactly as carveOut keeps it.
        mem[coh] = [n for n in names
                    if not (pd.notna(mcap_usd.get(n, float('nan')))
                            and mcap_usd.get(n) < floor)]
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
        build_rpy_map(cdx)          # filing-frequency map for the TTM helpers (R-N2)
        cols = [c for c in _EXT_CDX_COLS if c in cdx.columns] + ['source']
        c = cdx[cols].copy()
        c['date'] = pd.to_datetime(c['date'], errors='coerce')
        c = c.sort_values(['source', 'date'], ascending=[True, False])
        all_members = set()
        for lst in membership.values():
            all_members |= set(lst)
        sub = c[c['source'].isin(all_members)]
        value_cols = [x for x in cols if x != 'source']
        # Explicit per-source loop (not .apply): the reducer needs each source's rows_per_year
        # and the column-selected group frame carries no 'source' column to derive it from.
        _rows = {src: _ext_reducer(grp[value_cols], rpy_for_source(src))
                 for src, grp in sub.groupby('source', sort=False)}
        per = (pd.DataFrame.from_dict(_rows, orient='index') if _rows
               else pd.DataFrame(columns=[
                   'op_margin', 'fcf_margin', 'interest_coverage', 'net_debt_ebitda',
                   'effective_tax', 'dso', 'inv_days', 'sloan', 'p_ffo', 'affo_payout', 'ltv']))
        per.index.name = 'source'

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
    #  The per-pool Stage-1 veto report (register N-5).  Already in every Boresults pickle --
    #  it was simply never read here, so the deck said nothing about the largest edit the
    #  pipeline makes to the pool it ranks.  `None` (key absent, i.e. a pre-veto pickle) and
    #  `{}` (the guarded block raised, so the pools are UN-VETOED) mean different things and
    #  are both passed through unchanged for `veto_scope_banner` to distinguish.
    stage1_veto = boresults_dic.get('stage1_veto')

    # Load CSVs
    aggscore_df = None
    forensic_df = None
    if aggscore_file and os.path.exists(aggscore_file):
        aggscore_df = pd.read_csv(aggscore_file)
    if forensic_file and os.path.exists(forensic_file):
        forensic_df = pd.read_csv(forensic_file)

    # ------------------------------------------------------------------------------ #
    #  THE OFF-CSV FALLBACK'S INPUTS (reviewer H-1) -- all SAME-DATE, all optional.     #
    # ------------------------------------------------------------------------------ #
    #  SAME-DATE OR NOTHING, matching `resolve_run_artifacts`: mixing a 08-13 shortlist with
    #  a 08-11 imputation table or a 08-11 FX quote is the cross-run mixing that function
    #  exists to refuse, and it would be undetectable here because every value would still
    #  look plausible.  Each of these is OPTIONAL -- a run that predates it simply gets the
    #  em-dash it gets today.
    def _same_date(*names):
        for n in names:
            f = os.path.join(str(run_dir), n.format(d=run_date))
            if os.path.exists(f):
                return f
        return None

    #  THE PROFILE CAPTURE MUST COME FROM **THIS RUN**, EXPLICITLY.
    #  carveOut's cached default globs the REPO ROOT for the newest capture, which is right
    #  for the pipeline (it runs in the repo root and writes the run it is reading) and WRONG
    #  here: the deck is handed an arbitrary `--run-dir`, so the default would silently pair a
    #  2026-08-13 shortlist with whatever capture happens to sit in the repo.  Caught in
    #  testing exactly that way -- the deck rendered `price_asof 2026-08-11` on an 08-13 run,
    #  blanked CBSM.PA (absent from the older map) and gave ORIA.PA $7,337/day against this
    #  run's $5,915.
    #  NO CROSS-RUN FALLBACK.  If this run has no capture, the fields stay blank: a price and
    #  a traded value from a DIFFERENT day are worse than an em-dash, and `resolve_run_artifacts`
    #  refuses the same substitution for the same reason.
    #  ASKED FOR BY RUN, NOT BY FILENAME.  `carveOut.profile_map_for_run` owns the naming
    #  convention because `test_universes.test_the_volavg_pickle_still_has_exactly_ONE_reading_seam`
    #  pins the set of modules allowed to NAME that artifact at all -- a fourth module knowing
    #  the filename is one step from doing its own `read_pickle` and reading raw entries,
    #  which is how the single-seam guarantee lapses without anyone noticing.  That guard
    #  caught THIS file when the fix was first written the other way round, so the filename
    #  is deliberately absent from this module -- including from these comments.
    profile_map = {}
    try:
        sys.path.insert(0, str(valuation_repo))
        import carveOut as _co_pm
        profile_map = _co_pm.profile_map_for_run(run_dir, run_date)
    except Exception as _e:
        log.info(f'profile map unavailable: {type(_e).__name__}: {_e}')
    if profile_map:
        log.info(f"    profile capture: {len(profile_map)} entries for {run_date}")
    else:
        log.info(f'    profile capture: NONE for {run_date} in the run dir -- price, traded '
                 f'value and trading currency stay blank for names outside the AggScore CSV. '
                 f"Deliberately NOT falling back to another run's capture.")

    fill_by_name = {}
    _fill_f = _same_date('MissingDataFillReport_{d}.csv')
    if _fill_f:
        try:
            _fdf = pd.read_csv(_fill_f)
            _fdf = _fdf[_fdf['section'] == 'per_name']
            #  A source appears under exactly ONE pool (verified on 2026-08-13: 0 of 222
            #  sources carry more than one per_name row), so a flat map is unambiguous --
            #  which matters, because the cohort names this fallback serves are in their
            #  cohort's rows, not in `general`.
            fill_by_name = {r['source']: r['imputed_weight_share']
                            for _, r in _fdf.iterrows() if isinstance(r['source'], str)}
            log.info(f"    MissingDataFill: {os.path.basename(_fill_f)} "
                     f"({len(fill_by_name)} names)")
        except Exception as _e:
            log.info(f'missing-data fill map unavailable: {type(_e).__name__}: {_e}')

    clone_map = {}
    _cont_f = _same_date('VendorContaminationFlags_{d}.csv',
                         os.path.join('output', 'VendorContaminationFlags_{d}.csv'))
    if _cont_f:
        try:
            sys.path.insert(0, str(valuation_repo))
            import vendor_contamination as _vc
            clone_map = _vc.clone_counterparts(path=_cont_f)
            log.info(f"    Contamination: {os.path.basename(_cont_f)} "
                     f"({len(clone_map)} paired sources)")
        except Exception as _e:
            log.info(f'clone map unavailable: {type(_e).__name__}: {_e}')

    #  THE RUN'S OWN FX, NOT THE SANITY ANCHORS.  The deck never calls
    #  `fx_rates.install_for_run`, so `carveOut`'s module FX state is 'unset' here and an
    #  unqualified conversion would silently use the hardcoded anchors -- which are a UNITS
    #  check, not a rate source, and were measured ~7% off live (SHEL.L: $391.9M on anchors
    #  vs $416.6M live).  Worse, the two produce IDENTICAL basis strings.  So: prefer the
    #  run's dated table; if it is absent, still convert (a rough number beats no number for
    #  a display column) but LABEL every cell `fx=anchors` so the reader can see it.
    fx_table, fx_label = None, None
    _fx_f = _same_date(os.path.join('output', 'FxRates_{d}.csv'), 'FxRates_{d}.csv')
    if _fx_f:
        try:
            _fx = pd.read_csv(_fx_f)
            fx_table = {r.currency: float(r.rate) for r in _fx.itertuples()
                        if getattr(r, 'usable', True) and str(r.currency).strip()}
            log.info(f"    FxRates: {os.path.basename(_fx_f)} ({len(fx_table)} usable)")
        except Exception as _e:
            log.info(f'run FX table unavailable: {type(_e).__name__}: {_e}')
    if not fx_table:
        try:
            sys.path.insert(0, str(valuation_repo))
            import carveOut as _co_fx
            fx_table = dict(_co_fx.FX_TO_USD)
            fx_label = 'fx=anchors'
            log.info('    FxRates: NOT FOUND for this run date -- traded value is converted '
                     'with the sanity ANCHORS and every cell is labelled `fx=anchors`.')
        except Exception:
            fx_table = None

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
        'stage1_veto': stage1_veto,
        #  Inputs to the off-CSV fallback (reviewer H-1); each may be empty/None.
        'fill_by_name': fill_by_name,
        'clone_map': clone_map,
        'profile_map': profile_map,
        'fx_table': fx_table,
        'fx_label': fx_label,
        'aggscore_df': aggscore_df,
        'forensic_df': forensic_df,
        'get_industry': get_industry,
        # The raw map as well as the closure: the industry COUNTER needs to aggregate over a
        # whole list, and re-globbing the pickle for it would let the counter and the per-name
        # labels drift onto two different files.
        'industrydic': industrydic if isinstance(industrydic, dict) else {},
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
        build_rpy_map(cdx_df)       # filing-frequency map for the TTM helpers (R-N2)
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

        def _annualize_display(df):
            """Per-quarter -> ANNUAL for the flow/stock metrics, for DISPLAY (see
            ANNUALIZED_DISPLAY_METRICS). Uniform scale => percentiles/dot positions invariant."""
            if df is None or not hasattr(df, 'columns'):
                return df
            for c in ANNUALIZED_DISPLAY_METRICS:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce') * ANNUALIZE_DISPLAY_FACTOR
            return df

        # Annualize BEFORE the p10/p50/p90 stats are derived, so the spread, the percentile
        # and the marker all sit on the one annual scale.
        for _lbl in list(dist_pools):
            dist_pools[_lbl] = _annualize_display(dist_pools[_lbl])

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
        # SAME annualization as the pools above -- raw_all feeds the markers, raw_metric(),
        # Section C's displayed values and the verdict/flag rule inputs, so it must match.
        raw_all = _annualize_display(raw_all)
        if 'source' in raw_all.columns:
            raw_all = raw_all.set_index('source')

        # Newest-first cdx per ticker, for RAW latest markers.
        cdx_sorted = cdx_df.copy()
        cdx_sorted['date'] = pd.to_datetime(cdx_sorted['date'], errors='coerce')
        cdx_sorted = cdx_sorted.sort_values('date', ascending=False)
        cdx_by_ticker = {t: g for t, g in cdx_sorted.groupby('source', sort=False)}

        def latest_marker(ticker, cdx_col, pool_metric):
            """Raw marker: cdx latest if the metric has a native cdx column, else the
            reviewReference raw-pool value (e.g. freeCashFlowYield).

            EXCEPT the POOL_BASIS_MARKERS four (reviewer H2): those are period-corrected in
            the pool, so their marker is read from `raw_all` -- the same _pool_raw_fast output
            the pool distribution is built from -- so marker and pool share one basis."""
            if pool_metric in POOL_BASIS_MARKERS:
                if ticker in raw_all.index and pool_metric in raw_all.columns:
                    return safe_float(raw_all.loc[ticker, pool_metric])
                return np.nan
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
        #  OFF-CSV FALLBACK for the reporting-expansion fields (reviewer H-1).  Resolved
        #  lazily per ticker, because the AggScore CSV covers only the general top-100 while
        #  the deck renders cohort, side-list and market-cap-band names too.  See
        #  `offline_field`.
        self._offline_cache = {}
        self._validity_cache = {}
        self.html_parts = []

    #  ------------------------------------------------------------------------------- #
    #  THE DECK RENDERS MORE NAMES THAN THE AggScore CSV CONTAINS  (reviewer H-1)        #
    #  ------------------------------------------------------------------------------- #
    #  THE DEFECT.  `agg_val` looks a ticker up in `AggScoreTop100`, which is the general
    #  top-100.  The deck renders the general top-20 PLUS the top-5 of each of five cohorts --
    #  56 pages on the 2026-08-13 run, of which only 31 are in that CSV.  For the other 25,
    #  `Price`, `Traded/day` and `Score coverage` all rendered an em-dash.
    #
    #  AND IT FAILED IN THE ONE DIRECTION THAT MIS-SUGGESTS.  Four of the 25 sit BELOW the
    #  $25k/day thin line -- ESOZ.L $1,692, CBSM.PA $2,361, ORIA.PA $5,915, GDC.TO $6,136 --
    #  and every one showed a blank instead of the caution.  The deck fired the thin marker
    #  twice and swallowed it four times, on cohort pages whose median traded value ($796k) is
    #  ~19x thinner than the top-100's ($14.8M).  A blank where a caution belongs reads as "no
    #  concern", which is precisely the silence-as-approval failure the legend was written to
    #  prevent -- so this was the one place the reporting expansion made the deck WORSE.
    #
    #  Secondarily it made N-4 inert on this run: the maximum `imputed_weight_share` among the
    #  31 in-CSV names is 0.1155, under the 0.20 amber line, and all three >=90% names (STRT,
    #  ENS, PET.TO) are outside the top-20 and so have no page.  The marker never rendered.
    #
    #  NOTHING NEW IS FETCHED OR GUESSED.  Every input already ships with the run and resolves
    #  for 25 of 25: `volavgdic_*.pickle` (price, trading currency, volume),
    #  `MissingDataFillReport_<date>.csv` (per-name imputation for the cohort pools as well as
    #  general) and `FxRates_<date>.csv`.  The CSV still WINS wherever it has the name, so no
    #  in-CSV cell changes value.

    def offline_field(self, ticker, column):
        """One reporting-expansion field for `ticker`, from the run's own saved artifacts.

        RESOLVED LAZILY, PER TICKER, and that is a correctness choice rather than a
        performance one.  The obvious implementation -- build the table once over
        `_page_tickers()` -- is WRONG, and was: `_page_tickers` is the YAHOO FETCH set
        (general top-20 + top-5 per cohort = 45 names), while the banded layout renders the
        general top-20 PLUS Mid/Small/Micro top-5 each PLUS the cohorts = 56.  Eleven rendered
        pages were therefore absent from the table and blanked for a second, entirely
        different reason than the one this fallback was written to fix.  Keying off the
        ticker being rendered cannot drift from the layout, so the bug class is gone rather
        than fixed.

        THE PRICE HERE IS A DIFFERENT SNAPSHOT from the AggScore CSV's, which is why the CSV
        always wins and why this carries its own `price_asof`: the CSV price is the
        deliverables-stage profile call (03:36 on 2026-08-13), this is the map-building
        capture (00:17).  Same vendor endpoint, same run -- fine for a display price, NOT fine
        to leave unlabelled, so the page prints the capture date beside a fallback price.
        """
        cache = self._offline_cache
        if ticker not in cache:
            cache[ticker] = self._resolve_offline(ticker)
        return cache[ticker].get(column)

    def _resolve_offline(self, ticker):
        try:
            import carveOut as _co
            #  THIS RUN'S map, passed explicitly -- never carveOut's repo-root glob; see the
            #  `profile_map` block in `load_run_data`.  Empty map -> every field None -> the
            #  page renders the em-dash it renders today.
            pmap = self.data.get('profile_map') or {}
            if ticker not in pmap:
                return {'imputed_weight_share':
                        (self.data.get('fill_by_name') or {}).get(ticker)}
            e = pmap.get(ticker) or {}
            dv = _co.dollar_volume_frame([ticker], profile_map=pmap,
                                         fx=self.data.get('fx_table'),
                                         clone_map=self.data.get('clone_map'),
                                         fx_label=self.data.get('fx_label'))
            return {
                'price': e.get('price'),
                'price_asof': e.get('asof'),
                'priceCurrency': _co.trading_currency(ticker, profile_map=pmap),
                'dollarVolume_usd': dv['dollarVolume_usd'].iloc[0],
                'dollarVolume_basis': dv['dollarVolume_basis'].iloc[0],
                'imputed_weight_share': (self.data.get('fill_by_name') or {}).get(ticker),
            }
        except Exception as _e:
            #  A fallback that cannot be resolved costs the off-CSV pages their new fields --
            #  i.e. exactly the pre-fix behaviour -- and must never cost the deck.
            log.info(f'offline fields unavailable for {ticker}: {type(_e).__name__}: {_e}')
            return {}

    #  ------------------------------------------------------------------------------- #
    #  FORENSIC APPLICABILITY FOR EVERY PAGE, NOT JUST THE ONES IN THE CSV  (Q-44)       #
    #  ------------------------------------------------------------------------------- #
    #  THE DEFECT.  `evaluate_page` read `forensicValid` from `AggScoreTop100` ALONE and
    #  DEFAULTED IT TO True for anything absent.  `AggScoreTop100` is the GENERAL pool, which
    #  `carveOut` has already stripped of every financial -- so `forensicValid == False` does
    #  not occur in it and never has (0 rows across all seven saved runs, 08-13 -> 08-29), and
    #  the 25 of 45 pages that are not in the file at all took the True default.  The guard
    #  therefore could not fire on any deck the pipeline produces: not rarely, STRUCTURALLY.
    #  A guard that cannot fire is worse than no guard, because on the page it reads as
    #  coverage -- the reader infers "no low-confidence marker, so the forensic reading stands".
    #
    #  WHERE THE STATE ACTUALLY LIVES.  Not in one place, so this reads three, in order of how
    #  directly each one saw the name, and stops at the first that has an answer:
    #    1. `AggScoreTop100.forensicValid`   -- the run's own determination, and the only one
    #       reconciled against the API sector (`postBo` -> `ff.applySectorFallback`).
    #    2. `ForensicFlagsTop100.forensicValid` -- same determination pre-reconciliation; a
    #       real fallback rather than a duplicate, because the AggScore CSV is a MERGE and the
    #       column can be absent from it (the offline reduced schema drops columns).
    #    3. `carveout_labels` -- the run's own carve label for the name, mapped by
    #       `carveOut.cohort_forensic_validity`.  This is what covers the 25 cohort pages:
    #       `carveOut.classify` routes off the same sector labels `forensicFlags` calls
    #       bank/insurer/REIT, so the label already carries the verdict.
    #
    #  AND NOTHING ELSE -- DELIBERATELY.  The obvious fourth source is `sectorsdic_fmp.pickle`,
    #  and reading it here would be a defect, not a fallback: it is UNDATED and rebuilt every
    #  run, `resolve_run_artifacts` refuses exactly this kind of cross-run substitution for the
    #  dated artifacts, and the two copies on this machine are not even the same TAXONOMY (40
    #  keys / 40,164 symbols in the repo, 11 / 11,499 in the transfer dir -- see carveOut's own
    #  warning).  Measured: classifying the 45 pages of the 2026-08-29 deck from the repo copy
    #  instead of the run's flips SIX of them, in both directions.  A guard wired to a source
    #  that can silently disagree with the run is the Q-44 shape again with extra steps.
    #
    #  THE DEFAULT IS NOT True.  Unresolved is `None` -- undetermined -- and the caller fires
    #  the low-confidence marker on it.  An absent classification is an ABSENCE; asserting the
    #  fraud models apply to a name nobody classified is the assertion that made this guard
    #  dormant in the first place.
    _VALIDITY_NOTE = {
        True:  '',
        False: 'the forensic models (Beneish / Montier / Sloan) do not apply to a '
               'financial — read this on a financial lens',
        None:  'we could not determine whether the forensic models apply to this name '
               '— this is an absence of classification, not a clean reading',
    }

    def forensic_validity(self, ticker):
        """(tri_state, source) for one page name. `tri_state` is True / False / None, where
        None means UNDETERMINED and is never silently promoted to True."""
        cache = self._validity_cache
        if ticker not in cache:
            cache[ticker] = self._resolve_forensic_validity(ticker)
        return cache[ticker]

    def _resolve_forensic_validity(self, ticker):
        import forensicFlags as ff
        for key, src in (('aggscore_df', 'AggScoreTop100'),
                         ('forensic_df', 'ForensicFlagsTop100')):
            df = self.data.get(key)
            if df is None or getattr(df, 'empty', True) or 'forensicValid' not in df.columns:
                continue
            row = df[df['source'] == ticker]
            if row.empty:
                continue
            t = ff.published_forensic_validity(row.iloc[0].get('forensicValid'))
            if t is not None:
                return t, src
        labels = self.data.get('carveout_labels')
        if labels is not None and ticker in getattr(labels, 'index', []):
            try:
                import carveOut as _co
                t = _co.cohort_forensic_validity(labels.loc[ticker])
            except Exception as _e:
                log.info(f'carve-label validity unavailable for {ticker}: '
                         f'{type(_e).__name__}: {_e}')
                t = None
            if t is not None:
                return t, 'carve label %s' % labels.loc[ticker]
        return None, 'unresolved'

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
        _rpy = rpy_for_source(ticker)      # EXPLICIT (never rely on the frame-derived default)
        ni_ttm, cfo_ttm = ttm_aligned_sums(cdx, ['netIncome', 'netCashProvidedByOperatingActivities'], _rpy)
        ni_ttm_rev, rev_ttm = ttm_aligned_sums(cdx, ['netIncome', 'revenue'], _rpy)
        # POOL-BASIS for the four POOL_BASIS_MARKERS: the verdict icon and the flag rules MUST
        # be computed on the SAME value the page displays beside them (Sections C and G both
        # show the pool basis now), otherwise the icon contradicts the number it annotates.
        # BASIS (implemented state -- display-layer annualization): raw_metric() returns the DISPLAY-ANNUALIZED
        # value -- compute_cohort_percentiles already multiplied ANNUALIZED_DISPLAY_METRICS by
        # ANNUALIZE_DISPLAY_FACTOR (x4) in both raw_all and the pools. So these are ANNUAL
        # figures and the A.1 thresholds below (ROIC 0.15/0.10, R1/R4's ROE>0.15) are annual
        # rules of thumb applied to annual values -- correctly calibrated as-is.
        # DO NOT re-annualize here: another x4 would make the comparisons x16 too lenient.
        roic = self.raw_metric(ticker, 'returnOnCapitalEmployed')
        roe = self.raw_metric(ticker, 'returnOnEquity')
        roa = self.raw_metric(ticker, 'RoA')
        gm = L('grossProfitMargin'); income_qual = L('incomeQuality')
        # ANNUAL Net Debt/EBITDA from the extended pool -- the same value Section C displays
        # and the same basis R4's "> 4x" limb is written for (never the raw per-period field).
        net_debt_ebitda = self.ext_val(ticker, 'net_debt_ebitda')
        curr_ratio = L('currentRatio')
        op_margin = compute_operating_margin(cdx)
        fcf_margin = compute_fcf_margin_ttm(cdx, _rpy)
        cash_conv = compute_cash_conversion(cdx, _rpy)
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

        #  aggscore forensic fields.  THE COMMENT THAT USED TO SIT HERE -- "every rendered
        #  name is within Top-100" -- WAS FALSE, and was the whole of Q-44: the deck renders
        #  the general band pages PLUS the top-5 of five carve cohorts, and on 2026-08-29
        #  exactly 20 of its 45 pages were in that CSV.  M/C stay CSV-only (they are scores,
        #  and there is no offline source for the other 25 -- they render ⚪ 'value
        #  unavailable', which is true).  The APPLICABILITY verdict does NOT stay CSV-only;
        #  see `forensic_validity` for where it really lives and why it defaults to
        #  undetermined rather than to valid.
        m_score = c_score = np.nan; m_flag = False; m_drivers = ''
        agg = self.data.get('aggscore_df')
        if agg is not None and not agg.empty:
            ar = agg[agg['source'] == ticker]
            if not ar.empty:
                r0 = ar.iloc[0]
                m_score = safe_float(r0.get('M-Score')); c_score = safe_float(r0.get('C-Score'))
                m_flag = str(r0.get('M_flag_gt_-1.78')).strip().lower() in ('true', '1', 'yes')
                m_drivers = str(r0.get('M_drivers') or '')
        forensic_state, _ = self.forensic_validity(ticker)

        # dilution: shares +>10% over 3y (Section-H computation reused)
        dilution = False
        if 'weightedAverageShsOut' in cdx.columns:
            sh = cdx['weightedAverageShsOut'].dropna().values   # newest-first
            if len(sh) >= 12 and sh[-1] > 0 and (sh[0] / sh[-1] - 1) > 0.10:
                dilution = True

        # ---- Y2 low-confidence guards ----
        #  FIRES ON *NOT-ESTABLISHED-VALID*, NOT ON *ESTABLISHED-INVALID* (Q-44).  The two
        #  differ on exactly the names this guard was blind to, and the marker carries WHICH
        #  of the two it is, so an undetermined name is not dressed up as a classified one.
        low_conf_forensic = (self._VALIDITY_NOTE[forensic_state]
                             if forensic_state is not True else False)   # -> M/C/Sloan 🟡
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
        # The Altman-Z limb was REMOVED (soundness audit; re-derived 2026-07-26): our computed
        # "Z" is ~one leverage term (0.6*x4 = mcap/total-liabilities, 66.5% of the score, corr
        # 0.997), not the published five-factor statistic, so a "<1.8" trip is not calibrated
        # evidence of distress.
        # Both remaining limbs are HIGH-tier (the earlier Z-only -> Medium path is gone with it).
        if not is_fin:
            trips = []
            if not np.isnan(m_score) and m_score > 0: trips.append(f"M-Score {m_score:.2f} > 0")
            if not np.isnan(c_score) and c_score >= 4: trips.append(f"C-Score {c_score:.0f} of 5 ≥ 4")
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

        # LABEL THE TAXONOMY THAT WAS ACTUALLY USED (fix, 2026-07-30).  `sector` comes from the
        # API-sourced `sector` column when present, and otherwise falls back to
        # `get_industry()` -- a DIFFERENT taxonomy (156 FMP industries vs ~11 sectors).  The
        # field was labelled "Sector:" either way, so on any run without that column (e.g. the
        # offline-reduced deck) the page silently showed an industry under a sector label:
        # "Marine Shipping" is not a sector.  Suffix the fallback so the reader can see which
        # taxonomy they are looking at.
        sector = None
        if aggscore_df is not None and not aggscore_df.empty:
            row = aggscore_df[aggscore_df['source'] == ticker]
            if not row.empty:
                sector = row.iloc[0].get('sector', None)
        if sector is None or (isinstance(sector, float) and np.isnan(sector)) \
                or str(sector).strip() in ('', '—', 'nan'):
            _ind = self.data['get_industry'](ticker)
            sector = ('%s (industry)' % _ind) if _ind and _ind != 'Unknown' else '—'

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

    def agg_val(self, ticker, column, default=None):
        """One reporting-expansion field for `ticker`: the AggScore CSV if it has the name,
        otherwise the run's own saved artifacts, otherwise `default`.

        THE CSV WINS WHEREVER IT HAS THE NAME, so adding the fallback changed no in-CSV cell.
        The fallback exists because the deck renders 56 pages against a 100-name CSV that
        covers only the general pool -- see `_build_offline_fields` for the measured cost of
        not having it.

        THE ABSENT-COLUMN CASE IS STILL NORMAL, not an error: `priceCurrency`,
        `dollarVolume_usd` and `imputed_weight_share` first exist from the 2026-08-13
        reporting expansion, and the deck must still render an older run.  Returning
        `default` (None -> "—" at every call site) keeps absence reading as absence; the one
        thing that must never happen is a 0 standing in for "we did not record this".
        """
        return self._field(ticker, column, default)[0]

    def _field(self, ticker, column, default=None):
        """`(value, origin)` where origin is 'aggscore' | 'offline' | None.

        The origin is not decoration: a fallback PRICE is a different capture from the CSV's
        and the page has to say so, which it cannot do if the two are indistinguishable.
        """
        df = self.data.get('aggscore_df')
        if df is not None and not getattr(df, 'empty', True) and column in df.columns:
            r = df[df['source'] == ticker]
            if not r.empty:
                v = r.iloc[0].get(column, None)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    return v, 'aggscore'
        if getattr(self, '_offline_cache', None) is not None:
            v = self.offline_field(ticker, column)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return v, 'offline'
        return default, None

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

        # PRICE -- SOURCED FROM THE AggScore CSV AND LABELLED WITH ITS CURRENCY.
        # It used to read `row.get('price')` off postRank, WHICH HAS NO SUCH COLUMN, so this
        # field rendered "—" on every name of every run; the `f"${price:.2f}"` it fell through
        # to was a latent currency defect (register N-3) waiting for the column to appear.
        # Both halves now come from the same CSV row, so the number and its label cannot
        # disagree. `priceCurrency` is absent on pre-2026-08-13 runs -> "(currency unknown)",
        # never a guessed `$`.
        _px, _px_origin = self._field(ticker, 'price')
        price_str = price_with_currency(_px, self.agg_val(ticker, 'priceCurrency'))
        if _px_origin == 'offline':
            #  A FALLBACK PRICE IS A DIFFERENT CAPTURE and says so (reviewer H-1): the CSV's
            #  price is the deliverables-stage profile call, this one is the earlier
            #  map-building capture of the same endpoint on the same day.  Small enough not to
            #  matter to a reader, big enough that an unlabelled mix would be exactly the kind
            #  of silent two-basis column this whole change exists to remove.
            _asof = self.offline_field(ticker, 'price_asof')
            price_str += (f' <span class="pctile" title="This name is outside the general '
                          f'top-100, so its price comes from the run\'s profile capture '
                          f'rather than the AggScore CSV. Same vendor field, same run, '
                          f'earlier in the run.">(capture {escape(str(_asof))})</span>'
                          if _asof else '')

        # MARKET CAP -- FROM `marketCap_usd`, NOT `marketCap` (register N-3, same class).
        # `cdx_df['marketCap']` is denominated in each company's own reportedCurrency, so
        # `f"${mktcap/1e9:.2f}B"` printed 000660.KS as "$1890724.65B" (KRW 1.89e15 wearing a
        # dollar sign) and SKHY as "$568935.00B". `marketCap_usd` is the pipeline's own
        # converted column -- the SAME one the market-cap bands partition on -- so the deck,
        # the banding and the grading now agree instead of the deck being alone in raw units.
        # NO FALLBACK TO THE RAW FIELD: a wrong number wearing a right label is the failure
        # mode here, so an unconverted name reads "—".
        cdx = self.get_cdx_for_ticker(ticker)
        mktcap_usd = latest_row_value(cdx, 'marketCap_usd')
        mktcap_str = (f'<span title="Converted to USD from the company&#39;s reportedCurrency '
                      f'with the run&#39;s FX (carveOut.marketcap_usd_series) — the same '
                      f'conversion the market-cap bands use.">{money_format(mktcap_usd)}</span>'
                      if not np.isnan(safe_float(mktcap_usd)) else "—")

        # TRADED VALUE PER DAY (CEO 2026-08-13). Neutral number; one-sided thin-market
        # caution only -- see the valence table beside `dollar_volume_cell`.
        dollarvol_str = dollar_volume_cell(self.agg_val(ticker, 'dollarVolume_usd'))

        rating = "—"
        if aggscore_df is not None:
            ag_row = aggscore_df[aggscore_df['source'] == ticker]
            if not ag_row.empty:
                rating = ag_row.iloc[0].get('rating_fmp', '—')

        # Determine nav bucket
        nav_bucket = "top-5" if rank <= 5 else ("top-10" if rank <= 10 else "top-20")

        # REPORTING FREQUENCY (CEO-facing; previously printed nowhere, so a semi-annual filer
        # was indistinguishable from a quarterly one except via a sparkline span caption).
        # It drives every TTM window on the page, so it belongs beside the identity fields.
        freq_label, freq_inferred = reporting_frequency(ticker, self.data.get('cdx_df'))
        freq_tag = ' <span class="pctile">(inferred from filing cadence)</span>' if freq_inferred else ''
        freq_title = ('Filing frequency. Drives every trailing-twelve-month window on this page '
                      '(Sloan accruals, FFO/share, P/FFO, FCF vs NI). '
                      + ('INFERRED from the date cadence — this run predates the pipeline stamping '
                         'reportingFrequency at ingest.' if freq_inferred
                         else 'From the pipeline ingest verdict (reportingFrequency).'))

        html = f"""
        <div class="section-a">
            <h2>{ticker} <span class="subtitle">{name}</span></h2>
            <div class="meta-row">
                <span><strong>Exchange:</strong> {exchange}</span>
                <span><strong>Sector:</strong> {sector}</span>
                <span><strong>Price:</strong> {price_str}</span>
                <span><strong>Market Cap:</strong> {mktcap_str}</span>
                <span><strong>Traded/day:</strong> {dollarvol_str}</span>
                <span title="{escape(freq_title)}"><strong>Reporting:</strong> {escape(freq_label)}{freq_tag}</span>
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
        #  THE ABSTENTION SAYS WHY, ON THE DECK TOO (CEO, 2026-08-16).  After the O-13 domain
        #  guards a fifth of the shortlist carries `data-incomplete: dig-deeper`, and a tag
        #  that says only "incomplete" reads as a broken tool at that rate; the reason names
        #  the vendor input that is missing.  Appended to the SAME chip rather than added as a
        #  new one, because the reason is meaningless without the tag it qualifies -- the same
        #  adjacency argument the `Score coverage` cell rests on.
        #  READ FROM THE SAME ROW AS THE TAG, with the forensic CSV as the fallback.  There is
        #  deliberately no `offline_field` path: that fallback resolves PROFILE-map fields, and
        #  a page with no `forensicTag` has no abstention to explain in the first place.
        _why = ''
        if aggscore_df is not None:
            ag_row = aggscore_df[aggscore_df['source'] == ticker]
            if not ag_row.empty:
                forensic_tag = ag_row.iloc[0].get('forensicTag', '—')
                _why = ag_row.iloc[0].get('M_abstain_reason', '')
        if not str(_why).strip() or str(_why).strip().lower() == 'nan':
            _fdf = self.data.get('forensic_df')
            if _fdf is not None and not getattr(_fdf, 'empty', True) \
                    and 'M_abstain_reason' in _fdf.columns:
                _fr = _fdf[_fdf['source'] == ticker]
                _why = _fr.iloc[0].get('M_abstain_reason', '') if not _fr.empty else ''
        _why = '' if _why is None or (isinstance(_why, float) and np.isnan(_why)) else str(_why)
        if _why.strip() and _why.strip().lower() != 'nan':
            forensic_tag = f"{forensic_tag} <span class=\"forensic-why\">— {_why}</span>"
        #  AND WHETHER THE MODELS APPLY AT ALL (Q-44, 2026-08-31).  25 of the 45 pages on the
        #  2026-08-29 deck have no row in that CSV, so the chip above them read a bare
        #  em-dash -- and a bare em-dash beside the word "Forensic" is read as "nothing to
        #  report", which is the silence-as-approval failure the icon legend exists to
        #  prevent.  The applicability verdict is resolvable for those pages (see
        #  `forensic_validity`); the tag is not, so this states the one it has and names its
        #  own source, rather than leaving the reader to infer either.
        _vstate, _vsrc = self.forensic_validity(ticker)
        if _vstate is not True:
            _vtxt = ('forensic models INVALID here (financial) — use a financial lens'
                     if _vstate is False else
                     'applicability NOT DETERMINED — no forensic classification for this '
                     'name in this run')
            #  ITS OWN CLASS, not `forensic-why`.  That one explains why an M-score is
            #  ABSENT (a statement about the vendor's data); this states whether the models
            #  APPLY AT ALL (a statement about the business).  Sharing a class would let a
            #  future test or style change treat the two as one fact.
            forensic_tag = (f'{forensic_tag} <span class="forensic-applic">— {_vtxt} '
                            f'[{escape(str(_vsrc))}]</span>')

        # HOW MUCH OF THIS SCORE IS MEASURED (register N-4, CEO 2026-08-13).
        # Placed IMMEDIATELY BESIDE `Forensic`, deliberately and not anywhere else: the exact
        # misread this closes is a reader seeing `forensicTag = clean` on a name whose score
        # is 93% imputed and concluding the name is well-understood. The two facts have to be
        # adjacent for the reader to see they are different facts. NOT styled like the
        # forensic tag (which is red-on-white by class) -- see `data_coverage_cell`.
        coverage = data_coverage_cell(self.agg_val(ticker, 'imputed_weight_share'))

        html = f"""
        <div class="section-b banner">
            <div class="score-banner">
                <span class="score-item"><strong>AggScore:</strong> {ratio_format(agg_score, 1)} {orient_chip('AggScore')}</span>
                <span class="score-item"><strong>MoatScore:</strong> {ratio_format(moat_score, 1)} {orient_chip('moatScore')}</span>
                <span class="score-item forensic"><strong>Forensic:</strong> {forensic_tag}</span>
                <span class="score-item"><strong>Score coverage:</strong> {coverage}</span>
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
            # rpy EXPLICIT: FFO is a 12-month FLOW feeding P/FFO against a point-in-time
            # market cap, so a wrong window would misprice a semi-annual REIT (~2x) AND
            # mis-position its dot against the rpy-correct pool.
            ffo_per_share = compute_ffo_per_share(cdx_df, rpy_for_source(ticker))
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
            # ANNUAL multiple from the extended pool (= the marker its bar is drawn against
            # and the basis R4's "> 4x" limb uses); the raw cdx field is per-period.
            net_debt_ebitda = self.ext_val(ticker, 'net_debt_ebitda')
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
                    <tr><td><strong>Net Debt/EBITDA (ann.)</strong></td><td>{ratio_format(net_debt_ebitda)}x {nde_bar}</td><td>annual EBITDA · trailing EBITDA distorts cyclicals</td></tr>
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
            # POOL-BASIS (reviewer finding H2 + the display-annualization ruling that sits on
            # the pipeline's 2026-07-25 period-aware work): ROE/ROA are
            # POOL_BASIS_MARKERS, so BOTH the displayed number and the dot use the pool's
            # period-corrected basis (raw_metric -> raw_all -> _pool_raw_fast) -- never the
            # raw cdx latest row. This keeps Section C identical to Section G for the same
            # metric on the same page, and the dot can never disagree with the number beside
            # it. (Per-quarter trajectory is still visible in Section D's sparklines.)
            roe = self.raw_metric(ticker, 'returnOnEquity')
            roa = self.raw_metric(ticker, 'RoA')       # pool key is 'RoA' (cdx col: returnOnAssets)
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
                    <tr><td><strong>ROE (ann.)</strong> {orient_chip('returnOnEquity')}</td><td>{pct_format(roe)}{self._vf(ev, 'returnOnEquity')} {roe_bar}</td><td>annualized (pool basis)</td></tr>
                    <tr><td><strong>ROA (ann.)</strong> {orient_chip('RoA')}</td><td>{pct_format(roa)}{self._vf(ev, 'RoA')} {roa_bar}</td><td>annualized (pool basis)</td></tr>
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
            # POOL-BASIS for ROE/ROIC (see the Fin block above for the rationale).
            roe = self.raw_metric(ticker, 'returnOnEquity')
            roic = self.raw_metric(ticker, 'returnOnCapitalEmployed')

            roe_bar = self.dist_bar(ticker, cohort_label, 'returnOnEquity', marker=roe)
            roic_bar = self.dist_bar(ticker, cohort_label, 'returnOnCapitalEmployed', marker=roic)

            html = f"""
            <div class="section-c valuation">
                <h3>Valuation Ratios (InvestmentVehicle)</h3>
                <table class="metrics-table">
                    <tr><td><strong>P/B (NAV proxy)</strong></td><td>{ratio_format(pb)}</td><td>disc to NAV</td></tr>
                    <tr><td><strong>ROE (ann.)</strong> {orient_chip('returnOnEquity')}</td><td>{pct_format(roe)}{self._vf(ev, 'returnOnEquity')} {roe_bar}</td><td>annualized (pool basis)</td></tr>
                    <tr><td><strong>ROIC (ann.)</strong> {orient_chip('returnOnCapitalEmployed')}</td><td>{ratio_format(roic)}{self._vf(ev, 'returnOnCapitalEmployed')} {roic_bar}</td><td>annualized (pool basis)</td></tr>
                </table>
                <div class="gap-note">[NAV/holdings composition not obtainable from filter data]</div>
            </div>
            """
            return html

        else:
            # General block (default) - all raw values from cdx_df, not z-scores from postrank
            # EXCEPT the POOL_BASIS_MARKERS: ROIC uses the pool's period-corrected basis for
            # BOTH the number and the dot (see the Fin block above). grossProfitMargin is a
            # same-period ratio -> annualization cancels -> cdx latest is already on-basis.
            roic = self.raw_metric(ticker, 'returnOnCapitalEmployed')
            gm = latest_row_value(cdx_df, 'grossProfitMargin')
            op_margin = compute_operating_margin(cdx_df)
            _rpy_c = rpy_for_source(ticker)
            fcf_margin = compute_fcf_margin_ttm(cdx_df, _rpy_c)
            cash_conv = compute_cash_conversion(cdx_df, _rpy_c)
            income_qual = latest_row_value(cdx_df, 'incomeQuality')
            # ANNUAL multiple from the extended pool (= the marker its bar is drawn against
            # and the basis R4's "> 4x" limb uses); the raw cdx field is per-period.
            net_debt_ebitda = self.ext_val(ticker, 'net_debt_ebitda')
            int_cov = compute_interest_coverage(cdx_df)
            # P/E: general names -> AggScoreTop100['PE-ratio']; fallback / carve names not
            # in the top-100 CSV -> 1 / earnings yield, guarding non-positive or NaN earnings
            # (gap-tag rather than emit a garbage negative/huge P/E).
            # The fallback uses the ANNUALIZED pool-basis earnYield (raw_metric), NOT the raw
            # cdx latest row: inverting a per-PERIOD yield produced a P/E inflated by the
            # annualization factor (a quarterly row's yield is ~1/4 of the annual one, so 1/ey
            # read ~4x too high). 1 / annual-yield is a genuine annual P/E, and it is correct
            # for semi-annual filers too (the pool basis normalizes them before we scale).
            pe_ratio = np.nan
            if aggscore_df is not None and not aggscore_df.empty:
                ag_row = aggscore_df[aggscore_df['source'] == ticker]
                if not ag_row.empty:
                    pe_ratio = safe_float(ag_row.iloc[0].get('PE-ratio'))
            if np.isnan(pe_ratio):
                ey = self.raw_metric(ticker, 'earnYield')      # annualized (see above)
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
                    <tr><td><strong>ROIC/ROCE (ann.)</strong> {orient_chip('returnOnCapitalEmployed')}</td><td>{ratio_format(roic)}{self._vf(ev, 'returnOnCapitalEmployed')} {roic_bar}</td><td>annualized (pool basis)</td></tr>
                    <tr><td><strong>Gross Margin</strong> {orient_chip('grossProfitMargin')}</td><td>{pct_format(gm)}{self._vf(ev, 'grossProfitMargin')} {gm_bar}</td><td>latest</td></tr>
                    <tr><td><strong>Op Margin</strong></td><td>{pct_format(op_margin)}{self._vf(ev, 'op_margin')} {opm_bar}</td><td>TTM</td></tr>
                    <tr><td><strong>FCF Margin</strong></td><td>{pct_format(fcf_margin)}{self._vf(ev, 'fcf_margin')} {fcfm_bar}</td><td>TTM</td></tr>
                    <tr><td><strong>Cash Conversion</strong></td><td>{ratio_format(cash_conv)}{self._vf(ev, 'cash_conv')}</td><td>TTM FCF / NI</td></tr>
                    <tr><td><strong>Income Quality</strong> {orient_chip('incomeQuality')}</td><td>{ratio_format(income_qual)}{self._vf(ev, 'incomeQuality')} {iq_bar}</td><td>audit</td></tr>
                    <tr><td><strong>Net Debt/EBITDA (ann.)</strong></td><td>{ratio_format(net_debt_ebitda)}x{self._vf(ev, 'net_debt_ebitda')} {nde_bar}</td><td>annual EBITDA</td></tr>
                    <tr><td><strong>Interest Coverage</strong></td><td>{ratio_format(int_cov)}{self._vf(ev, 'interest_coverage')} {intcov_bar}</td><td>op inc / int exp</td></tr>
                    <tr><td><strong>Effective Tax</strong></td><td>{pct_format(eff_tax)} {efftax_bar}</td><td>latest, clip[0,1]</td></tr>
                    <tr><td><strong>Days Sales Outstanding</strong></td><td>{ratio_format(dso)} {dso_bar}</td><td>latest</td></tr>
                    <tr><td><strong>Inventory Days</strong></td><td>{ratio_format(inv_days)} {invd_bar}</td><td>goods cohorts</td></tr>
                    <tr><td><strong>P/E</strong></td><td>{ratio_format(pe_ratio)}{self._vf(ev, 'peRatio')}</td><td>traded or yield inv</td></tr>
                    <tr><td><strong>FCF Yield (ann.)</strong> {orient_chip('freeCashFlowYield')}</td><td>{pct_format(fcf_yield)}{self._vf(ev, 'freeCashFlowYield')} {fcfy_bar}</td><td>annualized (pool basis)</td></tr>
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

        # Header must state THIS company's period. It said "(Quarterly)" for every name, which
        # is plainly wrong for a semi-annual filer -- and it sat directly above the span caption
        # that reveals the real cadence. Chose accuracy over dropping the word: each point on
        # these sparklines IS one reporting period, so naming the period is genuinely useful.
        _trend_period_word = ('Semi-Annual' if rpy_for_source(ticker) == 2 else 'Quarterly')

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
            <h3>Multi-Year Trends ({escape(_trend_period_word)})</h3>
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
        _rpy_f = rpy_for_source(ticker)
        fcf_ttm = ttm_sum(cdx, 'freeCashFlow', _rpy_f)
        ni_ttm = ttm_sum(cdx, 'netIncome', _rpy_f)

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
                <tr><td><strong>Sloan Accruals</strong> — pipeline value <span class="pctile">(as-shipped CSV; (NI−CFO)/avg assets)</span> {orient_chip('sloanAccruals')}</td><td>{sloan}<span class="pctile"> · no bar/verdict here — see the peer row below</span></td></tr>
                <tr><td><strong>Sloan Accruals</strong> — peer-pool recompute <span class="pctile">(same definition: (NI−CFO)/avg assets, 12-mo window; THIS row carries the 0.10 bar, the verdict icon and the 🚩)</span> {orient_chip('sloanAccruals')}</td><td>{ratio_format(sloan_cohort)}{sloan_vf} {sloan_bar}</td></tr>
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

        # Solvency check REMOVED (soundness audit; re-derived 2026-07-26): it rested on the
        # computed Altman-Z being < 1.8, but that statistic is ~ONE term (0.6*x4 =
        # mcap/total-liabilities, 66.5% of the score, corr 0.997) plus a wrong x2 variable --
        # not the published five-factor Z -- so the 1.8 cut is NOT calibrated for
        # the quantity we compute -- a "Solvency Risk (Z<1.8)" banner was an uncalibrated
        # verdict. Altman-Z is now presented relative-only (⚪ verdict, no tick on its peer
        # bar). The interest-coverage and leverage checks below remain as the solvency signals
        # that do NOT depend on it. Do not reinstate a Z-based solvency flag without a
        # recalibration decision.

        # Interest coverage check
        int_cov = compute_interest_coverage(cdx)
        if not np.isnan(safe_float(int_cov)) and safe_float(int_cov) < 2:
            flags.append(('RED', 'Low Interest Coverage (<2)'))

        # Leverage check -- ANNUAL Net Debt/EBITDA (net_debt_to_ebitda_annual). The raw cdx
        # field is a per-PERIOD multiple, so testing it against the annual "> 4x" bar fired on
        # ~2.4x as many names as intended (the SECOND hard-coded >4 site; the first is R4).
        net_debt_ebitda = net_debt_to_ebitda_annual(cdx, rpy_for_source(ticker))
        if not np.isnan(safe_float(net_debt_ebitda)) and safe_float(net_debt_ebitda) > 4:
            flags.append(('AMBER', 'High Leverage (ND/EBITDA>4x, annual)'))

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

        # The GENERAL TOP-20 this deck renders: the banded General head-20 when currency data
        # is present, the flat postRank head-20 otherwise. Resolved ONCE here so the industry
        # counter counts exactly the names the page shows.
        general_top20 = (list(band_names.get('General') or [])[:20] if banded
                         else postrank_df.head(20)['source'].tolist())

        # rpy_basis_banner() is '' in the healthy case; it renders a loud page-level warning
        # if the filing-frequency map degraded (domain N9 -- never a silent basis regression).
        content = ("""<div class="content">""" + rpy_basis_banner()
                   + schema_note_banner(self.data.get('aggscore_df'))
                   + industry_counter_banner(postrank_df, general_top20,
                                             self.data.get('industrydic'),
                                             self.data.get('cdx_df'))
                   #  WHAT WAS REMOVED BEFORE THIS SHORTLIST WAS DRAWN (register N-5).
                   #  Placed with the industry counter, not with the warnings: both are
                   #  composition facts about the pool the reader is looking at.
                   + veto_scope_banner(self.data.get('stage1_veto'))
                   + self._icon_legend())
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
            top_20 = general_top20
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
            #  ⚪ CARRIES TWO DISTINCT FACTS and the legend now says both (P-5, 2026-08-29).
            #  It always did -- `compute_verdict` has returned ⚪ 'value unavailable' since the
            #  rules were written -- but the legend only advertised the no-rule half, and after
            #  P-5 an absent value is ⚪ even when the metric is also flagged low-confidence.
            #  Hover gives the exact one; the legend must not leave a reader inferring that a
            #  ⚪ means "we looked and there is no rule" when it can mean "there is no number".
            '<span class="leg">⚪ no standalone rule (peer-relative), or no value</span>'
            '<span class="leg-sep">·</span>'
            '<span class="leg">🚩<sub>H</sub> High — treat as fact, investigate before buying</span>'
            '<span class="leg">🚩<sub>M</sub> Medium — real signal, needs business context</span>'
            '<span class="leg">🚩<sub>L</sub> Low — a prompt to look, not a verdict</span>'
            '<div class="leg-note">The 🚩 is independent of the verdict icon — a 🟢🚩 or ⚪🚩 '
            'is valid (a good/no-rule number can still be flagged). A MISSING value always shows ⚪ '
            'even when it is also low-confidence: absence is the stronger fact, so it is the '
            'one the icon reports. '
            #  THE THIRD STATE, SAID OUT LOUD (Q-44).  M-Score / C-Score / Sloan carry a
            #  🟡 whenever the forensic models are not ESTABLISHED to apply -- which covers
            #  both a name classified as a financial and a name nobody classified.  Hover
            #  gives which; the legend must not let the reader read the second as the first.
            'M-Score / C-Score / Sloan show 🟡 whenever it is not established that the '
            'forensic models apply to the name — either because it is a financial (they do '
            'not apply) or because no forensic classification for it exists in this run '
            '(we do not know); hover says which. '
            'Hover any icon for the rule '
            'and, for a flag, the mechanism + offending metric + values + tier + rule id.</div>'
            #  THE VALENCE OF THE FIELDS ADDED 2026-08-13, STATED ON THE PAGE.  A marker whose
            #  direction the reader has to infer is a marker that can be inferred backwards,
            #  which is the exact failure the CEO named.  Two of these three are deliberately
            #  ONE-SIDED and the legend says so, so their SILENCE is not read as approval.
            '<div class="leg-note"><strong>Price / Market Cap / Traded per day / Score '
            'coverage —</strong> <em>Price</em> is quoted in the listing’s OWN currency and '
            'the code is printed beside it (no <code>$</code> is assumed). <em>Market Cap</em> '
            'is converted to USD. <em>Traded/day</em> is average traded VALUE, shown NEUTRALLY: '
            'a high figure earns no positive marker — liquidity is not merit, and this filter '
            'deliberately looks for neglected names — but an amber <span class="flag AMBER">'
            'thin market</span> marks a name you may not be able to size or exit. '
            '<em>Score coverage</em> is how much of the rank rests on MEASURED metrics rather '
            'than imputed fills; low coverage means we know little about the name, and is '
            'independent of the forensic tag.</div>'
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

/* WHY a name has no M-score.  Deliberately NOT the red of the tag it follows (CEO's standing
   "presentation must be correctly suggestive"): the tag is a forensic finding, this is a
   statement about the vendor's data, and rendering a data gap in alarm-red would read as a
   forensic red flag -- the exact inversion the CEO ruled against for the imputed-share
   marker. */
.forensic-why,
/* ...and, in the same muted italic and for the same reason, whether the forensic MODELS
   apply to this name at all (Q-44).  Also not the tag's red: "the Beneish model does not
   apply to a REIT" is a scope statement, not a finding about the company. */
.forensic-applic {
    color: #6c757d;
    font-style: italic;
    font-weight: normal;
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
.basis-warning {
    background: #fff3cd;
    border: 2px solid #b30000;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 0 0 14px 0;
    font-size: 0.9em;
    line-height: 1.5;
    color: #5a3d00;
}

/* INDUSTRY COUNTER -- informational composition block; deliberately styled NEUTRAL (same
   card as the legend, not the red .basis-warning) because a concentration is a fact for the
   reader to weigh, not a defect in the run. */
.industry-counter {
    background: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 0 0 14px 0;
    font-size: 0.85em;
    line-height: 1.8;
}
.industry-counter .ic-list { margin-top: 5px; }
.industry-counter .ic-head {
    display: inline-block;
    min-width: 130px;
    font-weight: 600;
    color: #444;
}
.industry-counter .ic-cell {
    display: inline-block;
    margin-right: 10px;
    white-space: nowrap;
}
.industry-counter .ic-hot { background: #ffe9c7; border-radius: 3px; padding: 0 4px; }
.industry-counter .ic-unc { color: #777; font-style: italic; }
.industry-counter .ic-pct { color: #777; }
.industry-counter .ic-detail {
    margin: 8px 0 0 0;
    padding: 6px 8px;
    background: #fff;
    border: 1px solid #e1e4e8;
    border-radius: 4px;
    font-size: 0.95em;
    white-space: pre-wrap;
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
