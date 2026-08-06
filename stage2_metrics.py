"""Stage-2 per-ticker metric formulas -- the SINGLE source of truth.

These are the exact metric computations of the Stage-2 (AggScore) scorer.  They
are imported by BOTH:

  * production  postBoRank.postBoScoreRanking   (the LIVE scorer)
  * offline     baseline_tools/stage2_pit._stage2_metric_loop_offline
                (the certified point-in-time reproduction / validation gate)

Before this module existed the two carried hand-copied duplicates of every
formula, kept in sync only by a "MUST stay in lockstep" comment -- a silent-
divergence hazard for the validation north-star (structural review S1).  Housing
each formula ONCE here removes the hand-sync: touch a formula and both callers
move together.

Behaviour contract: every function reproduces the *production* formula
bit-for-bit (production is the shipped scorer).  For CycleHeat the two paths were
NOT truly equivalent before -- on duplicate-dated (restated) EPS quarters the
live scorer and the offline reproduction picked DIFFERENT tied rows for the
"current" quarter (the ILMN-type case).  They are now ALIGNED, by construction,
via a canonical restatement tie-break in the shared EPS prep
(prepare_eps_series -> keep the last-ingested record per date); both paths call
the one shared cycleheat().  Every other metric is a straight relocation of the
production formula.

Pure functions only: they take pandas objects in and return a scalar (or, for
add_mcap_quants, a Series).  No network, no I/O, no DataFrame mutation of the
caller's frames.

TWO PRECONDITIONS ON EVERY `tempcdx` PASSED IN HERE, both owned by the caller:

  1. ROW ORDER = reporting_period.NEWEST_FIRST (row 0 = the most recent period).  Every
     window below is POSITIONAL -- head(w), iloc[0], iloc[lag], pct_change(-1),
     shift(-rpy) -- so the wrong orientation reads the wrong end of the history silently.
     The boundary that guarantees it is postBoRank._sort_cdx_newest_first for the live
     scorer and the equivalent sort in baseline_tools/stage2_pit for the offline path;
     the shared vocabulary and the check live in reporting_period ("ROW ORDER" section).
     These functions do NOT re-sort: re-sorting per metric would hide a mis-oriented
     caller and would change tie order on duplicate-dated rows.
  2. `rpy` is THIS source's rows-per-year.  What each metric does with it -- window basis
     and frequency treatment -- is declared ONCE in STAGE2_METRIC_SPEC below, not decided
     at the call sites.
"""

import numpy as np
import pandas as pd

import nan_policy as npol
import reporting_period as rp

# =========================================================================== #
#  THE STAGE-2 METRIC REGISTRY -- THE AUTHORITY, NOT A DESCRIPTION             #
# =========================================================================== #
# WHY THIS TABLE EXISTS, AND WHY IT DRIVES rather than DOCUMENTS.  Frequency/window
# knowledge used to live at the CALL SITES, with a tuple beside it that merely *claimed* to
# say which metrics get the flow correction.  Three shipped consequences:
#   * `CycleHeat` received NEITHER `nq` NOR `rpy` -- the only metric in the block taking
#     neither -- so a semi-annual filer's self-reference baseline spanned 11.04 years
#     against a quarterly peer's 5.53 off the same ~22.5 rows (~14% of the universe, 31 of
#     57 rendered deck pages);
#   * `EPStoEPSmean` was the LAST uncapped window: its baseline WAS the fetch depth, so at
#     `-nrperiods 80` it became a ~20-year growth penalty;
#   * the old `STAGE2_FLOW_OVER_STOCK` tuple was WRONG -- `incomeQuality` was NOT in it and
#     yet DID receive the per-quarter correction, because `postbm_metric` special-cased the
#     key and applied the factor inside the metric function.  A reader checking "does
#     incomeQuality get the flow correction?" against the tuple got the wrong answer, and
#     that is exactly the reasoning path that let the Stage-1/Stage-2 accruals divergence
#     hide for a month.
# So the table is now the single authority and the code READS it: `flow_factor()` and
# `window_quarters()` below are the only places a per-metric frequency decision is made, and
# `unregistered_metrics()` -- checked once per pool by postBoScoreRanking -- REFUSES the run
# for a metric that has no entry rather than defaulting it silently.  Adding a metric to
# createDicts.getPostDict() without adding it here is a LOUD failure by design.
#
# EVERY window in this module is written for QUARTERLY rows and scaled by `rpy` (rows per
# year: 4 quarterly, 2 semi-annual -- reporting_period).  rpy defaults to 4 everywhere, so a
# caller that passes nothing is BIT-IDENTICAL to the pre-2026-07-25 behaviour; only a source
# classified semi-annual takes a different window.  The scaling is always
# CALENDAR-equivalent: a 4-row YoY becomes 2 rows, a head(16) window becomes head(8).

# --- window basis: WHICH window the metric is defined over ---------------------------
WINDOW_SCORING = 'scoring_nq'        # the ambient scoring window (`nq`, 16 quarters today)
WINDOW_CYCLEHEAT_BASE = 'cycleheat_base_nq'   # CYCLEHEAT_BASE_NQ -- a business cycle
WINDOW_EPS_MEAN_BASE = 'eps_mean_base_nq'     # EPS_MEAN_BASE_NQ -- a business cycle
WINDOW_POINT_IN_TIME = 'point_in_time'        # newest row only (+ a `rpy` YoY lag)
WINDOW_NONE = 'none'                          # not a windowed quantity at all

# --- frequency treatment: WHAT the reporting convention does to the value -------------
# PER_QUARTER is the only one that produces a multiplicative factor (rp.per_quarter_factor:
# x1.0 quarterly -- an exact no-op -- x0.5 semi-annual).  The others are declarations that
# NO factor is correct, each with its own reason, so that "no correction" is a recorded
# decision rather than an omission.
FREQ_PER_QUARTER = 'per_quarter'     # flow / stock -> reads ~2x on a semi-annual filer
FREQ_SCALE_FREE = 'scale_free'       # flow/flow or stock/stock -> the scale cancels
FREQ_YOY_WINDOW = 'yoy_window'       # handled by the `rpy`-row YoY shift itself
FREQ_SELF_NORMALISED = 'self_normalised'      # measured against the name's own history
FREQ_ANNUAL_SUM = 'annual_sum'       # trailing full-year sum inside the metric (Altman)
FREQ_PERIOD_SPAN = 'period_span'     # a per-period SPAN the window cannot fix (priceGrowth)
FREQ_CORRECTED_UPSTREAM = 'corrected_upstream'  # already corrected at ingest
FREQ_NOT_A_TIME_SERIES = 'not_a_time_series'   # pool-level / pass-through, no periods

_FREQ_TREATMENTS = (FREQ_PER_QUARTER, FREQ_SCALE_FREE, FREQ_YOY_WINDOW,
                    FREQ_SELF_NORMALISED, FREQ_ANNUAL_SUM, FREQ_PERIOD_SPAN,
                    FREQ_CORRECTED_UPSTREAM, FREQ_NOT_A_TIME_SERIES)
_WINDOW_BASES = (WINDOW_SCORING, WINDOW_CYCLEHEAT_BASE, WINDOW_EPS_MEAN_BASE,
                 WINDOW_POINT_IN_TIME, WINDOW_NONE)

#  key -> (window basis, frequency treatment, why the frequency treatment is what it is)
#  EVERY key of createDicts.getPostDict() must appear here (test_stage2_registry pins it,
#  and postBoScoreRanking refuses the run otherwise).
STAGE2_METRIC_SPEC = {
    # ---- postBmRankingDict ----------------------------------------------------------
    'RoA':                  (WINDOW_SCORING, FREQ_PER_QUARTER,
                             'netIncome (flow) / totalAssets (stock)'),
    'earnYield':            (WINDOW_SCORING, FREQ_PER_QUARTER,
                             'earnings (flow) / marketCap (stock)'),
    'returnOnEquity':       (WINDOW_SCORING, FREQ_PER_QUARTER,
                             'netIncome (flow) / equity (stock)'),
    'returnOnCapitalEmployed': (WINDOW_SCORING, FREQ_PER_QUARTER,
                             'EBIT (flow) / capital employed (stock)'),
    'incomeQuality':        (WINDOW_SCORING, FREQ_PER_QUARTER,
                             '(CFO - netIncome) (flow) / totalAssets (stock) -- the entry '
                             'the old tuple was MISSING while the code applied the factor '
                             'anyway; the factor is now applied HERE, from this table'),
    'grossProfitMargin':    (WINDOW_SCORING, FREQ_SCALE_FREE, 'flow / flow'),
    'currentRatio':         (WINDOW_SCORING, FREQ_SCALE_FREE, 'stock / stock'),
    'bVpRatio':             (WINDOW_SCORING, FREQ_SCALE_FREE, 'stock / stock'),
    'grahamNumberToPrice':  (WINDOW_SCORING, FREQ_CORRECTED_UPSTREAM,
                             'grahamNumber already uses the frequency-corrected EPS_ttm at '
                             'ingest, and it carries sqrt(2) not 2x -- re-correcting here '
                             'would OVERSHOOT'),
    'revenueGrowth':        (WINDOW_SCORING, FREQ_YOY_WINDOW,
                             'pct_change(-rpy) is one YEAR for either frequency'),
    # ---- postNewRankingDict ---------------------------------------------------------
    'freeCashFlowYield':    (WINDOW_SCORING, FREQ_PER_QUARTER,
                             'FCF (flow) / marketCap (stock)'),
    'freeCashFlowPerShareGrowth': (WINDOW_SCORING, FREQ_YOY_WINDOW,
                             'pct_change(-rpy) is one YEAR for either frequency'),
    'tbVpRatio':            (WINDOW_SCORING, FREQ_SCALE_FREE, 'stock / stock'),
    'priceGrowth':          (WINDOW_SCORING, FREQ_PERIOD_SPAN,
                             'a semi-annual period IS a 6-month move -- a LEVEL difference '
                             'the window cannot fix; w = 0.000, so nothing rests on it'),
    'EPStoEPSmean':         (WINDOW_EPS_MEAN_BASE, FREQ_SELF_NORMALISED,
                             "a deviation from the name's own EPS baseline, divided by "
                             '|that baseline| -- dimensionless; the window carries the '
                             'frequency'),
    'CycleHeat':            (WINDOW_CYCLEHEAT_BASE, FREQ_SELF_NORMALISED,
                             "a z-score against the name's own EPS history -- the window "
                             'IS the correction'),
    'Altman-Z':             (WINDOW_POINT_IN_TIME, FREQ_ANNUAL_SUM,
                             'x3/x5 are TRUE trailing full-year sums over rpy rows, '
                             'because Altman coefficients are absolute'),
    'Piotroski':            (WINDOW_POINT_IN_TIME, FREQ_YOY_WINDOW,
                             'six of nine criteria compare row 0 against row rpy'),
    #  The two Piotroski components EXTRACTED as standalone metrics for the FIN-1
    #  (investment vehicle) vector -- issue E-2, 2026-08-04.  Identical window and
    #  frequency treatment to `Piotroski` itself BY CONSTRUCTION: they read the same two
    #  rows (0 and rpy) that its p7 and p5 read, so a faithful extraction cannot declare
    #  anything else.
    'shareCountChange':     (WINDOW_POINT_IN_TIME, FREQ_YOY_WINDOW,
                             'Piotroski p7 (no new shares) made continuous: row 0 against '
                             'row rpy, one YEAR for either frequency'),
    'longTermDebtChange':   (WINDOW_POINT_IN_TIME, FREQ_YOY_WINDOW,
                             'Piotroski p5 (falling leverage) made continuous: the CHANGE '
                             'in a stock/stock ratio, so scale-free on both sides; the '
                             'rpy-row lag carries the frequency'),
    #  THE S-BLOCK TIER-1 INSTRUMENT (2026-08-06).  Both legs are the SAME PERIOD'S FLOW, so
    #  the ratio is frequency-invariant and takes NO per-quarter factor -- deliberately the
    #  same declaration Stage-1 makes for its own `uInterestCoverage` (createDicts: "BOTH LEGS
    #  ARE THE SAME PERIOD'S FLOW ... it gets NO reporting_period.STAGE1_FLOW_CORRECTION
    #  entry").  The two stages agreeing on the frequency treatment is not cosmetic: they read
    #  the same two fields, and a factor on one side only would make a semi-annual filer's
    #  Stage-1 and Stage-2 coverage readings differ by 2x.
    'interestCoverage':     (WINDOW_SCORING, FREQ_SCALE_FREE,
                             'operatingIncome (flow) / interestExpense (flow), same period -- '
                             'the scale cancels, exactly as in the Stage-1 criterion'),
    #  FIN-1's R-block Tier-1 carrier (2026-08-06).  See scoringWeights D.4 for WHY this is
    #  not the price-to-NAV metric that was asked for.
    'navPerShareGrowth':    (WINDOW_SCORING, FREQ_SCALE_FREE,
                             'a ratio of two book-values-per-share (stock/stock) raised to '
                             '1/years -- the scale cancels and the ANNUALISATION carries the '
                             'frequency, because `years` is derived from the row count and rpy'),
    'marketCapRevQuants':   (WINDOW_POINT_IN_TIME, FREQ_NOT_A_TIME_SERIES,
                             'a POOL-level market-cap quartile code read off the newest row'),
    'DcfToPrice':           (WINDOW_SCORING, FREQ_NOT_A_TIME_SERIES,
                             'a head(nq) mean of a DCF frame with its own cadence, not the '
                             'fundamentals panel; w = 0.000'),
    'BoScore':              (WINDOW_NONE, FREQ_NOT_A_TIME_SERIES,
                             'a straight pass-through of the Stage-1 score; w = 0.000'),
}


def _spec(key):
    """The registry row for `key`, or a LOUD failure.  The whole point of the table."""
    try:
        return STAGE2_METRIC_SPEC[key]
    except KeyError:
        raise KeyError(
            "Stage-2 metric %r has NO entry in stage2_metrics.STAGE2_METRIC_SPEC. Add one "
            "-- declare its WINDOW basis (%s) and its FREQUENCY treatment (%s) -- rather "
            "than letting it default silently: an unregistered metric is how CycleHeat "
            "shipped with no window and EPStoEPSmean shipped uncapped."
            % (key, '/'.join(_WINDOW_BASES), '/'.join(_FREQ_TREATMENTS)))


def flow_factor(key, rpy):
    """The multiplicative frequency correction for `key`, FROM THE REGISTRY.

    rp.per_quarter_factor(rpy) for a flow-over-stock metric (1.0 quarterly = exact no-op,
    0.5 semi-annual); 1.0 for every other declared treatment.  Raises on an unregistered
    key -- see _spec.
    """
    _window, freq, _why = _spec(key)
    if freq not in _FREQ_TREATMENTS:
        raise ValueError('stage2_metrics: %r declares unknown frequency treatment %r'
                         % (key, freq))
    return rp.per_quarter_factor(rpy) if freq == FREQ_PER_QUARTER else 1.0


def window_quarters(key, scoring_nq):
    """The window `key` is defined over, IN QUARTERS, before `rp.scale_window` scales it.

    `scoring_nq` is the caller's ambient scoring window (production passes 16).  A metric
    whose baseline is a business cycle takes its OWN constant instead, which is why the
    fetch depth can no longer decide it.  Returns None for a metric that has no window.
    """
    window, _freq, _why = _spec(key)
    if window == WINDOW_SCORING:
        return scoring_nq
    if window == WINDOW_CYCLEHEAT_BASE:
        return CYCLEHEAT_BASE_NQ
    if window == WINDOW_EPS_MEAN_BASE:
        return EPS_MEAN_BASE_NQ
    if window in (WINDOW_POINT_IN_TIME, WINDOW_NONE):
        return None
    raise ValueError('stage2_metrics: %r declares unknown window basis %r' % (key, window))


#  The window bases that make a metric a WINDOWED metric -- i.e. one whose value is a reduction
#  over a run of fundamentals rows, and which therefore must take the coverage, gappiness and
#  calendar-gap tests.  Kept as a tuple beside the bases themselves so `windowed_metric_keys`
#  cannot fall out of step with the declarations.
_WINDOWED_BASES = (WINDOW_SCORING, WINDOW_CYCLEHEAT_BASE, WINDOW_EPS_MEAN_BASE)


def windowed_metric_keys(spec=None):
    """Every registry key whose value is a reduction over a run of FUNDAMENTALS rows.

    DERIVED FROM THE REGISTRY, NEVER HAND-LISTED, and that is the whole point of the function.
    `test_nan_policy.test_every_windowed_metric_goes_through_the_ONE_seam` used to enumerate seven
    functions by hand -- and OMITTED `eps_to_eps_mean`, which is exactly the metric that had
    silently opted out of the seam.  A test named for enumerating *every* windowed metric that can
    miss one is decorative; this makes the enumeration a property of the declarations instead.

    `FREQ_NOT_A_TIME_SERIES` is excluded, and by a DERIVED rule rather than a carve-out list:
    `DcfToPrice` is declared WINDOW_SCORING but its window runs over a DCF frame with its own
    cadence, not over the fundamentals panel, so there are no period-end dates for a gap test to
    read and no fundamentals rows for coverage to count.  Its registry row already says so.
    """
    spec = STAGE2_METRIC_SPEC if spec is None else spec
    return tuple(k for k, (w, f, _why) in spec.items()
                 if w in _WINDOWED_BASES and f != FREQ_NOT_A_TIME_SERIES)


def structural_lag(key, rpy):
    """Rows at the OLD end of `key`'s series that CANNOT be computable, by arithmetic.

    Read from the registry, never guessed at a call site.  A YoY metric is
    `pct_change(-rpy)`, so its oldest `rpy` rows have no counterpart; `priceGrowth` is
    `pct_change(-1)`, so one row.  Everything else is 0.

    `nan_policy.window_verdict` subtracts this from the COVERAGE DENOMINATOR when those rows
    fall inside the window (which only happens on a source whose panel is shorter than
    window+lag).  Without it, a short history would fail coverage for an ARITHMETIC reason --
    the exact defect the "denominator = rows present" choice exists to avoid, reappearing one
    level down.  Measured: 54 extra `revenueGrowth` sources and 28 extra
    `freeCashFlowPerShareGrowth` sources would be flagged.
    """
    _window, freq, _why = _spec(key)
    if freq == FREQ_YOY_WINDOW:
        return int(rpy)
    if freq == FREQ_PERIOD_SPAN:
        return 1
    return 0


def _reduce(values, key, w, rpy, tempcdx=None, boundary_ok=None,
            scoring_nq=npol.SCORING_WINDOW_NQ):
    """The windowed reduction for `key`, WITH the two-tier NaN policy applied.

    THE ONE SEAM.  Every windowed metric in this module goes through here, so coverage,
    gappiness and boundary imputation are decided in exactly one implementation
    (`nan_policy.window_verdict`) for the live scorer, the certified PIT reproduction and the
    offline tools alike.  Before this existed each metric ended in a bare `.head(w).mean()`,
    which is a reduction that cannot tell "measured over 15 of 16 quarters" from "measured
    over 2 of 16".
    """
    return npol.window_verdict(values, w, key, rpy, tempcdx=tempcdx,
                               structural_lag=structural_lag(key, rpy),
                               scoring_nq=scoring_nq, boundary_ok=boundary_ok)


def unregistered_metrics(keys):
    """Which of `keys` have NO registry entry.  Empty list = the vector is fully declared.

    Called once per pool by postBoScoreRanking, which REFUSES the run when it is non-empty:
    a metric added to getPostDict() without a registry entry must not be scored on a
    silently-defaulted window, which is the defect class this table exists to close.
    """
    return [k for k in keys if k not in STAGE2_METRIC_SPEC]


#  BACKWARDS-COMPATIBLE VIEW, now DERIVED from the registry rather than hand-maintained
#  beside it -- which is what made it wrong (incomeQuality was absent from the tuple while
#  receiving the correction).  Read-only: nothing decides behaviour from this name any more.
STAGE2_FLOW_OVER_STOCK = tuple(k for k, (_w, f, _y) in STAGE2_METRIC_SPEC.items()
                               if f == FREQ_PER_QUARTER)


# --------------------------------------------------------------------------- #
#  Pool-level helper                                                          #
# --------------------------------------------------------------------------- #
def _mcap_for_quants(cdxtop):
    """The market-cap series the size quartiles are cut over: USD where derivable,
    raw marketCap otherwise (audit H-3/H8 fix, 2026-07-19).

    marketCap is stored in each company's REPORTING currency, mixed across the pool, so
    cutting quartiles over the raw field ranked companies partly by which currency they
    report in: a SEK reporter looks ~10x bigger than an equally sized USD reporter and is
    pushed into a larger-cap quartile, an ISK/KRW/JPY reporter far more.  mcapQuants is
    the #2 ranking driver (w = 0.080), so this was a live selection effect, not cosmetic:
    on the 2026-07-17 universe 838 of 7,752 names (10.8%) sit in a DIFFERENT size
    quartile once the field is converted.

    Conversion uses carveOut.marketcap_usd_series with the coarse exchange-suffix
    fallback ON, because the quartiles must produce a value for every name in the pool.
    Unknown suffix -> rate 1.0 = the previous raw behaviour, so no name is lost.  Falls
    back to the raw column entirely if carveOut is unavailable (offline tooling) or the
    conversion yields nothing, so this can never make the metric MISSING.
    """
    raw = pd.to_numeric(cdxtop["marketCap"], errors="coerce")
    try:
        import carveOut as _co
        usd = _co.marketcap_usd_series(cdxtop, allow_suffix_fallback=True)
        usd = pd.to_numeric(usd, errors="coerce")
        if usd.notna().any():
            return usd.where(usd.notna(), raw)
    except Exception:
        pass
    return raw


#  mcapQuants value for a row whose market cap is NOT KNOWN: the NEUTRAL midpoint of the
#  metric's own [-0.5 .. +0.5] range, matching the Stage-2 missing-data convention used
#  everywhere else (normalizeAndDropNA maps a NaN metric to z = 0 = pool-neutral).
MCAP_QUANT_MISSING = 0.0


def add_mcap_quants(cdxtop):
    """Pool-level marketCap quartile code, mapped to [-0.5 .. +0.5] with the
    sign flipped so SMALLER caps score HIGHER (set by
    postBoRank._compute_ticker_metrics as `marketCapRevQuants`).

    Cut over the USD market cap (see _mcap_for_quants), NOT the mixed-currency raw
    field.

    A row with NO market cap scores MCAP_QUANT_MISSING = 0.0, i.e. neutral (fix,
    2026-07-25).  ``pd.qcut`` assigns ``cat.codes == -1`` to NaN, and -1 fell straight
    through the ``(-1) * (code/3 - 0.5)`` mapping to **+0.8333** -- OUTSIDE the metric's
    intended range and, because smaller caps score higher, BETTER than the
    most-rewarded real bucket (the smallest-cap quartile at +0.5) by 0.333.  Missing
    market cap therefore earned the maximum small-cap reward in a w=0.080 metric (the
    #2 ranking driver).  Verified on the 2026-07-17 panel: exactly the 746 rows with no
    market cap carried +0.8333, and `cat.codes == -1` coincided with them one-for-one.
    It is the same defect class as the EPStoEPSmean `return 0` sentinel and the Montier
    `fillna(99999)`: absent data scoring as a real, favourable value.

    Uses ``duplicates='drop'`` + a 0.0 fallback so a pool with coincident
    quartile edges degrades gracefully rather than raising.  (NOTE, pre-existing and
    unchanged: when `duplicates='drop'` collapses the pool to fewer than 4 bins the
    codes no longer span 0..3, so the mapping does not cover the full [-0.5, +0.5];
    that path only triggers on a degenerate pool.)
    """
    try:
        codes = pd.qcut(_mcap_for_quants(cdxtop), 4,
                        duplicates="drop").cat.codes
        vals = (-1) * ((codes / 3) - 0.5)
        # codes < 0 is qcut's NaN sentinel -- NOT a quartile. Neutral, not best-in-pool.
        return vals.where(codes >= 0, MCAP_QUANT_MISSING)
    except Exception:
        return pd.Series(0.0, index=cdxtop.index)


# --------------------------------------------------------------------------- #
#  postBmRankingDict metrics                                                  #
# --------------------------------------------------------------------------- #
#  Minimum book equity for any ratio that DIVIDES by it.  Negative equity does not make a
#  return ratio bad -- it makes it INVERT, so a loss over a deficit reads as a high return.
#  Mirrors _IQ_MIN_ASSETS in role; `> 0` because zero equity is division by zero.
_ROE_MIN_EQUITY = 0.0


def postbm_metric(key, met, tempcdx, nq, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """One postBmRankingDict metric for a single ticker (postBoRank._compute_ticker_metrics).

    grahamNumberToPrice / returnOnEquity / bVpRatio / revenueGrowth / incomeQuality are
    special-cased in the ARITHMETIC; every other key is the head(window) mean of its ``eqMet``
    column.

    THE WINDOW AND THE FREQUENCY FACTOR BOTH COME FROM THE REGISTRY, not from this function:
    `window_quarters(key, nq)` and `flow_factor(key, rpy)`.  That is the point of the table
    -- there is no branch here that can apply a correction the table does not declare, and
    no key can reach the arithmetic without an entry (`_spec` raises).  `incomeQuality` used
    to get its factor INSIDE income_quality_accruals, which is why the tuple that was
    supposed to list the corrected metrics did not list it.

    The window is scaled to `rpy` so the average spans the same CALENDAR time for a
    semi-annual filer, and revenueGrowth's YoY shift is `rpy` rows (4 quarters OR 2 halves)
    instead of a hard-coded 4 -- on a semi-annual name a 4-row shift measured TWO-year
    growth and called it annual.
    """
    _wq = window_quarters(key, nq)
    if _wq is None:
        # The registry says this metric has NO window (point-in-time / pass-through), so it
        # cannot be computed as a head(w) mean.  Loud, because the silent alternative --
        # rp.scale_window(None, rpy) raising a bare TypeError deep in the arithmetic, or worse
        # a defaulted window -- is the failure mode the registry exists to remove.
        raise ValueError(
            "stage2_metrics.postbm_metric was asked for %r, which STAGE2_METRIC_SPEC declares "
            "as window basis %r -- it has no averaging window, so it cannot be computed here. "
            "Either it belongs in the postNewRankingDict block with its own function, or its "
            "registered window basis is wrong." % (key, _spec(key)[0]))
    w = rp.scale_window(_wq, rpy)
    ff = flow_factor(key, rpy)          # x1.0 unless the registry says flow-over-stock
    if key == "grahamNumberToPrice":
        #  THE ONE TYPE-D COLUMN THAT TAKES A BOUNDARY (nan_policy.BOUNDARY_LIMIT).  Fully
        #  undefined over the window AND undefined because an input was ADVERSE -> the limit
        #  0.0, which is this metric's floor.  Undefined because the INPUTS WERE MISSING ->
        #  refused (NaN -> column median).  `grahamUndefinedReason` is what tells the two
        #  apart, and it is 99.1% adverse / 0.9% gap on this panel -- which is the whole
        #  empirical case for treating "undefined" and "missing" as different objects.
        #  PARTIAL coverage keeps its own observations and is NOT collapsed (see
        #  window_verdict): a name with 4 of 16 computable Graham quarters WAS profitable four
        #  times.
        return _reduce(tempcdx["grahamNumber"] / tempcdx["price"], key, w, rpy,
                       tempcdx=tempcdx, boundary_ok=npol.graham_adverse_mask(tempcdx),
                       scoring_nq=nq) * ff
    elif key == "returnOnEquity":
        # SIGN-INVERSION GUARD (2026-08-04).  ROE = netIncome/equity, so a NEGATIVE equity with
        # a NEGATIVE net income gives a POSITIVE, often LARGE, return on equity -- the same
        # denominator-sign defect that `incomeQuality` was rewritten for (see
        # income_quality_accruals).  Stage-1 carries this criterion too and refuses the same
        # rows; this is the Stage-2 half of one fix, and the two must not diverge.
        #
        # MEASURED on Stage-1's head(8) window: the netIncome<0 / equity<0 cell passes the 12%
        # hurdle on 2,255 of 2,426 rows (92.95%), while netIncome>0 / equity<0 passes 0 of
        # 1,235 -- so the perverse group is essentially the whole negative-equity population.
        #
        # THE ROW-MASK IDIOM IS income_quality_accruals's, DELIBERATELY: mask the inadmissible
        # denominator to NaN, let the window mean skip it, and a name with NO admissible row
        # comes out NaN -> z = 0 = pool-neutral, never a real score.  Two consequences a
        # reviewer should weigh rather than take on trust:
        #  * a name with SOME negative-equity quarters is now scored on its admissible ones,
        #    which is mildly adversely selected -- the same objection that applies to
        #    incomeQuality today.  A stricter "refuse the whole name" rule is defensible; it
        #    would be a NEW convention, and inventing a third one here is worse than matching
        #    the existing one.
        #  * a fully-refused name lands at the column median rather than at a FLOOR, and the
        #    median sits ABOVE the single-negative band -- so strict "double negative ranks
        #    below every single negative" is NOT achieved here.  Deferred, and the reason is
        #    SCOPE, not impossibility: an earlier note claimed no finite floor exists because
        #    the single-negative band is unbounded below in the metric's own units.  That is
        #    true of the RAW units and irrelevant, because the pipeline ranks in SQUASHED z,
        #    which is BOUNDED -- `zeta = -squash_k` is a perfectly stable, pool-INDEPENDENT
        #    floor.  What is actually missing is a REFUSED channel distinct from MISSING:
        #    normalizeAndDropNA replaces +-inf with NaN before the ruler runs
        #    (postBoRank.py, the `df_clean[col].replace([np.inf, -np.inf], np.nan)` line), so a
        #    sentinel cannot reach the squash, and adding that channel is boundary-imputation
        #    work touching the shared normaliser -- a different, open item.
        #    Measured proportionality: 0 of 100 names in the deployed pool have ANY
        #    double-negative ROE row, so the floor changes nothing today.
        # Either way this is a strict improvement on the status quo, which scored those names
        # ABOVE the median on a number built from two negatives.
        _eq = pd.to_numeric(tempcdx["totalStockholdersEquity"], errors="coerce")
        _roe = pd.to_numeric(tempcdx[met], errors="coerce").where(_eq > _ROE_MIN_EQUITY)
        #  THE ADVERSE SELECTION THE COMMENT ABOVE FLAGS IS NOW PRICED, and by the general rule
        #  rather than by a carve-out for this metric.  The masked rows are non-computable rows,
        #  so `nan_policy`'s coverage test sees them: a name with FEWER THAN HALF its window
        #  admissible is no longer scored on its selected good quarters -- it goes to the column
        #  median.  MEASURED: 338 sources (4.37%) [universe], 0 [pool].  The "mildly adversely
        #  selected" residual is therefore bounded at coverage >= 0.50 rather than unbounded.
        return _reduce(_roe, key, w, rpy, tempcdx=tempcdx, scoring_nq=nq) * ff
    elif key == "bVpRatio":
        return _reduce(1 / tempcdx[met], key, w, rpy, tempcdx=tempcdx, scoring_nq=nq) * ff
    elif key == "revenueGrowth":
        return _reduce(tempcdx[met].pct_change(-int(rpy), fill_method=None),
                       key, w, rpy, tempcdx=tempcdx, scoring_nq=nq) * ff
    elif key == "incomeQuality":
        return income_quality_accruals(tempcdx, nq, rpy=rpy) * ff
    else:
        return _reduce(tempcdx[met], key, w, rpy, tempcdx=tempcdx, scoring_nq=nq) * ff


# --------------------------------------------------------------------------- #
#  incomeQuality -- SIGN-SAFE, SCALE-FREE  (audit D2, fixed 2026-08-01)        #
# --------------------------------------------------------------------------- #
# Minimum totalAssets for the denominator.  TA is a positive stock for any going concern;
# this only rejects the degenerate rows (0.35% of the universe panel are exactly 0, none
# negative), so it is a guard, not a filter.
_IQ_MIN_ASSETS = 0.0


def income_quality_accruals(tempcdx, nq, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """Earnings quality as CASH-BACKING OF EARNINGS, scaled by assets:

        mean over the window of   (CFO - netIncome) / totalAssets

    HIGH = operating cash flow exceeds accounting earnings = low accruals = GOOD.

    WHY THE RATIO IT REPLACES WAS BROKEN (audit D2).  Stage-2 weighted FMP's `incomeQuality`
    = CFO / netIncome at w = +0.072.  That is a ratio whose DENOMINATOR CHANGES SIGN, so it
    INVERTS for loss-makers, and it EXPLODES as NI -> 0.  Stage-1 got the sign-safe treatment
    in July (`CFOlessEarnings = CFO - NI`, createDicts.BoMetric_special_dict); Stage-2 was
    missed and kept the ratio.  Measured on the shipped 2026-07-17 top-100 panel, by quadrant
    (MEDIANS -- the means are meaningless because the ratio explodes):
        NI>0 CFO>0  healthy                     n=1827   median  +1.625
        NI>0 CFO<0  profit but no cash          n= 195   median  -1.078   (correctly bad)
        NI<0 CFO>0  loss but CASH-GENERATIVE    n= 184   median  -1.987   <-- the GOOD case,
                                                                              PENALISED hardest
        NI<0 CFO<0  loss AND burning cash       n=  85   median  +0.489   <-- the BAD case,
                                                                              REWARDED
    A company losing money while generating cash is the single clearest "earnings understate
    the business" signal there is, and the shipped metric ranked it BELOW a company losing
    money AND burning cash.  The per-name metric spanned -19.78 .. +190.41 on that pool.

    THE FIX, MEASURED PER SOURCE ON THE FULL 7,729-SOURCE UNIVERSE (trailing-year NI/CFO
    quadrants; the shipped top-100 is 99/100 profitable, so the inversion is invisible there
    and only bites in the carve-out cohort pools and any future pool holding loss-makers):
        quadrant                              n      OLD median      NEW median
        NI>0 CFO>0  healthy                 4596        +1.392         +0.0087
        NI>0 CFO<0  profit but no cash       322        +0.069         -0.0054
        NI<0 CFO>0  loss but CASH-GEN        1100        +0.128        +0.0222
        NI<0 CFO<0  loss AND burning        1575        +0.742         +0.0266
    THE CONSTRAINT THAT MATTERS IS MET: the GOOD loss-maker moves from far BELOW the healthy
    median (+0.128 vs +1.392, i.e. heavily penalised at w=+0.072) to ABOVE it (+0.0222 vs
    +0.0087, i.e. rewarded).  And "profit but no cash" is the only quadrant that goes negative,
    which is the one that should.

    SEMANTICS -- READ THIS BEFORE CALLING THE LAST ROW A RESIDUAL INVERSION.  The cash-BURNING
    loss-maker still edges the cash-GENERATING one (+0.0266 vs +0.0222).  That is correct by
    construction and is the SAME accepted semantics as the Stage-1 form (see
    createDicts.BoMetric_special_dict, which states it explicitly): this is an EARNINGS-QUALITY
    (accruals) test, NOT a profitability test.  A company with a large non-cash writedown
    (NI very negative, CFO mildly negative) genuinely does have earnings that understate its
    cash, and that is what this metric is asking.  Profitability is scored separately and by
    other metrics in this same vector (RoA, earnYield) and at Tier S in Stage-1.  Ordering
    loss-makers by profitability is not this metric's job, and building that in would
    re-introduce a second quantity into a metric that finally measures one thing.

    WHY totalAssets AND NOT REVENUE (the denominator must never approach or cross zero, or the
    explosion is simply rebuilt with a new fuse).  Measured over the full 176,781-row panel:
        totalAssets :  0 negative rows,  613 zero rows (0.35%),  1 NaN
                       -> and ZERO sources have all-nonpositive assets
        revenue     :  1,307 NEGATIVE rows,  10,477 zero rows (6.67% non-positive)
                       -> and 244 SOURCES are entirely zero-revenue
    Revenue can be legitimately zero (pre-revenue biotech, a shell, a holding company) and can
    even go NEGATIVE (contra-revenue / refunds), which would re-introduce the exact
    sign-flipping-denominator defect being fixed.  Total assets cannot go negative -- a
    NEGATIVE-EQUITY company still has positive assets, because equity, not assets, is what
    goes negative -- and no source in the universe lacks them entirely.  It is also the
    denominator the accruals literature uses (Sloan 1996).

    THEY DO NOT "AGREE ON BASIS BY CONSTRUCTION" WITH forensicFlags.sloanAccruals -- an
    earlier version of this docstring claimed that, and it is FALSE (corrected 2026-08-02,
    re-read off forensicFlags.buildSloanAccruals).  The two share only the DENOMINATOR
    VARIABLE, `totalAssets`, and differ on every other axis a basis is made of:
      * numerator period : PER-PERIOD (CFO - NI) here, TTM sums over `rpy` rows there;
      * denominator      : the POINT-IN-TIME totalAssets of the same row here, the AVERAGE
                           of the closing and one-year-earlier levels there;
      * reduction        : a mean over the scoring window here, the single most recent value
                           there;
      * SIGN             : OPPOSITE.  This metric is (CFO - NI), high = GOOD; sloanAccruals
                           is (NI - CFO), high = more accruals = BAD.
    So they are two different quantities that happen to be scaled by the same field name, and
    a reader who took the "agree by construction" line at face value would have read their
    signs the same way round.  Nothing in this function depends on them agreeing -- the claim
    was load-bearing only as reassurance, which is exactly why a wrong one is worse than none.

    A zero/NaN/negative TA row yields NaN, which normalizeAndDropNA maps to z = 0 = neutral --
    the Stage-2 convention for "not computable", never a real score.

    SIGN: unchanged at w = +0.072, and this is VERIFIED, not assumed.  The replaced ratio was
    high-is-good (for a profitable firm, more cash per unit of earnings is better) and so is
    this: CFO - NI > 0 means earnings are more than fully cash-backed.  On the healthy
    NI>0/CFO>0 population the two quantities are positively rank-correlated, so +0.072 keeps
    its meaning on the names where the old metric was not inverted.  No flip is applied.

    SCALE: (flow - flow) / stock, so it is a flow/stock ratio and takes the same per-quarter
    normalisation as earnYield / RoA -- a semi-annual filer's six-month CFO and NI over a
    point-in-time asset base would otherwise read ~2x a quarterly peer's.  x1.0 for quarterly,
    so the quarterly path is unchanged by the frequency correction itself.

    THAT FACTOR IS APPLIED BY THE CALLER, FROM THE REGISTRY -- `postbm_metric` multiplies by
    `flow_factor('incomeQuality', rpy)`, and STAGE2_METRIC_SPEC declares this metric
    FREQ_PER_QUARTER.  It is deliberately NOT applied inside this function any more (moved
    2026-08-02): while it lived here, `incomeQuality` was absent from the table that was
    supposed to list every flow-corrected metric, so the table said the opposite of what the
    code did.  A DIRECT caller of this function therefore gets the UNCORRECTED per-period
    value -- which is the honest thing for a function that takes no view on frequency
    treatment -- and must apply `flow_factor` itself if it wants the scored quantity.

    WEIGHT PROVENANCE, stated because it is a real caveat and not a defect to hide: 0.072 was
    fitted against the RATIO.  It is therefore a weight INHERITED by a different quantity.
    Re-fitting is a separate, unauthorised exercise; this is a known and accepted consequence
    of correcting the metric without re-running the weight fit.
    """
    ni = pd.to_numeric(tempcdx["netIncome"], errors="coerce")
    cfo = pd.to_numeric(tempcdx["netCashProvidedByOperatingActivities"], errors="coerce")
    ta = pd.to_numeric(tempcdx["totalAssets"], errors="coerce")
    ta = ta.where(ta > _IQ_MIN_ASSETS)          # 0 / negative / NaN -> NaN, never a divisor
    val = (cfo - ni) / ta
    #  Reduced through the shared NaN-policy seam, so the TA-masked rows count against coverage
    #  exactly as the ROE-masked ones do.  MEASURED: coverage < 0.50 on 6 sources (0.08%)
    #  [universe] / 0 [pool]; >= 2 interior gaps on 47 (0.61%) / 0.
    return _reduce(val, 'incomeQuality', rp.scale_window(nq, rpy), rpy,
                   tempcdx=tempcdx, scoring_nq=nq)


# --------------------------------------------------------------------------- #
#  postNewRankingDict metrics                                                 #
# --------------------------------------------------------------------------- #
def free_cash_flow_yield(tempfcf, tempmcap, nq, rpy=rp.DEFAULT_ROWS_PER_YEAR,
                         tempcdx=None):
    """FCF / marketCap, head(nq) mean (call site: postBoRank._compute_ticker_metrics).

    FLOW/STOCK: a semi-annual row's FCF is a 6-month flow over a point-in-time market
    cap, so the per-row yield reads ~2x a quarterly peer's.  The value is z-scored
    downstream, so the correction only needs to fix the SEMI-ANNUAL/QUARTERLY ratio and is
    applied on a common per-quarter basis (x1 quarterly = no-op).  Ruling applied
    2026-07-25; the previous note here said this was deferred.

    `tempcdx` is OPTIONAL and ADDITIVE (2026-08-05).  It is the only thing this function
    needs for the NAME-level calendar-gap test, and it is optional purely so the five existing
    call sites keep working unchanged; the two SCORING paths (postBoRank and
    baseline_tools/stage2_pit) DO pass it, and a test pins that they do.  Coverage and
    interior-gap tests apply either way -- they read the series, not the dates."""
    return _reduce(tempfcf / tempmcap, 'freeCashFlowYield', rp.scale_window(nq, rpy), rpy,
                   tempcdx=tempcdx, scoring_nq=nq) * rp.per_quarter_factor(rpy)


def free_cash_flow_per_share_growth(tempfcf, tempshares, nq,
                                    rpy=rp.DEFAULT_ROWS_PER_YEAR, tempcdx=None):
    """YoY growth of FCF-per-share over `rpy` rows, head(nq) mean
    (call site: postBoRank._compute_ticker_metrics).

    `rpy` rows back is one YEAR for either frequency; the hard-coded 4 was two years for
    a semi-annual filer.  `tempcdx` -- see free_cash_flow_yield.

    THE STRUCTURAL LAG MATTERS MOST HERE.  `pct_change(-rpy)` leaves the oldest `rpy` rows NaN
    by arithmetic, and `structural_lag` takes them out of the coverage denominator so a short
    panel is not failed for a gap it cannot have.  MEASURED: coverage < 0.50 on 143 sources
    (1.85%) [universe] / 0 [pool] with the correction, 171 without it."""
    fcfps = tempfcf / tempshares
    return _reduce(fcfps.pct_change(-int(rpy), fill_method=None),
                   'freeCashFlowPerShareGrowth', rp.scale_window(nq, rpy), rpy,
                   tempcdx=tempcdx, scoring_nq=nq)


def interest_coverage(tempcdx, nq, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """operatingIncome / interestExpense, head(nq) mean -- the S block's Tier-1 instrument
    (call site: postBoRank._compute_ticker_metrics).

    "Can one period's operating profit cover one period's interest bill?"  This is the
    question a STOCK leverage ratio cannot answer, and until 2026-08-06 the Stage-2 solvency
    block had no direct instrument for it at all -- `currentRatio` is a liquidity proxy and
    `Altman-Z` is a 1968 discriminant fitted on US manufacturers.

    Both legs are the SAME PERIOD'S FLOW, so the ratio is frequency-invariant: it takes no
    per-quarter factor, only the averaging window scales.  That matches Stage-1's own
    `uInterestCoverage`, which deliberately has no STAGE1_FLOW_CORRECTION entry.

    THE GUARD IS THE SUBSTANTIVE HALF, and it is the same one Stage-1 makes (calcMetrics):
    **rows with `interestExpense <= 0` are REFUSED (NaN), not scored.**  FMP reports 0 for a
    DEBT-FREE name, so dividing would give +/-inf -> a debt-free company would be marked down
    or up on an arithmetic artifact of having no debt.  Refusing the row hands the leverage
    question to the rest of the block instead.  A NEGATIVE interestExpense (a net-interest-
    income presentation, which is why this metric is out of domain for FIN-3 altogether) is
    refused on the same line.

    A negative `operatingIncome` over a positive interest bill is NOT refused: an operating
    loss that cannot service the debt is a real and adverse reading, and it is the exact
    reading this block exists to catch.
    """
    oi = pd.to_numeric(tempcdx["operatingIncome"], errors="coerce")
    ie = pd.to_numeric(tempcdx["interestExpense"], errors="coerce")
    #  refuse the row rather than divide: see the guard note above
    ie = ie.where(ie > 0)
    return _reduce((oi / ie).replace([np.inf, -np.inf], np.nan),
                   'interestCoverage', rp.scale_window(nq, rpy), rpy,
                   tempcdx=tempcdx, scoring_nq=nq)


def nav_per_share_growth(tempcdx, nq, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """Annualised growth of BOOK VALUE PER SHARE across the scoring window -- FIN-1's R-block
    Tier-1 carrier (call site: postBoRank._compute_ticker_metrics).

        (BVPS_newest / BVPS_oldest) ** (1 / years) - 1,   years = (rows - 1) / rpy

    **THIS IS A PROXY, AND THE NAME OVERSTATES IT.**  The NAV leg is GAAP
    `bookValuePerShare`, which EQUALS net asset value per share only under US
    investment-company accounting (ASC 946, where the portfolio is carried at fair value).
    For every other vehicle in the cohort it is an APPROXIMATION whose error is whatever the
    balance sheet carries at cost rather than at fair value.  No endpoint this pipeline
    fetches publishes a real NAV, so no exact form is available; do not read, quote or render
    this column as a fund-published NAV.  (reviewReference.METRIC_BASIS repeats the caveat on
    the review page, which is the artifact the CEO actually reads.)

    WHAT IT MEASURES, AND WHY IT IS NOT `bVpRatio` AGAIN.  It is deliberately NOT the
    price-to-NAV DISCOUNT -- that level is already carried by `bVpRatio` (P-A), and a
    discount-persistence column measures rho = +0.806 against it.  This asks whether the NAV
    is REAL: a stated NAV that is an accounting artifact fails to COMPOUND, whatever the
    discount to it.  Measured on the 88-name FIN-1 cohort: computable on 87, and
    rho = -0.277 against `bVpRatio`.  Full reasoning in scoringWeights D.4.

    REFUSED (NaN) when either endpoint is absent or <= 0.  A non-positive book value makes the
    ratio meaningless rather than merely large -- and for a vehicle whose thesis IS its book
    value, a negative one is a different object, not a bad score.  Fractional powers of a
    negative base are also undefined, so there is nothing to compute.  NaN then imputes to the
    column MEDIAN like every other Stage-2 metric; it is never 0.0, which on a positive-weight
    column would assert "did not compound" -- a judgement made from missing data.

    Both legs are per-share stocks, so the ratio is scale-free and currency-invariant; the
    ANNUALISATION carries the frequency, because `years` is derived from the row count and
    `rpy` (a 16-row quarterly window and an 8-row semi-annual one are both 4 years).
    """
    try:
        bvps = pd.to_numeric(tempcdx["bookValuePerShare"], errors="coerce")
        bvps = bvps.replace([np.inf, -np.inf], np.nan)
        w = rp.scale_window(nq, rpy)
        #  POLICY GATE, the `eps_to_eps_mean` idiom: this metric is an ENDPOINT PAIR, not a
        #  reduction, so it cannot be expressed through `_reduce` -- but it IS a registered
        #  windowed metric and must therefore take the same NAME-level calendar-gap refusal as
        #  every other one.  Only the NaN-ness of `_gate` is used; its value (the mean BVPS
        #  over the window) is not the metric.
        _gate = _reduce(bvps, 'navPerShareGrowth', w, rpy, tempcdx=tempcdx, scoring_nq=nq)
        if pd.isna(_gate):
            return np.nan
        if w and len(bvps) > w:
            bvps = bvps.head(w)
        rows = len(bvps)
        if rows < 2:
            return np.nan
        newest = _finite(bvps.iloc[0])
        oldest = _finite(bvps.iloc[rows - 1])
        if newest is None or oldest is None or newest <= 0 or oldest <= 0:
            return np.nan
        years = (rows - 1) / float(rpy)
        if years <= 0:
            return np.nan
        return (newest / oldest) ** (1.0 / years) - 1.0
    except Exception:
        return np.nan


def tbv_p_ratio(tempcdx, nq, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """Tangible book value per share / price, head(nq) mean
    (call site: postBoRank._compute_ticker_metrics).
    Both inputs are point-in-time STOCKS, so only the averaging window scales."""
    return _reduce(tempcdx["tangibleBookValuePerShare"] / tempcdx["price"],
                   'tbVpRatio', rp.scale_window(nq, rpy), rpy,
                   tempcdx=tempcdx, scoring_nq=nq)


# Minimum |mean EPS| for the EPStoEPSmean denominator, as a FRACTION of the mean
# ABSOLUTE EPS over the same window.  Below this the mean is a near-cancellation of
# a swinging EPS series (mean ~ 0 with large |EPS|), so the ratio explodes on
# arithmetic rather than on economics -> NaN (= neutral) instead of a huge number.
EPS_MEAN_FLOOR_FRAC = 0.01


#  EPStoEPSmean's BASELINE window, in QUARTERS, scaled to the filer's own frequency by
#  rp.scale_window -- the same treatment CYCLEHEAT_BASE_NQ gives CycleHeat, and for the
#  same reason.
#
#  WHY A CAP EXISTS AT ALL (fix, 2026-07-31).  This was the LAST uncapped window in the
#  Stage-2 block: `epsmean = eps.mean()` took no `nq` and averaged the WHOLE per-ticker
#  panel, and `cdx_dftop100` has no row cap (postBo.py), so the baseline length WAS the
#  fetch depth.  The metric is (epsmean - ewma_recent) / |epsmean| with w = +0.056, so
#  POSITIVE = recent EPS below the stock's own history = REWARDED.  A longer baseline drags
#  `epsmean` toward older, smaller earnings, which makes the numerator more negative for any
#  company whose EPS has GROWN -- i.e. the longer the fetch, the harder a grower is
#  penalised.  Measured on the shipped 2026-07-17 top-100 (90 names defined on both windows),
#  full 24-row panel vs the most recent 12 rows:
#      mean  -1.317 -> -0.623      std 2.715 -> 1.291      spearman 0.785
#      14 of 90 names CHANGE SIGN, i.e. flip from penalised to rewarded or back
#      rising-EPS names (n=78): -1.461 on the full panel vs -0.533 on the recent half
#      falling-EPS names (n=9):  -0.004 on the full panel vs -1.159 on the recent half
#  So the full-panel baseline is inverted relative to the metric's own thesis: it punishes
#  growth and spares decline.  At `-nrperiods 80` the baseline becomes ~20 YEARS and the
#  metric stops being a mean-reversion/trough detector at all -- it becomes a 20-year growth
#  penalty.  The cap makes the baseline a fixed CALENDAR span regardless of fetch depth.
#
#  WHY 28 QUARTERS -- deliberately the SAME VALUE as CYCLEHEAT_BASE_NQ, chosen by the same
#  two constraints: long enough to contain a full business cycle (the history a
#  mean-reversion baseline needs), and >= the ~24 rows a quarterly filer carries on the
#  CURRENT panel so it CANNOT BIND there -- making this change BIT-IDENTICAL for every
#  quarterly name today, which is the regression that matters.  It is kept as its OWN
#  constant rather than an alias of CYCLEHEAT_BASE_NQ: these are two different metrics with
#  two different baselines, and one must not move silently when the other is retuned.
#
#  DECLARED IN THE REGISTRY as STAGE2_METRIC_SPEC['EPStoEPSmean'] -> WINDOW_EPS_MEAN_BASE, and
#  `window_quarters('EPStoEPSmean', nq)` returns THIS constant.  The registry decides WHICH
#  window basis the metric uses; this constant is the VALUE of that basis and keeps the
#  reasoning above.  test_stage2_registry pins the two together so the default below cannot
#  drift from the declaration.
EPS_MEAN_BASE_NQ = 28


def eps_to_eps_mean(tempcdx, nq=EPS_MEAN_BASE_NQ, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """Exponentially-weighted recent EPS vs its baseline-window mean, expressed as a
    FRACTION OF THAT MEAN (call site: postBoRank._compute_ticker_metrics).

      (epsmean - ewma_recent_eps) / |epsmean|

    Positive = the most recent year sits BELOW the stock's own EPS history (the
    mean-reversion side the +0.056 weight is betting on); negative = above.

    `epsmean` -- the BASELINE -- is the mean over the most recent
    `scale_window(nq, rpy)` rows, NOT over the whole panel (fix 2026-07-31; see
    EPS_MEAN_BASE_NQ for the measured defect).  Both the baseline and the
    EPS_MEAN_FLOOR_FRAC scale are computed over that same window, so the floor test
    stays a comparison of two quantities on one window.  `nq` is in QUARTERS and
    `rpy` scales it to the filer's own frequency, so the baseline spans the same
    CALENDAR time for a semi-annual filer as for a quarterly one -- and, crucially,
    the same span whatever `-nrperiods` the fetch used.

    DIMENSIONLESS (audit C4 fix, 2026-07-19).  The numerator alone is in currency
    per share, so the metric used to be a mixed-currency PRICE/EPS-LEVEL ranking,
    not a deviation measure: cross-sectionally z-scoring it ranked a KRW-quoted
    name (SKHY, ~807,000/share) far above every USD name for arithmetic reasons.
    Measured on the shipped 2026-07-17 top-100: spearman(|metric|, share price) =
    0.642 (pearson 1.00, driven by SKHY alone; 0.786 excluding it) BEFORE, and
    -0.058 after dividing by |epsmean|.  A share split -- which changes nothing
    economically -- used to move this metric by the split factor; now it does not.

    NaN, NOT 0, when the metric is undefined (audit C4, second limb).  The old
    `return 0` sentinel was a RAW zero that then z-scored to (0 - poolmean)/std,
    i.e. a real, pool-dependent, non-neutral score for "not computable".  NaN is
    the honest answer and normalizeAndDropNA maps it to z = 0 = genuinely neutral,
    which is the Stage-2 convention for missing data everywhere else.

    NO BOUNDARY IMPUTATION HERE -- THIS COLUMN IS *REFUSED*, DELIBERATELY (2026-08-05).
    nan-policy.md ADDENDUM A2 assigns this metric a boundary of -1.0.  That is wrong twice
    over, and both reasons are recorded in `nan_policy.REFUSED_NOT_IMPUTED['EPStoEPSmean']`:
      * THE SPEC'S LIMIT IS THE LIMIT OF THE SIGN-REVERSED FORMULA.  It writes
        "(EPS - mean)/abs(mean)"; the shipped formula below is (mean - recent)/|mean|.
      * TAKE THE ACTUAL LIMIT AND IT IS NOT A FLOOR.  Driving the non-positive recent EPS to
        0+ sends the EWMA term to 0, leaving epsmean/|epsmean| = +1 whenever epsmean > 0 --
        and +1.0 is this metric's OBSERVED MAXIMUM, i.e. the most-REWARDED value (w = +0.0516).
        Measured over the 3,888 sources that hit the positivity gate: the limit is not even a
        constant (min -275.79, median -1.00, max +1.00) and is positive for 25.8% of them.
      ADDENDUM A1 admits a boundary only where the limit is FINITE *and* the metric's WORST
      admissible value, so its own escape clause fires: REFUSE.
    AND REFUSING IS ALSO WHAT THE CEO'S INSTRUCTION WANTS, which is the part worth reading
    twice: this NaN was never a punishment.  The positivity gate exists to stop a LOSS-MAKER
    collecting the maximum mean-reversion reward, so imputing at any boundary would either
    hand it that reward (+1) or invent a fresh punishment (-1) -- and "I don't think we should
    punish them again for it" rules out the second.  The column median is the neutral answer
    and it is what a NaN already gets.
    """
    eps = tempcdx["netIncome"] / tempcdx["weightedAverageShsOut"]
    eps = eps.replace([np.inf, -np.inf], np.nan)
    #  POLICY GATE (added 2026-08-05 -- REVIEW FINDING).  This function did NOT go through the
    #  shared seam, so it took NO calendar-gap test at all: on a frame with two filing stoppages
    #  `RoA` and `CycleHeat` came back NaN and this metric came back with a real value.  Three
    #  places asserted the property the code lacked (`_reduce`'s "THE ONE SEAM" docstring,
    #  `nan_policy.SCORING_WINDOW_NQ`'s "falls back to this", and a test that enumerated seven
    #  functions by hand and omitted this one).  The enumeration is now DERIVED from the registry
    #  (`windowed_metric_keys`) so it cannot omit a metric again.
    #  Only the NaN-ness of `_gate` is used; its value (the mean EPS over the baseline window) is
    #  not the metric.  This column is TYPE-D, so `window_verdict` deliberately does NOT apply the
    #  coverage or interior-gap tests to it -- a partial Graham/EPS history keeps its own
    #  observations (ADDENDUM A's closing clause).  What the gate adds here is the NAME-level
    #  calendar-gap refusal, which is not Type-D-conditional: a company that stopped filing twice
    #  has no trustworthy window for ANY windowed metric.
    _gate = _reduce(eps, 'EPStoEPSmean', rp.scale_window(nq, rpy), rpy, tempcdx=tempcdx)
    if pd.isna(_gate):
        return np.nan
    # BASELINE TRUNCATION.  tempcdx is NEWEST-FIRST (postBoRank._sort_cdx_newest_first, and
    # the offline PIT loop sorts the same way), so head() keeps the MOST RECENT rows -- the
    # same end of the series the EWMA below reads via iloc[0:rpy].  Truncating by ROW (no
    # dropna) rather than by observation is deliberate: it matches every other head()-window
    # metric in this module, and it leaves the positional iloc indices the EWMA depends on
    # exactly where they were, so a panel shorter than the window is untouched.
    _win = rp.scale_window(nq, rpy)
    if _win and len(eps) > _win:
        eps = eps.head(_win)
    epsmean = eps.mean()
    a = 0.4
    tw = a * (1 + (1 - a) + (1 - a) ** 2 + (1 - a) ** 3)
    # POSITIVITY GATE over the most recent YEAR: `rpy` rows, not a hard-coded 4.  On a
    # semi-annual filer 4 rows is TWO years, so the gate demanded two straight profitable
    # years where a quarterly peer only had to show one.
    # The length guard is a no-op for the live top-100 pool (always >= 4 rows) but is
    # load-bearing for the offline survivorship-clean reproduction, whose dead-merged
    # universe includes short-history names -- without it the indexed access below raises.
    _ny = int(rpy)
    if len(eps) >= max(4, _ny) and all(eps.iloc[0:_ny] > 0):
        den = abs(epsmean)
        scale = eps.abs().mean()
        if (not np.isfinite(den) or not np.isfinite(scale)
                or den <= EPS_MEAN_FLOOR_FRAC * scale):
            return np.nan
        # The exponential weighting spans the most recent YEAR: `rpy` terms with
        # geometric weights (1-a)^k, renormalised by their own sum so the weights still
        # total 1 whatever rpy is.  With rpy=4 `tw` and the four terms are unchanged, so a
        # quarterly name is bit-identical.
        _w = [(1 - a) ** k for k in range(_ny)]
        _tw = a * sum(_w)
        raw = epsmean - (a / _tw) * sum(eps.iloc[k] * _w[k] for k in range(_ny))
        return raw / den
    return np.nan


def price_growth(tempcdx, nq, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """Per-period price appreciation, head(nq) mean
    (call site: postBoRank._compute_ticker_metrics).

    cdx is NEWEST-first, so pct_change(-1) = (newer - older)/older is POSITIVE
    when the price rose.  NO negation (the leading '-' was a sign bug removed in
    lockstep here and offline).  NaN when the price column is missing/empty.
    """
    if "price" in tempcdx.columns and not tempcdx["price"].empty:
        # pct_change(-1) is ONE REPORTING PERIOD, which is already frequency-relative --
        # it needs no rescaling. Only the averaging window does, so the mean covers the
        # same calendar span for both frequencies. (A semi-annual name's per-period price
        # change is a 6-month move vs a quarterly name's 3-month one -- a LEVEL
        # difference the window cannot fix; priceGrowth is w=0.000 so nothing rests on
        # it, but it is noted with freeCashFlowYield in the report.)
        return _reduce(tempcdx["price"].pct_change(-1, fill_method=None),
                       'priceGrowth', rp.scale_window(nq, rpy), rpy,
                       tempcdx=tempcdx, scoring_nq=nq)
    return np.nan


def altman_z(tempcdx, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """Altman-Z from fundamentals, most-recent row
    (call site: postBoRank._compute_ticker_metrics).

    Z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5.  NaN when unusable.

    x3 (EBIT/TA) and x5 (Sales/TA) are FULL-YEAR flows in the published model, and are
    now computed as TRAILING FULL-YEAR SUMS over `rpy` rows (valuation-specialist
    annualization ruling, 2026-07-25).  They used to take a SINGLE period's
    operatingIncome and revenue against total assets, which (a) put both terms at ~1/4
    of their intended magnitude -- the audit's 1/4-scale defect, which is why the mcap
    term x4 dominated Z -- and (b) exposed them to the quarter's seasonality.  Summing
    the trailing year fixes both at once, and does so for BOTH frequencies (4 quarters or
    2 halves).  x1 / x2 / x4 are STOCK/STOCK and are untouched.

    NOTE this is NOT a no-op for quarterly names: x3 and x5 grow ~4x, so Z rises (median
    4.369 -> 5.303 on the shipped top-100).  It does NOT make the published 1.8/3.0
    distress bands usable -- an earlier version of this comment claimed that, wrongly.
    Those bands are DELIBERATELY UNUSED here: the display is VERDICT_GRAY with no tick and
    the R3 Z-limb was removed, precisely because this quantity is not the published Z.  It
    is still ~one term: independently re-derived under the annualized code, 0.6*x4
    (marketCap/totalLiabilities) is 66.5% of mean |contribution| and correlates +0.997 with
    Z.  The value of the fix is that the flow terms are now sign-correct and
    seasonality-free, not that Z became interpretable.
    Altman's absolute coefficients are why a genuine
    full-year sum is required here rather than the per-quarter normalisation used for the
    z-scored metrics.
    """
    try:
        n = int(rpy)
        if len(tempcdx) >= 1:
            curr = tempcdx.iloc[0]
            ta = curr["totalAssets"]
            tl = curr["totalLiabilities"]
            if ta > 0 and tl > 0:
                x1 = (curr["totalCurrentAssets"] - curr["totalCurrentLiabilities"]) / ta
                x2 = curr["totalStockholdersEquity"] / ta
                # Trailing full-year flow sums (newest-first frame -> head(n)).  NaN when
                # the year is incomplete, so a short history yields NaN rather than a
                # silently part-year Z.
                if len(tempcdx) < n:
                    return np.nan
                _oi = pd.to_numeric(tempcdx["operatingIncome"], errors="coerce").head(n)
                _rev = pd.to_numeric(tempcdx["revenue"], errors="coerce").head(n)
                if _oi.isna().any() or _rev.isna().any():
                    return np.nan
                x3 = _oi.sum() / ta
                x4 = curr["marketCap"] / tl
                x5 = _rev.sum() / ta
                return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        return np.nan
    except Exception:
        return np.nan


#  Piotroski's prior-period row is ONE YEAR back, which is `rpy` rows: 4 for a quarterly
#  filer, 2 for a semi-annual one.  tempcdx is NEWEST-FIRST
#  (postBoRank._sort_cdx_newest_first), so row `rpy` is one year older than row 0.
_PIOTROSKI_YOY_LAG = 4          # quarterly default; see piotroski(rpy=...)


def piotroski(tempcdx, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """Piotroski F-score (9 binary criteria) from fundamentals, current vs the
    SAME QUARTER ONE YEAR EARLIER (call site:
    postBoRank._compute_ticker_metrics).  NaN when unusable.

    YEAR-OVER-YEAR, not quarter-over-quarter (audit C2 fix, 2026-07-19).  Six of
    the nine criteria (p3 dROA, p5 dLeverage, p6 dLiquidity, p7 dilution, p8
    dMargin, p9 dTurnover) are Piotroski DELTA tests defined against the prior
    YEAR.  They used to read tempcdx.iloc[1] -- the previous QUARTER -- which
    makes every delta a seasonal comparison (a retailer's Q4-vs-Q3 margin move is
    seasonality, not fundamental improvement).  Measured on the 2026-07-17
    shipped top-100: corr(quarterly version, yearly version) = 0.545 and 33% of
    the pool crosses the >=7 / <7 "strong F-score" line, so this is a material
    re-ordering, not a rounding difference.

    SEMI-ANNUAL REPORTERS -- RESOLVED (2026-07-25).  FMP labels H1/H2 as Q2/Q4, so a
    semi-annual filer has 2 rows per year and a fixed lag of 4 compared against TWO
    years ago.  The lag is now `rpy` (reporting_period), so a semi-annual name compares
    H1 against the PRIOR YEAR'S H1 and a quarterly name is unchanged at 4.

    A NaN COMPONENT NOW MAKES THE COMPOSITE NaN (nan-policy.md D-9 / section 4a, 2026-08-05).
    IT DID NOT BEFORE, AND THAT WAS THE DEFECT: `NaN > x` is False, so a missing input scored
    the point 0 -- INDISTINGUISHABLE FROM FAILING THE TEST.  A company was therefore marked
    down for a gap in the provider's data, which is the one thing this project's standing
    premise forbids ("missing data must never reward by default" cuts both ways: it must not
    silently PUNISH either, because a punishment for absence is not a measurement of the
    company).
    WHY NaN AND NOT "the fraction of the computable tests passed": that form would REWARD a
    company for having fewer tests apply to it.  Making the COMPOSITE NaN reaches the honest
    place instead -- the column becomes unavailable, `normalizeAndDropNA` imputes it at the
    column MEDIAN, and nothing is either credited or docked.  The precedent is already in this
    module and was built this way on purpose: `share_count_change` and `long_term_debt_change`.
    MEASURED [panel]: 117 sources (1.51%) [universe], 33 of the 4,287 general-carved names
    (0.77%), **0 of the 100 deployed pool names** -- and every one of them arrives via
    `netCashProvidedByOperatingActivities` (p2 and p4).
    p5's STRUCTURAL ZERO IS NOW DISCRIMINATED (register B-8 extended to Piotroski, CEO ruling
    2026-08-06).  The note below used to say p5 was a SEPARATE ruling because Piotroski is a
    metric this project only MIMICS (D-9) and changing it would be a redesign.  THE CEO RULED
    THE OTHER WAY: this is CONFORMANCE, NOT REDESIGN.  p5 asks "did long-term debt fall".  A
    row whose `longTermDebt == 0` while a sibling debt field is strictly POSITIVE is levered
    somewhere, so its zero is NON-DISCLOSURE OR MISALLOCATION -- and evaluating `0 < 0` on it
    does not answer Piotroski's question, it fails it by default.  Applying
    `_long_term_debt_undisclosed` (the SAME discriminator built for
    `long_term_debt_change`) does not change WHAT p5 asks; it stops the metric answering a
    question the data cannot support.  Absence is then NaN, which under the composite rule
    above makes `Piotroski` NaN -- unavailable, imputed at the column median, neither credited
    nor docked.  THAT is the honest place; a forced FAIL is a punishment for a provider gap.

    SAME CAVEAT AS B-8, VERBATIM, BECAUSE IT IS THE SAME FIELDS.  `totalDebt` and
    `shortTermDebt` were captured on 2026-08-05 and A SAVED PICKLE CAN NEVER GAIN A COLUMN, so
    they are ABSENT FROM EVERY EXISTING PANEL.  When neither sibling is present the
    discriminator CANNOT FIRE and this function behaves EXACTLY as it did before -- so the
    change is UNEXERCISABLE on saved data and only becomes live after the next full fetch.  A
    PRESENT-BUT-ZERO sibling is evidence FOR debt-freedom and correctly does not fire.  How
    many of the 476 sources it will actually reach is therefore UNMEASURED and UNMEASURABLE
    here: 476 is the count of sources whose p5 reads a zero, NOT the count whose zero is
    contradicted -- the genuinely unlevered ones are a real observation and must keep failing
    p5, because their leverage genuinely did not fall.

    AND DO NOT RECORD D-9 AS CLOSED BY THIS.  Two of the three structural zeros are STILL
    OPEN and no NaN rule reaches them: `revenue == 0` fails p9 on 380 sources and
    `grossProfitMargin == 0` fails p8 on 380.  Neither has a sibling field that contradicts it
    the way `totalDebt` contradicts `longTermDebt`, so the B-8 pattern does not transfer --
    they need a provider-level presence flag at ingest, not a metric change (nan-policy.md
    section 5).
    """
    try:
        lag = int(rpy)
        if len(tempcdx) >= lag + 1:
            curr = tempcdx.iloc[0]     # Most recent quarter
            prev = tempcdx.iloc[lag]   # Same quarter one year earlier (4 rows older)
            #  Every input the nine components read, from BOTH rows.  `_finite` returns None
            #  for absent / non-numeric / non-finite -- the same helper the two extracted
            #  components use, so "what counts as present" is stated once for all three.
            need_curr = ('totalAssets', 'netIncome',
                         'netCashProvidedByOperatingActivities', 'longTermDebt',
                         'currentRatio', 'weightedAverageShsOut', 'grossProfitMargin',
                         'revenue')
            need_prev = ('totalAssets', 'netIncome', 'longTermDebt', 'currentRatio',
                         'weightedAverageShsOut', 'grossProfitMargin', 'revenue')
            c = {k: _finite(curr[k]) for k in need_curr}
            p = {k: _finite(prev[k]) for k in need_prev}
            if any(v is None for v in c.values()) or any(v is None for v in p.values()):
                return np.nan
            #  B-8 EXTENDED TO p5 (CEO 2026-08-06): a `longTermDebt == 0` that a sibling
            #  debt field CONTRADICTS is non-disclosure, so p5 is UNANSWERABLE and the
            #  composite is NaN -- not a forced FAIL. Absent siblings (every saved pickle)
            #  -> no verdict -> byte-identical behaviour. Placed with the other
            #  all-inputs-present checks and BEFORE any component is scored, so a single
            #  return covers it exactly as the NaN-input check above does.
            if _long_term_debt_undisclosed(curr) or _long_term_debt_undisclosed(prev):
                return np.nan
            ta_curr = c["totalAssets"]
            ta_prev = p["totalAssets"]
            if ta_curr > 0 and ta_prev > 0:
                p1 = 1 if c["netIncome"] / ta_curr > 0 else 0
                p2 = 1 if c["netCashProvidedByOperatingActivities"] > 0 else 0
                roa_curr = c["netIncome"] / ta_curr
                roa_prev = p["netIncome"] / ta_prev
                p3 = 1 if roa_curr > roa_prev else 0
                p4 = 1 if c["netCashProvidedByOperatingActivities"] > c["netIncome"] else 0
                ltd_ratio_curr = c["longTermDebt"] / ta_curr
                ltd_ratio_prev = p["longTermDebt"] / ta_prev
                p5 = 1 if ltd_ratio_curr < ltd_ratio_prev else 0
                p6 = 1 if c["currentRatio"] > p["currentRatio"] else 0
                p7 = 1 if c["weightedAverageShsOut"] <= p["weightedAverageShsOut"] else 0
                p8 = 1 if c["grossProfitMargin"] > p["grossProfitMargin"] else 0
                at_curr = c["revenue"] / ta_curr
                at_prev = p["revenue"] / ta_prev
                p9 = 1 if at_curr > at_prev else 0
                return p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        return np.nan
    except Exception:
        return np.nan


# =========================================================================== #
#  TWO PIOTROSKI COMPONENTS, EXTRACTED AS STANDALONE METRICS (E-2, 2026-08-04)  #
# =========================================================================== #
#  WHY THEY EXIST.  For a FIN-1 investment vehicle (closed-end fund / BDC) SEVEN of
#  Piotroski's nine components are undefined or degenerate -- and, as `piotroski` above
#  shows line by line, an undefined component does NOT propagate NaN: a NaN input makes
#  every `>` / `<` / `<=` comparison False, so the point scores 0, which is
#  indistinguishable from FAILING the test.  The composite is therefore not merely
#  uninformative in that cohort, it is SYSTEMATICALLY PUNITIVE against every member of it,
#  which is why `Piotroski` carries no weight there at all (scoringWeights, `C` block OOD
#  in FIN-1).
#
#  But two of the nine ARE meaningful for a vehicle, and they are the closest thing the
#  pipeline has to the NAV-quality instrument that cohort otherwise entirely lacks:
#  SHARE ISSUANCE (p7) and CHANGE IN LONG-TERM DEBT (p5).  Issuing shares below net asset
#  value is the canonical BDC red flag -- total NAV rises while NAV PER SHARE falls, so a
#  vehicle can look cheap on book-to-price while bleeding per-share value.
#
#  THE TWO DESIGN DECISIONS, both stated because either could reasonably have gone the
#  other way:
#
#  1. CONTINUOUS, NOT BINARY.  Piotroski's own points are 0/1; these return the underlying
#     SIGNED QUANTITY whose sign is that point.  Three reasons.  (a) A 1% issuance and a
#     40% issuance are not the same red flag, and the whole purpose here is a magnitude the
#     cross-sectional ruler can rank.  (b) A binary column on a small cohort is almost all
#     ties, and the pool is ~25 names.  (c) It is monotone-consistent with the point it came
#     from: sign(shareCountChange) <= 0 IS p7, sign(longTermDebtChange) < 0 IS p5, so
#     nothing about the extraction reverses Piotroski's judgement.
#
#  2. AN UNDEFINED INPUT YIELDS NaN -- NEVER A PASS AND NEVER A FAIL.  This is the standing
#     project premise ("missing data must never reward by default") and it is the specific
#     defect the parent composite has.  These functions are deliberately NOT a "fraction of
#     the computable tests passed": such a form would REWARD a company for having fewer
#     tests apply to it.  NaN goes to `normalizeAndDropNA`, which median-centres the column
#     and imputes NaN to exactly the column MEDIAN -- genuinely neutral, and neutral on the
#     RAW scale too because the weight is negative (a NaN is not quietly credited with
#     "did not dilute").
#
#  THE LIMIT, RECORDED: both are CHANGE measures, so an unlevered vehicle that stays
#  unlevered reads 0.0 while a levered one that deleverages reads better than it.  That is
#  the extracted component's own semantics; a LEVEL term would be a new metric, not an
#  extraction, and is not in scope here.


def _yoy_rows(tempcdx, rpy):
    """(row 0, row rpy) -- this period and the SAME period one year earlier -- or None.

    THE SHARED SEAM for the two extracted components, and it is deliberately the same
    indexing `piotroski` does: newest-first frame, so row `rpy` is exactly one year older
    than row 0 (4 rows quarterly, 2 semi-annual).  Sharing it is what makes "extracted from
    Piotroski" true of the ROWS as well as of the formula -- a second hand-written lag here
    is how the two would drift apart.
    """
    lag = int(rpy)
    if len(tempcdx) < lag + 1:
        return None
    return tempcdx.iloc[0], tempcdx.iloc[lag]


def _finite(value):
    """`value` as a float, or None when it is absent / non-numeric / non-finite.

    An ABSENT input must reach the caller as None so the metric can return NaN.  The one
    thing it must never do is coerce to 0.0, which is how the parent composite turns
    "unknown" into "failed".
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def share_count_change(tempcdx, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """Fractional YoY change in the share count -- Piotroski p7 made continuous
    (call site: postBoRank._compute_ticker_metrics).

        (shsOut_now - shsOut_one_year_ago) / shsOut_one_year_ago

    POSITIVE = DILUTION (shares issued), negative = buyback, 0.0 = unchanged.  The metric
    is deliberately named for the QUANTITY rather than for the judgement: the weight carries
    the sign (negative in the FIN-1 vector), exactly as `CycleHeat` does, so the published
    column always holds the change as measured.

    Dimensionless (a count over a count), so no frequency factor applies -- only the lag,
    which `rpy` supplies.

    NaN when: fewer than rpy+1 rows; either share count absent / non-finite; or the prior
    share count is <= 0 (the denominator would be meaningless, not merely large).  NaN is
    the honest answer for "not computable" and normalizeAndDropNA imputes it to the column
    median; returning 0.0 there would assert "did not dilute", which is a PASS awarded for
    missing data.
    """
    try:
        rows = _yoy_rows(tempcdx, rpy)
        if rows is None:
            return np.nan
        curr, prev = rows
        now = _finite(curr["weightedAverageShsOut"])
        then = _finite(prev["weightedAverageShsOut"])
        if now is None or then is None or then <= 0:
            return np.nan
        return (now - then) / then
    except Exception:
        return np.nan


#  The B-8 sibling fields, named once.  `totalDebt` and `shortTermDebt` were captured on
#  2026-08-05 and are ABSENT FROM EVERY EXISTING PICKLE, so every reader of this must treat
#  "column not on the frame" as NO VERDICT rather than as evidence either way.
_B8_DEBT_SIBLINGS = ('totalDebt', 'shortTermDebt')


def _long_term_debt_undisclosed(row):
    """Is this row's `longTermDebt == 0` CONTRADICTED by a sibling debt field?

    True  -- longTermDebt is exactly 0 while some sibling is strictly positive: the entity
             is levered somewhere, so the zero is non-disclosure/misallocation, not
             debt-freedom.
    False -- everything else, INCLUDING the case where no sibling is present at all.  That
             degradation is the point: on a panel without the siblings this function can
             never fire and callers behave exactly as they did before B-8.
    """
    ltd = _finite(row.get("longTermDebt") if hasattr(row, "get") else row["longTermDebt"])
    if ltd is None or ltd != 0:
        return False
    for col in _B8_DEBT_SIBLINGS:
        try:
            sib = _finite(row[col])
        except (KeyError, IndexError):
            continue          # column absent from this panel -> no verdict from it
        if sib is not None and sib > 0:
            return True
    return False


def long_term_debt_change(tempcdx, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """YoY change in the long-term-debt-to-assets RATIO -- Piotroski p5 made continuous
    (call site: postBoRank._compute_ticker_metrics).

        longTermDebt_now / totalAssets_now  -  longTermDebt_then / totalAssets_then

    POSITIVE = leverage ADDED, negative = deleveraging, 0.0 = unchanged.  As with
    `share_count_change` the sign lives in the weight, not in the name.

    THE RATIO, NOT THE RAW DELTA, and that is the faithful extraction: `piotroski`'s p5
    compares `longTermDebt / totalAssets` across the two periods, so differencing that ratio
    is the quantity whose sign p5 tests.  It also makes the metric scale-free and
    currency-invariant on both sides, which a raw currency delta would not be -- and this
    pool mixes GBp / SEK / USD / CAD reporters.

    NaN when: fewer than rpy+1 rows; either totalAssets absent or <= 0; or either
    longTermDebt absent / non-finite.  A REPORTED ZERO long-term debt is NOT absent -- an
    unlevered vehicle genuinely has a leverage ratio of 0 and differencing two zeros
    genuinely gives 0.0, so that case returns 0.0 and is a real observation.  Only a MISSING
    field yields NaN.

    MEASURED RESIDUAL ON THAT CONTRACT, recorded because it is the one way it can bite
    (2026-08-04, local 9,012-name panel): `longTermDebt` is **0.00% NaN and 25.33% EXACTLY
    ZERO**.  So the NaN branch above is effectively UNREACHABLE in practice -- FMP reports 0
    rather than omitting the field.  The contract is still the right one and there is no
    upstream zero-fill to undo, but the consequence must be stated plainly rather than left
    implied: **if FMP's 0 ever conflates "has no long-term debt" with "did not disclose it",
    this metric reads the second as the first, and the -0.15 weight then awards a PASS for
    missing data -- in the one cohort where leverage IS the solvency signal.**

    THE NON-DISCLOSURE DISCRIMINATOR (register B-8, CEO ruling 2026-08-05).  The note above
    said the two cases are "the same byte on the wire".  They are not, once the SIBLING debt
    fields are on the frame: a row with `longTermDebt == 0` while `totalDebt > 0` (or
    `shortTermDebt > 0`) is levered somewhere, so its zero is NON-DISCLOSURE OR
    MISALLOCATION, not a debt-free balance sheet -- whereas all of them at zero is genuinely
    unlevered.  RULING: the non-disclosure case is **NaN**, i.e. the metric is UNAVAILABLE
    and takes the existing secondary (column-median) NaN treatment.  ABSENCE IS NOT A PASS;
    it is the -0.15 weight that makes the distinction directional, so conflating them hands
    a free pass to non-disclosers.

    DEGRADATION IS THE NORMAL CASE TODAY, and it is deliberate.  `totalDebt` /
    `shortTermDebt` were captured on 2026-08-05 and **A SAVED PICKLE CAN NEVER GAIN A
    COLUMN**, so they are ABSENT FROM EVERY EXISTING PANEL.  When neither sibling is present
    the discriminator cannot fire and this function behaves EXACTLY AS IT DID BEFORE -- a
    reported zero stays a real observation.  So this change is unexercisable on saved data
    and only becomes live after the next full fetch; the test pins both directions.
    A PRESENT-BUT-ZERO sibling is evidence FOR debt-freedom and correctly does not fire.

    STILL OPEN, and NOT closed by this: `piotroski`'s p5 reads the same conflated zero and
    fails `0 < 0` forever on 476 sources (6.17%).  The identical discriminator would fix it,
    but p5 is a component of a metric this project is deliberately only MIMICKING (D-9), so
    it is a separate ruling, not a silent ride-along on this one.
    """
    try:
        rows = _yoy_rows(tempcdx, rpy)
        if rows is None:
            return np.nan
        curr, prev = rows
        ta_now = _finite(curr["totalAssets"])
        ta_then = _finite(prev["totalAssets"])
        if not ta_now or not ta_then or ta_now <= 0 or ta_then <= 0:
            return np.nan
        ltd_now = _finite(curr["longTermDebt"])
        ltd_then = _finite(prev["longTermDebt"])
        if ltd_now is None or ltd_then is None:
            return np.nan
        # B-8: a zero that the sibling debt fields CONTRADICT is non-disclosure -> NaN.
        # Absent siblings (every saved pickle) -> no verdict -> unchanged behaviour.
        if _long_term_debt_undisclosed(curr) or _long_term_debt_undisclosed(prev):
            return np.nan
        return ltd_now / ta_now - ltd_then / ta_then
    except Exception:
        return np.nan


def cycleheat_zscore(eps_clean, eps_current):
    """CycleHeat's core: self-normalised z-score of ``eps_current`` vs the
    stock's own EPS history ``eps_clean`` (callers: cycleheat below, and
    baseline_tools.stage2_pit).

    This is the drift-prone FORMULA -- shared so the live scorer and the offline
    reproduction can never disagree on it.  It deliberately does NOT decide the
    ORDER of ``eps_clean`` or which row is "current": each caller prepares its
    own EPS series (see cycleheat() below vs the offline loop) and passes
    ``eps_current`` explicitly.  Keeping series-preparation caller-side preserves
    each path's exact floating-point reduction order bit-for-bit while still
    unifying the formula.

    Positive = earnings well above the stock's own mean (hot / late-cycle);
    negative = below (cold / potential recovery).  No market-beta multiplier
    (removed as an axis error).  Capped to [-3, 3].  NaN when < 2 observations.
    """
    if len(eps_clean) >= 2:
        eps_mean = eps_clean.mean()
        eps_std = eps_clean.std()
        if eps_std > 0 and not np.isnan(eps_std):
            eps_zscore = (eps_current - eps_mean) / eps_std
        elif eps_mean != 0:
            eps_zscore = (eps_current - eps_mean) / abs(eps_mean)
        else:
            eps_zscore = 0.0
        return max(-3.0, min(eps_zscore, 3.0))
    return np.nan


def prepare_eps_series(tempcdx):
    """Canonical per-ticker EPS history for CycleHeat, shared by the live scorer
    (cycleheat) and the offline reproduction (stage2_pit) so they agree BY
    CONSTRUCTION on duplicate-dated (restated) quarters.

    THE PROBLEM this solves: FMP sometimes carries >1 record for a single period
    `date` (a restatement, or a fiscal/calendar-boundary collision).  cdx_df has
    NO filing-date / acceptedDate / period column to disambiguate, and the live
    scorer (date-ascending, iloc[-1]) and the offline path (newest-first,
    iloc[0]) used to pick DIFFERENT tied rows for the "current" quarter -- e.g.
    ILMN as-of 2023-01-27 has two 2022-10-01 rows (netIncome -139M vs -3.816B),
    and the two paths disagreed on which is "now".

    CANONICAL RULE (deterministic; documented assumption): collapse each date to
    ONE row, keeping the LAST-INGESTED record.  cdx_df stores fundamentals in
    ingestion order oldest-first and both callers derive tempcdx from it via
    stable sorts, so same-date ties preserve ingestion order; keep-last selects
    the most-recently-obtained (restated / current) figure.  Verified on the ILMN
    2022-Q3 GRAIL-impairment case: keep-last yields the GAAP-reported -3.816B,
    not the -139M pre-restatement line.  (No filing-date exists to do better;
    the value shows no consistent magnitude pattern across names, so ingestion
    recency is the principled, deterministic signal.)

    Returns the cleaned EPS Series in DATE-ASCENDING order (most-recent quarter =
    .iloc[-1]); rows with an unparseable date and inf/NaN EPS are dropped.
    """
    f = tempcdx[["date", "netIncome", "weightedAverageShsOut"]].copy()
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    f = f.dropna(subset=["date"])
    f = f.sort_values("date", kind="stable")            # ascending; ties keep ingestion order
    f = f.drop_duplicates(subset="date", keep="last")   # canonical = last-ingested per date
    eps = f["netIncome"] / f["weightedAverageShsOut"]
    return eps.replace([np.inf, -np.inf], np.nan).dropna()


#  CycleHeat's baseline window, in QUARTERS, scaled to the filer's own frequency by
#  rp.scale_window (28 quarterly rows / 14 semi-annual rows = ~7 years either way).
#
#  WHY A CAP EXISTS AT ALL (fix, 2026-07-30).  CycleHeat is a SELF-REFERENCE z-score: how far
#  the latest EPS sits above the stock's OWN history.  It therefore depends entirely on how
#  much history it is handed -- and it was handed "whatever rows exist", the only metric in the
#  Stage-2 block taking neither `nq` nor `rpy`.  Two measured consequences on the 07-17 panel:
#    * A SEMI-ANNUAL filer's window already spanned 11.04 years (median 11.50, max 25.25)
#      against a quarterly peer's 5.53 (median 5.75) -- ~2x the calendar time off the same
#      ~22.5 rows, because rows-per-year differs.  ~14% of the universe, and 31 of 57 rendered
#      deck pages, were being measured against a baseline twice as long as their peers'.
#    * A LONGER window RAISES h for an improving company, because a longer history means a
#      lower baseline to exceed.  Measured on the 1,065 deep-history names (span > 8y):
#      mean h on the full window vs the most recent half was +0.981 vs +0.735 for rising-EPS
#      names (+0.246 sigma) and -0.797 vs -0.642 for falling ones.  w(CycleHeat) = -0.080, so
#      that is a straight PENALTY on exactly the population the CEO singled out as needing
#      protection: "a real value company that has recently become one".
#  At `-nrperiods 80` those spans would become ~20 and ~40 years, so the deep-history fetch --
#  undertaken to make cyclicality DETECTABLE -- would have intensified the penalty on improving
#  companies instead.  The cap makes h a fixed-calendar-time measure regardless of fetch depth.
#
#  WHY 28 QUARTERS.  Two constraints pick it: it must be long enough to contain a full business
#  cycle (the quantity the metric is named for), and it must be >= the ~24 rows a quarterly
#  filer carries on the CURRENT panel so it CANNOT BIND there -- making this change bit-identical
#  for every quarterly name today, which is the regression that matters.  7 years satisfies both.
#
#  DECLARED IN THE REGISTRY as STAGE2_METRIC_SPEC['CycleHeat'] -> WINDOW_CYCLEHEAT_BASE, and
#  `window_quarters('CycleHeat', nq)` returns THIS constant.  The registry decides WHICH window
#  basis the metric uses; this constant is the VALUE of that basis and keeps the reasoning
#  above.  test_stage2_registry pins the two together so the default below cannot drift from
#  the declaration.
CYCLEHEAT_BASE_NQ = 28


def cycleheat(tempcdx, nq=CYCLEHEAT_BASE_NQ, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """CycleHeat for BOTH the live scorer and the offline reproduction
    (call site: postBoRank._compute_ticker_metrics).

    Uses the shared canonical EPS history (prepare_eps_series) -- one row per
    date, restatement tie broken to the last-ingested figure -- TRUNCATED to the most recent
    `scale_window(nq, rpy)` observations, then delegates the z-score to cycleheat_zscore.
    "current" = the most-recent quarter (.iloc[-1] of the date-ascending series).  NaN on
    failure.

    `rpy` is THIS filer's rows-per-year, so the window spans the same CALENDAR time for a
    semi-annual filer as for a quarterly one -- the same treatment every neighbouring metric in
    the Stage-2 block already had.  Defaults keep old behaviour only in the degenerate sense
    that a caller omitting `rpy` gets the quarterly window; callers MUST pass it (all three do).
    """
    try:
        w = rp.scale_window(nq, rpy)
        #  POLICY GATE (2026-08-05).  CycleHeat is Type U, so coverage and interior gappiness
        #  apply to it like every other windowed metric -- but they must be measured on the
        #  ROW series, before `prepare_eps_series` drops the non-computable rows, because a
        #  dropna'd series has no gaps left to find.  Only the NaN-ness of `_gate` is used; its
        #  value (the mean EPS) is not the metric.  MEASURED: coverage < 0.50 on 0 sources,
        #  >= 2 interior gaps on 15 sources (0.19%) [universe] / 0 [pool].
        #  NOTE the two windows are not the same object and that is pre-existing: this gate is
        #  row-based (head(w) of the panel), while prepare_eps_series takes the most recent w
        #  OBSERVATIONS after its own dropna/dedup.  The gate is deliberately the stricter,
        #  row-based reading -- it is the one that can see a gap.
        _gate = _reduce((pd.to_numeric(tempcdx["netIncome"], errors="coerce")
                         / pd.to_numeric(tempcdx["weightedAverageShsOut"], errors="coerce")),
                        'CycleHeat', w, rpy, tempcdx=tempcdx)
        if pd.isna(_gate):
            return np.nan
        eps_clean = prepare_eps_series(tempcdx)
        if w and len(eps_clean) > w:
            eps_clean = eps_clean.tail(w)          # most-recent w observations
        if len(eps_clean) >= 2:
            return cycleheat_zscore(eps_clean, eps_clean.iloc[-1])   # iloc[-1] = MOST RECENT
        return np.nan
    except Exception:
        return np.nan


def dcf_to_price(dcf, nq):
    """DCF fair value / price from a per-ticker DCF frame
    (call site: postBoRank._compute_ticker_metrics).

    PRODUCTION-ONLY: the offline PIT reproduction has no point-in-time DCF, so it
    drops this metric (DcfToPrice is weight 0 in the live vector).  Kept here for
    readability of the live scorer.  Handles the JSON-API ('Stock Price') and CSV
    bulk ('StockPrice'/'stock_price') column variants.  NaN when unusable.
    """
    if dcf is None or dcf.empty:
        return np.nan
    price_col = None
    if "Stock Price" in dcf.columns:
        price_col = "Stock Price"
    elif "StockPrice" in dcf.columns:
        price_col = "StockPrice"
    elif "stock_price" in dcf.columns:
        price_col = "stock_price"
    if "dcf" in dcf.columns and price_col:
        temp = dcf["dcf"].head(nq).mean()
        temp2 = dcf[price_col].iloc[0] if len(dcf) > 0 else None
        if temp2 is not None and temp2 != 0:
            return temp / temp2
        return np.nan
    return np.nan
