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
"""

import numpy as np
import pandas as pd

import reporting_period as rp

# Stage-2 metrics whose numerator is a per-period FLOW over a STOCK, so a semi-annual
# filer reads ~2x a quarterly peer purely from its reporting convention (valuation-
# specialist annualization ruling, 2026-07-25).  Every one of these is z-scored
# cross-sectionally in normalizeAndDropNA, which is invariant to multiplying ALL rows by
# one constant -- so only the SEMI-ANNUAL/QUARTERLY ratio matters and the correction is
# expressed on a common PER-QUARTER basis (x1 quarterly = exact no-op, x0.5 semi-annual).
# NOT corrected, per the same ruling: flow/flow ratios (grossProfitMargin,
# netProfitMargin, incomeQuality) and stock/stock ratios (bVpRatio, tbVpRatio,
# currentRatio) -- the scale cancels; grahamNumberToPrice (already corrected at ingest via
# EPS_ttm, and it carries sqrt(2) not 2x, so re-correcting would overshoot); revenueGrowth
# and freeCashFlowPerShareGrowth (handled by the YoY window); CycleHeat (self-normalised);
# priceGrowth (a period SPAN, w=0.000).
STAGE2_FLOW_OVER_STOCK = ('RoA', 'earnYield', 'returnOnEquity',
                          'returnOnCapitalEmployed')

# EVERY window below is written for QUARTERLY rows and scaled by `rpy` (rows per year:
# 4 quarterly, 2 semi-annual -- reporting_period).  rpy defaults to 4 everywhere, so a
# caller that passes nothing is BIT-IDENTICAL to the pre-2026-07-25 behaviour; only a
# source classified semi-annual takes a different window.  The scaling is always
# CALENDAR-equivalent: a 4-row YoY becomes 2 rows, a head(16) window becomes head(8).


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
    sign flipped so SMALLER caps score HIGHER (postBoRank.py mcapQuants).

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
def postbm_metric(key, met, tempcdx, nq, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """One postBmRankingDict metric for a single ticker (postBoRank.py:205-219).

    grahamNumberToPrice / bVpRatio / revenueGrowth are special-cased; every
    other key is the head(nq) mean of its ``eqMet`` column.

    `nq` is scaled to `rpy` so the averaging window spans the same CALENDAR time for a
    semi-annual filer, and revenueGrowth's YoY shift is `rpy` rows (4 quarters OR 2
    halves) instead of a hard-coded 4 -- on a semi-annual name a 4-row shift measured
    TWO-year growth and called it annual.
    """
    w = rp.scale_window(nq, rpy)
    # Flow/stock keys are put on a common per-quarter basis; everything else is x1.0.
    ff = rp.per_quarter_factor(rpy) if key in STAGE2_FLOW_OVER_STOCK else 1.0
    if key == "grahamNumberToPrice":
        return (tempcdx["grahamNumber"] / tempcdx["price"]).head(w).mean()
    elif key == "bVpRatio":
        return (1 / tempcdx[met]).head(w).mean()
    elif key == "revenueGrowth":
        return tempcdx[met].pct_change(-int(rpy), fill_method=None).head(w).mean()
    else:
        return tempcdx[met].head(w).mean() * ff


# --------------------------------------------------------------------------- #
#  postNewRankingDict metrics                                                 #
# --------------------------------------------------------------------------- #
def free_cash_flow_yield(tempfcf, tempmcap, nq, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """FCF / marketCap, head(nq) mean (postBoRank.py:234).

    FLOW/STOCK: a semi-annual row's FCF is a 6-month flow over a point-in-time market
    cap, so the per-row yield reads ~2x a quarterly peer's.  The value is z-scored
    downstream, so the correction only needs to fix the SEMI-ANNUAL/QUARTERLY ratio and is
    applied on a common per-quarter basis (x1 quarterly = no-op).  Ruling applied
    2026-07-25; the previous note here said this was deferred."""
    return ((tempfcf / tempmcap).head(rp.scale_window(nq, rpy)).mean()
            * rp.per_quarter_factor(rpy))


def free_cash_flow_per_share_growth(tempfcf, tempshares, nq,
                                    rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """YoY growth of FCF-per-share over `rpy` rows, head(nq) mean (postBoRank.py:240-242).

    `rpy` rows back is one YEAR for either frequency; the hard-coded 4 was two years for
    a semi-annual filer."""
    fcfps = tempfcf / tempshares
    return (fcfps.pct_change(-int(rpy), fill_method=None)
            .head(rp.scale_window(nq, rpy)).mean())


def tbv_p_ratio(tempcdx, nq, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """Tangible book value per share / price, head(nq) mean (postBoRank.py:304-306).
    Both inputs are point-in-time STOCKS, so only the averaging window scales."""
    return (tempcdx["tangibleBookValuePerShare"] / tempcdx["price"]
            ).head(rp.scale_window(nq, rpy)).mean()


# Minimum |mean EPS| for the EPStoEPSmean denominator, as a FRACTION of the mean
# ABSOLUTE EPS over the same window.  Below this the mean is a near-cancellation of
# a swinging EPS series (mean ~ 0 with large |EPS|), so the ratio explodes on
# arithmetic rather than on economics -> NaN (= neutral) instead of a huge number.
EPS_MEAN_FLOOR_FRAC = 0.01


def eps_to_eps_mean(tempcdx, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """Exponentially-weighted recent EPS vs its full-window mean, expressed as a
    FRACTION OF THAT MEAN (postBoRank.py:248-256).

      (epsmean - ewma_recent_eps) / |epsmean|

    Positive = the last four quarters sit BELOW the stock's own EPS history (the
    mean-reversion side the +0.056 weight is betting on); negative = above.

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
    """
    eps = tempcdx["netIncome"] / tempcdx["weightedAverageShsOut"]
    eps = eps.replace([np.inf, -np.inf], np.nan)
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
    """Per-period price appreciation, head(nq) mean (postBoRank.py:411-428).

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
        return (tempcdx["price"].pct_change(-1, fill_method=None)
                .head(rp.scale_window(nq, rpy)).mean())
    return np.nan


def altman_z(tempcdx, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """Altman-Z from fundamentals, most-recent row (postBoRank.py:310-349).

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
    SAME QUARTER ONE YEAR EARLIER (postBoRank.py:353-402).  NaN when unusable.

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
    """
    try:
        lag = int(rpy)
        if len(tempcdx) >= lag + 1:
            curr = tempcdx.iloc[0]     # Most recent quarter
            prev = tempcdx.iloc[lag]   # Same quarter one year earlier (4 rows older)
            ta_curr = curr["totalAssets"]
            ta_prev = prev["totalAssets"]
            if ta_curr > 0 and ta_prev > 0:
                p1 = 1 if curr["netIncome"] / ta_curr > 0 else 0
                p2 = 1 if curr["netCashProvidedByOperatingActivities"] > 0 else 0
                roa_curr = curr["netIncome"] / ta_curr
                roa_prev = prev["netIncome"] / ta_prev
                p3 = 1 if roa_curr > roa_prev else 0
                p4 = 1 if curr["netCashProvidedByOperatingActivities"] > curr["netIncome"] else 0
                ltd_ratio_curr = curr["longTermDebt"] / ta_curr
                ltd_ratio_prev = prev["longTermDebt"] / ta_prev
                p5 = 1 if ltd_ratio_curr < ltd_ratio_prev else 0
                p6 = 1 if curr["currentRatio"] > prev["currentRatio"] else 0
                p7 = 1 if curr["weightedAverageShsOut"] <= prev["weightedAverageShsOut"] else 0
                p8 = 1 if curr["grossProfitMargin"] > prev["grossProfitMargin"] else 0
                at_curr = curr["revenue"] / ta_curr
                at_prev = prev["revenue"] / ta_prev
                p9 = 1 if at_curr > at_prev else 0
                return p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        return np.nan
    except Exception:
        return np.nan


def cycleheat_zscore(eps_clean, eps_current):
    """CycleHeat's core: self-normalised z-score of ``eps_current`` vs the
    stock's own EPS history ``eps_clean`` (postBoRank.py:456-483).

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


def cycleheat(tempcdx):
    """CycleHeat for BOTH the live scorer and the offline reproduction
    (postBoRank.py:433-488).

    Uses the shared canonical EPS history (prepare_eps_series) -- one row per
    date, restatement tie broken to the last-ingested figure -- then delegates
    the z-score to cycleheat_zscore.  "current" = the most-recent quarter
    (.iloc[-1] of the date-ascending series).  NaN on failure.
    """
    try:
        eps_clean = prepare_eps_series(tempcdx)
        if len(eps_clean) >= 2:
            return cycleheat_zscore(eps_clean, eps_clean.iloc[-1])   # iloc[-1] = MOST RECENT
        return np.nan
    except Exception:
        return np.nan


def dcf_to_price(dcf, nq):
    """DCF fair value / price from a per-ticker DCF frame (postBoRank.py:261-288).

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
