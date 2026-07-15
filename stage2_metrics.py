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


# --------------------------------------------------------------------------- #
#  Pool-level helper                                                          #
# --------------------------------------------------------------------------- #
def add_mcap_quants(cdxtop):
    """Pool-level marketCap quartile code, mapped to [-0.5 .. +0.5] with the
    sign flipped so SMALLER caps score HIGHER (postBoRank.py mcapQuants).

    Uses ``duplicates='drop'`` + a 0.0 fallback so a pool with coincident
    quartile edges degrades gracefully rather than raising.  On any pool with
    distinct edges (every successful scorer run) this equals the plain
    ``pd.qcut(marketCap, 4)`` the live scorer historically used.
    """
    try:
        return (-1) * ((pd.qcut(cdxtop["marketCap"], 4,
                                duplicates="drop").cat.codes / 3) - 0.5)
    except Exception:
        return pd.Series(0.0, index=cdxtop.index)


# --------------------------------------------------------------------------- #
#  postBmRankingDict metrics                                                  #
# --------------------------------------------------------------------------- #
def postbm_metric(key, met, tempcdx, nq):
    """One postBmRankingDict metric for a single ticker (postBoRank.py:205-219).

    grahamNumberToPrice / bVpRatio / revenueGrowth are special-cased; every
    other key is the head(nq) mean of its ``eqMet`` column.
    """
    if key == "grahamNumberToPrice":
        return (tempcdx["grahamNumber"] / tempcdx["price"]).head(nq).mean()
    elif key == "bVpRatio":
        return (1 / tempcdx[met]).head(nq).mean()
    elif key == "revenueGrowth":
        return tempcdx[met].pct_change(-4, fill_method=None).head(nq).mean()
    else:
        return tempcdx[met].head(nq).mean()


# --------------------------------------------------------------------------- #
#  postNewRankingDict metrics                                                 #
# --------------------------------------------------------------------------- #
def free_cash_flow_yield(tempfcf, tempmcap, nq):
    """FCF / marketCap, head(nq) mean (postBoRank.py:234)."""
    return (tempfcf / tempmcap).head(nq).mean()


def free_cash_flow_per_share_growth(tempfcf, tempshares, nq):
    """YoY (4-quarter) growth of FCF-per-share, head(nq) mean (postBoRank.py:240-242)."""
    fcfps = tempfcf / tempshares
    return fcfps.pct_change(-4, fill_method=None).head(nq).mean()


def tbv_p_ratio(tempcdx, nq):
    """Tangible book value per share / price, head(nq) mean (postBoRank.py:304-306)."""
    return (tempcdx["tangibleBookValuePerShare"] / tempcdx["price"]).head(nq).mean()


def eps_to_eps_mean(tempcdx):
    """Exponentially-weighted recent EPS vs its full-window mean (postBoRank.py:248-256).

    Returns 0 unless the four most-recent quarters are all positive.
    """
    eps = tempcdx["netIncome"] / tempcdx["weightedAverageShsOut"]
    epsmean = eps.mean()
    a = 0.4
    tw = a * (1 + (1 - a) + (1 - a) ** 2 + (1 - a) ** 3)
    # The `len(eps) >= 4` guard is a no-op for the live top-100 pool (always >= 4
    # quarters) but is load-bearing for the offline survivorship-clean reproduction,
    # whose dead-merged universe includes short-history names -- without it the
    # iloc[3] access below raises on a < 4-quarter name.
    if len(eps) >= 4 and all(eps.iloc[0:4] > 0):
        return epsmean - (a / tw) * (eps.iloc[0] + eps.iloc[1] * (1 - a) +
                                     eps.iloc[2] * (1 - a) ** 2 +
                                     eps.iloc[3] * (1 - a) ** 3)
    return 0


def price_growth(tempcdx, nq):
    """Per-period price appreciation, head(nq) mean (postBoRank.py:411-428).

    cdx is NEWEST-first, so pct_change(-1) = (newer - older)/older is POSITIVE
    when the price rose.  NO negation (the leading '-' was a sign bug removed in
    lockstep here and offline).  NaN when the price column is missing/empty.
    """
    if "price" in tempcdx.columns and not tempcdx["price"].empty:
        return tempcdx["price"].pct_change(-1, fill_method=None).head(nq).mean()
    return np.nan


def altman_z(tempcdx):
    """Altman-Z from fundamentals, most-recent row (postBoRank.py:310-349).

    Z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5.  NaN when unusable.
    """
    try:
        if len(tempcdx) >= 1:
            curr = tempcdx.iloc[0]
            ta = curr["totalAssets"]
            tl = curr["totalLiabilities"]
            if ta > 0 and tl > 0:
                x1 = (curr["totalCurrentAssets"] - curr["totalCurrentLiabilities"]) / ta
                x2 = curr["totalStockholdersEquity"] / ta
                x3 = curr["operatingIncome"] / ta
                x4 = curr["marketCap"] / tl
                x5 = curr["revenue"] / ta
                return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        return np.nan
    except Exception:
        return np.nan


def piotroski(tempcdx):
    """Piotroski F-score (9 binary criteria) from fundamentals, current vs prior
    row (postBoRank.py:353-402).  NaN when unusable.
    """
    try:
        if len(tempcdx) >= 2:
            curr = tempcdx.iloc[0]   # Most recent
            prev = tempcdx.iloc[1]   # Previous period
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
