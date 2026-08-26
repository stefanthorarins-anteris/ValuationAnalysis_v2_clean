"""
DERIVED TOTAL-RETURN PRICE SOURCE  --  the second, deep-coverage leg of the outcome
variable (offline, no network).

WHY THIS EXISTS
---------------
`returns_core.PriceSource` reads `price_data/real_prices*.csv`, which starts at
2018-12-31 and is an EXCHANGE-COVERAGE LOTTERY at the year-end grid dates: Korea is
0 of 211 names priced at 2018-12-31 (and 0% at every 12-31 grid date), London runs
42-91%.  Meanwhile `getData_fmp.py` ALREADY computes, per statement row,

    price = marketCap / weightedAverageShsOut

and writes it into `cdx_df`.  On the 2026-08-20 deep panel that column is populated on
141,780 of 143,839 rows, back to 1985, and it reaches years the price file does not.  The
backtest was ignoring it.  This module turns it into a drop-in second leg.

BUT THE COVERAGE CASE IS MUCH NARROWER THAN THE ROW COUNT SUGGESTS -- READ THIS BEFORE
CITING A COVERAGE GAIN (2026-08-21).  The guards below (listing-currency match, backfill
rejection) cut the usable panel from 2,645 sources to 2,009 and from 141,768 rows to
100,841, and much of the advantage that used to be claimed over the real route was
FABRICATED PRICES rather than coverage: per anchor 76-133 picks landed inside a
repeated-price run, and 71-88% of those are names the real file cannot price either.
Measured after the guards:

  * on the PANEL universe the composite still adds +245..+308 names at five of eight
    anchors -- but it is -23 at 2020-12-31 and -1 at 2022-12-30, because a ticker assigned
    to the derived leg loses its real-leg price at an anchor where the derived leg has a
    gap (the per-ticker all-or-nothing rule; see CompositePriceSource).
  * on the DEPLOYED top-100 grid the composite includes 3,218 names against the real
    route's 3,222.  It buys essentially NOTHING there.
  * what it does still buy, uniquely: 1,025 full 36-month pairs at 2008 and 1,296 at 2012,
    years the real price file does not reach at all, plus a dividend-inclusive total return
    on the live half.

That is the honest case for this leg: DEPTH OF HISTORY, not breadth at the recent anchors.

ONE THING TO READ BEFORE USING IT AS *THE* OUTCOME VARIABLE: the panel is SURVIVORS-ONLY,
so those coverage gains are gains within the LIVE universe.  A real backtest universe is
about half delisted names, which this leg cannot price at all -- see CompositePriceSource,
and use the 'derived+real' route for anything universe-wide.  Using the bare 'derived'
route on a universe containing delisted names REINTRODUCES SURVIVORSHIP BIAS.

WHY IT WAS WRITTEN OFF (a stale docstring, NOT a real finding)
-------------------------------------------------------------
`fetch_prices.py` lines 13-17 still say the derived price is "SYNTHETIC
(getData_fmp.py:171 reconstructs it as priceEarningsRatio * netIncome/shares) ...
+/-15-23pp per-name error -- too coarse."  That describes a formula the code NO LONGER
USES, and the error figure belongs to that retired version.  The shipped column is
`marketCap / weightedAverageShsOut` (verified bit-for-bit on all 141,780 populated rows,
2026-08-20).  `fetch_prices.py` is CEO-reserved and is deliberately NOT edited here.

THE DESIGN DECISION THAT MATTERS: THIS EMITS A TOTAL-RETURN INDEX LEVEL, NOT A PRICE
-----------------------------------------------------------------------------------
`returns_core.compute_returns` computes `p_eval / p_buy - 1` and relies on the price
source handing it a DIVIDEND-ADJUSTED level (that is exactly what `adjClose` is).  A raw
price ratio is a PRICE return and understates a total return by the yield -- measured
median 3.5%-4.4% over 36 months, and CONCENTRATED IN THE PAYERS, i.e. precisely the
population a value filter selects.

So this source does not return `price`.  It returns

    level_t = price_t * PROD_{s <= t} (1 + y_s),      y_s = dividend-per-share_s / price_s

which is the standard chain-linked total-return index:

    level_t / level_{t-1} = (price_t / price_{t-1}) * (1 + D_t/P_t) = (P_t + D_t) / P_{t-1}

i.e. FULL REINVESTMENT of each period's dividend at that period's end price.  Two
consequences worth stating:

  * `returns_core.compute_returns` is UNTOUCHED.  The terminal-value policy, the floor
    policy, the beat-rate and every derived view keep working, because the level is
    interface-identical to an adjClose.
  * The window convention falls out for free.  A ratio of cumulative products over
    (buy, eval] automatically excludes a dividend paid in the buy period (already gone
    when you bought at the buy-period-end price) and includes one paid in the eval
    period.  No off-by-one to get wrong, and it is correct for RAGGED period ends.

REINVESTMENT CONVENTION -- CHOSEN AND MEASURED (2026-08-20)
----------------------------------------------------------
The prototype this replaces used a SIMPLE SUM of period yields with NO compounding and a
per-period CLIP at 25%, and left a residual rank-IC OVERSHOOT of about +0.009.  Measured
head-to-head over 5 anchor pairs x 7 signals = 35 cells (residual = derived IC - real IC),
re-measured 2026-08-21 under the FIXED configuration (start-cap denominator, ceiling 1.0,
currency and backfill guards on).  Reproduce with
`python baseline_tools/derived_price_validate.py`:

    convention                       mean     median   mean|.|   max|.|
    price only (no dividend)       -0.0202   -0.0201   0.0202   0.0452
    additive sum   + clip          +0.0029   +0.0033   0.0092   0.0205   <- the prototype
    additive sum   + reject        +0.0040   +0.0043   0.0096   0.0216
    multiplicative + clip          +0.0026   +0.0016   0.0076   0.0175
    multiplicative + reject        +0.0028   +0.0031   0.0074   0.0166   <- SHIPPED

MULTIPLICATIVE is shipped because it is the correct algebra, not because of these margins:
`g + sum(y)` is not a total return and `g * prod(1+y)` is (see the chain-link identity
above).  It is also lowest on both dispersion columns.

BE HONEST ABOUT REJECT-vs-CLIP: it is NO LONGER LOAD-BEARING.  With the denominator fixed
and the ceiling moved to the plausibility edge, the ceiling only trips 56 rows instead of
188, so the two differ by ~0.0002 on the mean and the ranking between them flips column to
column.  Reject is kept because its failure direction is bounded and known (it can only
under-credit a real dividend, never manufacture return), not because it measurably wins.

`reject` rather than `clip` is the second half of the choice.  The per-period yield is
`dividendsPaid / marketCap` and its tail is VENDOR UNIT ERROR, not dividends: TCX
2012-12-31 carries marketCap $58M against dividendsPaid $8.7 TRILLION (y = 149,761);
STNG 2010 and IBKR 2008 are the same shape.  188 of 63,321 dividend-paying rows exceed
y > 0.25, and the median |dividendsPaid|/marketCap among y>1 rows is 2.6 -- i.e. "paid
260% of market cap".  CLIPPING such a row keeps 25% of a garbage number and COMPOUNDS it
(0.25 per quarter over 12 quarters is 1.25^12 = 14.6x of manufactured return) on names
that are payers by construction.  Rejecting it to zero is a bounded error in a KNOWN
direction (understates by at most one real period dividend, ~1-3%).  Rejected rows are
counted and reported, never silently dropped -- see `diagnostics()`.

WHAT THE DIVIDEND TERM DOES AND DOES NOT FIX (measured, 2026-08-20)
------------------------------------------------------------------
It closes the level bias on the MEDIAN, and the median is the statistic that misled this
work for a day -- read the warning below before quoting any of it.  Price-only median log
gap runs -0.026 .. -0.045 per anchor pair; the TR leg takes that to 0.0000 .. 0.0011.

    !!! A MEDIAN CANNOT SEE A DEFECT THAT LIVES IN A SUBPOPULATION.  The original version
    of this module shipped with a currency defect affecting ~20% of names at a mean log gap
    near -0.12, and BOTH the median gap and the acceptance test built on it read 0.0000
    throughout.  The per-anchor MEAN gap is the honest headline, and after the fixes it is
    -0.0161 / +0.0119 / +0.0257 / -0.0025 / +0.0064 -- centred near +0.005 but NOT zero, and
    an order of magnitude larger than the median suggests.  `fidelity_table` reports mean
    and 5th/95th percentiles alongside the median for exactly this reason, and
    `currency_split_table` exists so the subpopulation is looked at directly.

Rank fidelity against the real leg, after the fixes: Spearman 0.9772 .. 0.9855 (price-only
0.9712 .. 0.9788), on 1,354-1,933 names per pair.

It does NOT fully close the rank-IC residual, and the reason is structural rather than a
dividend defect.  Splitting the 35 cells by whether the SIGNAL contains marketCap:

    signals with NO marketCap (ROE, margins, leverage) : residual -0.0000
    marketCap-denominated signals (E/P, B/P, FCF/P)    : residual +0.0067

A marketCap-denominated signal shares its denominator with this source's buy-leg price
(`marketCap / shares`), so any error in that one marketCap number moves the signal and
the measured return TOGETHER and manufactures a small positive IC.  It is a property of
pricing off marketCap, so it cannot be removed by a better dividend convention or by a
tighter timing filter (a `max_lag_days=7` filter moves it only +0.0048 -> +0.0032,
measured 2026-08-21 on the MC signals alone).
==> READ `IC_CAVEAT` BEFORE USING THIS LEG TO VALIDATE A VALUE SIGNAL.

ANCHORING: CALENDAR DEC-31, NEVER THE PRICE-FILE GRID DATE
----------------------------------------------------------
`returns_core.DEFAULT_ANCHORS` contains real trading dates -- 2022-12-30, 2023-12-29 --
because that is when the exchange was open.  Selecting "newest period end <= anchor"
against 2022-12-**30** pushes EVERY December filer back a full quarter, since their
period end is 2022-12-31.  Measured at that anchor: median lag 0 -> 91 days, and the
share of names more than 45 days stale goes 9.3% -> 97.4%.  So the cutoff here is always
Dec-31 of the anchor's YEAR, and the surviving timing noise is REPORTED rather than
hidden (`timing_report()`): median 0 days, mean 9.6-17.3, p90 31, and 9.0-10.1% of names
sitting more than 45 days before the anchor.

This is a 1-day forward snap (2022-12-30 -> 2022-12-31) and it is deliberate.  It is not
look-ahead: only the PRICE and DIVIDEND off that row are used -- never its fundamentals
-- and a price is a market observable on its own date even though the statement carrying
it is filed months later.  THIS SOURCE IS FOR THE OUTCOME VARIABLE ONLY.  Do not feed it
into a Stage-1 or Stage-2 input.

RAISED, NOT FIXED -- three things a reader of any number off this leg needs (2026-08-21)
--------------------------------------------------------------------------------------
1.  THE REAL ROUTE IS ITSELF CURRENCY-INCOHERENT, and this leg cannot fix that.  `adjClose`
    is in the LISTING currency while the URTH benchmark is USD, so for a USD-reporting,
    TSX-listed name the DERIVED value is the USD-correct one and the real leg is inflated by
    CAD depreciation -- worth roughly 3-4pp of the 26.3pp pooled 36-month excess.  The
    listing-currency guard here makes the derived leg CONSISTENT WITH the real leg; it does
    not make either correct.  A real fix needs a point-in-time FX series putting every leg
    and the benchmark in one currency, which this repo does not have.
2.  buy2020's rank-IC SIGN FLIPS between routes at 36 months (+0.0925 real -> -0.0541
    derived).  A degenerate anchor, but it is the largest ranking-level disagreement between
    the two routes and it has not been explained.
3.  THE POOLED-CLEAN 36-MONTH STATISTIC RESTS ON 37 PICKS OVER 2 OVERLAPPING ANCHORS.  One
    name at +/-1.0 moves the mean by 0.027, and a single name (CPH.TO) accounted for 0.026
    of the -0.122 that started this investigation.  State the n wherever the number is
    reported; it is not a population estimate.
"""

import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

#  `carveOut` lives in the REPO ROOT, not in baseline_tools, so the root has to be on the
#  path explicitly -- the same two-line idiom depth_horizon_grid uses.  Without it this
#  module imports only when the interpreter happens to be started from the repo root, which
#  is exactly the kind of cwd-dependence that makes a module fail under one pytest
#  invocation and pass under another.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import returns_core as rc

#  The repo's own exchange-suffix table, imported rather than copied so it cannot drift.
#  NOTE THE DIRECTION OF USE.  carveOut documents this table as a coarse PRIOR on
#  REPORTING currency and warns it is wrong for e.g. a USD-reporting issuer on the LSE.
#  Here it is used for the LISTING currency instead -- what the exchange actually quotes in
#  -- which is the thing a suffix does determine well, and the mismatch it detects is
#  exactly the reporting-vs-listing divergence carveOut is warning about.
from carveOut import SUFFIX_TO_CURRENCY

#  Default location of the deep fundamentals panel.  A GLOB, so the newest date-stamped
#  panel wins -- matching how the other baseline_tools resolve their HomeGDrive inputs
#  (none of which cross git).
_HOME = r"C:\Users\stefanthorarinsson\Documents\HomeGDrive"
DEFAULT_PANEL_GLOB = os.path.join(_HOME, "pipeline", "Bometric_dic-*.pickle")

#  Per-period yield above this is treated as a vendor unit error and REJECTED to zero
#  (not clipped -- see the module docstring).
#
#  RAISED 0.25 -> 1.0 (2026-08-21) TOGETHER WITH THE DENOMINATOR FIX.  At 0.25 over
#  period-END market cap the ceiling was rejecting GENUINE special dividends: ZIM
#  2022-06-30 read y=0.4197 and 2023-06-30 y=0.5165 and both were zeroed, and panel-wide
#  106 rows across 87 sources sat in the plausible (0.25, 1.0] band.  A value screen
#  selects payers, so that bit exactly where it hurts.
#
#  But the ceiling was COMPENSATING FOR THE REAL DEFECT, which was the denominator.  A
#  dividend paid while the price collapses is divided by the post-collapse cap and reads
#  far too large: ZIM Q2-22 paid $2.38B against an END cap of $5.67B (0.42) but a START cap
#  of $8.72B (0.27).  The yield is now taken over period-START market cap, which is what
#  the holder actually owned when the dividend was declared, and only then is the ceiling
#  raised.  Lifting the ceiling ALONE overshoots (ZIM +0.355 against a real +0.034).
#
#  1.0 is the plausibility edge, not a tuned number: a single period cannot distribute
#  more than the whole company's starting value except as a terminal liquidation, while the
#  vendor unit errors sit far above it (start-cap p99 = 0.095, p99.9 = 1.79, max 155,546;
#  80 rows / 34 sources exceed 1.0 and the median |dividendsPaid|/marketCap among them is
#  2.6, i.e. "paid 260% of market cap").
MAX_PERIOD_YIELD = 1.0

#  A pick more than this many days before the anchor is REPORTED as stale.  Reporting
#  threshold only -- it does NOT drop the name (use `max_lag_days` for that).
STALE_LAG_DAYS = 45

#  CARRY-FORWARD CAP -- the one place this source can silently invent a return.
#
#  The anchor rule is "newest period end <= Dec-31", which means a company that STOPS
#  FILING keeps its last known level at every later anchor, forever.  Read through
#  `compute_returns` that turns a delisting into `status='ok'` with a FLAT return, instead
#  of firing the terminal/floor policy that exists precisely for a name that stopped
#  trading.  Beyond a missed full annual cycle the name is not stale, it is gone.
#
#  550 days = 18 months: comfortably past any legitimate ragged filer (a January
#  fiscal-year end measured at Dec-31 is 334 days) while still tripping on a skipped
#  annual report.  Past the cap `price()` returns None, so the eval leg reads as MISSING
#  and the existing audited terminal-value policy applies.
#
#  STATUS: UNVERIFIED, KEPT ON REASONING ALONE (independent decomposition, 2026-08-21).
#  On this panel the cap is INERT: capped vs uncapped differ in 0 of 216 grid cells, and
#  the 33 drops it reports are panel-wide (source, anchor) pairs, NONE of which is a top-100
#  pick at any anchor.  So nothing in the grid measures it and the paragraph below is an
#  argument, not evidence.  It is kept because the failure it guards against is structural
#  and silent -- this panel is a LIVE universe whose delisted names arrive through the
#  delisted-registry path (`dead_merge.py`) rather than cdx_df, so point this leg at a panel
#  that DOES carry delisted names and an uncapped carry-forward would quietly suppress the
#  losers.  THIS IS NOT A SURVIVORSHIP FIX -- only a guard against fabricating a live
#  price; the delisted registry remains the survivorship mechanism.
MAX_CARRY_FORWARD_DAYS = 550

#  REQUIRE THE REPORTING CURRENCY TO MATCH THE LISTING CURRENCY.
#
#  This leg's price is `marketCap / weightedAverageShsOut`, and marketCap is in the
#  REPORTING currency -- so the derived return is a REPORTING-currency return.  `adjClose`
#  is in the LISTING currency, so the real leg's return is a LISTING-currency one.  When the
#  two differ the gap is not noise, it is the FX move over the window.
#
#  Measured on the 2,057 names both legs price at 2021->2024 (independent decomposition,
#  2026-08-21):
#      currency-MATCHED    n=1,601 (77.8%)   mean log gap  +0.0067
#      currency-MISMATCHED n=  456 (22.2%)   mean log gap  -0.1234
#  and the pair table is the FX story outright: USD-reported/CAD-listed n=174 mean -0.140,
#  CAD/USD +0.124, USD/EUR -0.151, CNY/USD +0.105.
#
#  This single defect drove 55% of the composite's error (revaluation on the shared names,
#  mean -0.0671).  It was invisible to the old acceptance gate because that gate asserted a
#  MEDIAN, and a median over a 78%-matched universe is 0.0000 by construction.
#
#  THE FIX HERE IS A RESTRICTION, NOT A CONVERSION: a mismatched name is handed to the real
#  leg instead.  A proper fix needs a POINT-IN-TIME FX series to put both legs in one
#  currency, which this repo does not have -- and `marketCap_usd` is NOT it (see the
#  docstring: it is a single spot rate per currency applied to all history).
REQUIRE_LISTING_CURRENCY_MATCH = True

#  REJECT A PRICE THAT EXACTLY REPEATS THE PREVIOUS PERIOD'S -- pre-listing backfill.
#
#  The vendor carries a placement/IPO price backwards across quarters the name did not
#  trade.  TORO: price 7.97 and marketCap 75,404,241.73 IDENTICAL on 5 consecutive quarters
#  (2021-12-31..2022-12-31) while real_prices.csv first prices it 2023-12-29.  HAFN: price
#  7.31 identical on 11 consecutive quarters while marketCap MOVES 2.65B -> 3.73B (so the
#  share count is being back-solved), first real price 2024-12-31.  Returns computed off
#  those buy legs are returns on a price that never existed, and the real route's `no_buy`
#  on such a name was CORRECT BEHAVIOUR rather than a coverage gap.
#
#  Prevalence: 5,755 of 141,768 rows (4.06%); 605 of 2,645 sources carry a run >= 3 and 332
#  a run >= 5.  Per anchor, 76-133 picks land inside a stale run -- and 71-88% of those are
#  names the real file cannot price either, so most of the apparent coverage advantage over
#  the real route was fabricated prices rather than coverage.
#
#  A SINGLE exact repeat is enough -- no run-length threshold.  `price` is a full-precision
#  float ratio of two large numbers, so two consecutive periods agreeing to the last bit is
#  a carried-forward value, not a market coincidence.
#
#  THE WHOLE RUN GOES, HEAD INCLUDED -- corrected 2026-08-21 after measuring that the
#  alternative does not work.  Keeping the run head and dropping only the repeats left TORO
#  priced at 7.97 at buy2021 AND buy2022 (the head, then carried forward into the next
#  anchor) while real_prices.csv has nothing for it until 2023-12-29 at 2.05.  The reason is
#  that in the observed cases the head is the EARLIEST row and is itself fabricated: the run
#  is a spin-off/placement reference price stamped BACKWARDS across quarters the name did not
#  trade, so there is no real observation in it to keep.  Dropping only the repeats also
#  achieved nothing on the grid statistic (-0.0045, from removing 6 names).
#
#  Both legs are dropped rather than just the buy leg, because a fabricated eval price is
#  equally wrong.  Cost of the stricter rule: a genuinely unchanged price loses both of its
#  rows, which for a full-precision float ratio is a case that essentially does not arise.
#
#  THE COMPARISON IS TO A RELATIVE TOLERANCE, NOT EXACT EQUALITY, and HAFN is why.  There the
#  vendor holds the price and BACK-SOLVES the share count against a marketCap that moves
#  (2.65B -> 3.73B over 11 quarters at "7.31"), so the quotient lands within ONE ULP rather
#  than bit-identical -- max relative deviation 2.2e-16, with only 7 of 10 adjacent pairs
#  exactly equal.  Exact equality caught TORO and missed HAFN entirely.
#
#  1e-9 is chosen off a PLATEAU, not tuned.  Counting only the REPEATS (not the head), rows
#  flagged run 5,755 (exact equality) -> 6,021 at both 1e-12 and 1e-9 -> 6,038 at 1e-6 ->
#  6,494 at 1e-3.  The flat stretch from 1e-12 to 1e-9 is float-representation noise being
#  absorbed; past 1e-6 it starts eating genuine small price moves.
#
#  Prevalence of the SHIPPED rule (whole run, 1e-9): 7,577 of 141,768 clean rows (5.34%)
#  across 1,066 sources, of which 6,668 rows are still present to be dropped by the time
#  this guard runs (the listing-currency guard has already removed the rest).
REPEATED_PRICE_REL_TOL = 1e-9

REJECT_REPEATED_PRICE = True

IC_CAVEAT = (
    "derived-leg IC caveat: a marketCap-denominated signal (E/P, B/P, FCF/P) shares its "
    "denominator with this source's price (marketCap/shares), which inflates measured "
    "rank-IC by ~+0.005 (measured 2026-08-20; signals without marketCap show -0.0003). "
    "Use the real-price route to validate a marketCap-denominated signal."
)

#  cdx_df columns this source needs.  Kept explicit so a panel missing one fails loudly
#  at construction instead of producing a silently-wrong level.
REQUIRED_COLS = ("source", "periodEndDate", "price", "marketCap",
                 "weightedAverageShsOut", "dividendsPaid", "reportedCurrency")


def _load_cdx(panel):
    """Accept a cdx_df, a full Bometric dict, or a path/glob to the panel pickle.

    Unpickling needs the repo on sys.path (the pickle references `exclusions`); callers
    running from the repo root, or via baseline_tools' own sys.path insert, already have
    it.
    """
    if isinstance(panel, pd.DataFrame):
        return panel
    if isinstance(panel, dict):
        return panel["cdx_df"]
    hits = sorted(glob.glob(panel)) if any(ch in panel for ch in "*?") else [panel]
    if not hits:
        raise FileNotFoundError("no panel pickle matched %r" % (panel,))
    with open(hits[-1], "rb") as f:      # newest by lexical date-stamped name
        obj = pickle.load(f)
    return obj["cdx_df"] if isinstance(obj, dict) else obj


def _listing_currency(symbol):
    """Currency the LISTING quotes in, from the ticker's exchange suffix.

    No suffix -> 'USD' (a bare FMP symbol is a US listing).  An unrecognised suffix returns
    None, which callers must treat as "unknown", NOT as a match.
    """
    if not isinstance(symbol, str) or "." not in symbol:
        return "USD" if isinstance(symbol, str) else None
    return SUFFIX_TO_CURRENCY.get(symbol.rsplit(".", 1)[1].strip())


class DerivedPriceSource:
    """A `returns_core.PriceSource`-compatible TOTAL-RETURN level source built from the
    fundamentals panel's own `price = marketCap / weightedAverageShsOut` column.

    Exposes exactly the protocol `returns_core.compute_returns` consumes --
    `price(ticker, anchor)` / `last_before(ticker, anchor)` -- plus `benchmark_series`
    (DELEGATED: the panel has no ETF rows, so the URTH benchmark must come from a real
    price source) and the diagnostics the harness reports.

    Parameters
    ----------
    panel            : cdx_df / Bometric dict / path / glob to the deep panel pickle.
    anchors          : anchor grid (default returns_core.DEFAULT_ANCHORS).  Anchor LABELS
                       may be trading dates (2022-12-30); the cutoff used is always
                       Dec-31 of the label's year.
    benchmark_source : a real PriceSource, used ONLY for `benchmark_series`.
    require_listing_currency_match : keep only rows whose reportedCurrency equals the
                       LISTING currency implied by the ticker suffix, so the derived leg is
                       never compared against an adjClose in a different currency.  See
                       REQUIRE_LISTING_CURRENCY_MATCH -- this is the single largest defect
                       found in this leg and the default is ON.
    reject_repeated_price : drop a row whose price exactly repeats the previous period's
                       (pre-listing backfill).  See REJECT_REPEATED_PRICE; default ON.
    strict_currency  : 61 of 2,645 sources change reportedCurrency
                       mid-history, and a price ratio spanning that switch is contaminated
                       by the FX rate.  True (default) keeps only each source's LATEST
                       contiguous same-currency run, so a cross-switch window resolves to
                       None (the name drops out) instead of returning a garbage return.
    max_period_yield : reject ceiling for the per-period yield (see MAX_PERIOD_YIELD).
    max_lag_days     : if set, a pick staler than this many days before the anchor is
                       withheld.  Raises return-level fidelity against the real leg
                       (Spearman 0.972 -> 0.984 at 7 days) at the cost of ~15% of names;
                       default None keeps every name and reports the staleness instead.
    max_carry_forward_days : hard cap on how long a non-filing name keeps its last level
                       (see MAX_CARRY_FORWARD_DAYS).  Past it `price` returns None so
                       compute_returns' terminal/floor policy fires instead of reporting a
                       fabricated flat return.  None disables the cap.
    """

    def __init__(self, panel, anchors=None, benchmark_source=None,
                 strict_currency=True,
                 require_listing_currency_match=REQUIRE_LISTING_CURRENCY_MATCH,
                 reject_repeated_price=REJECT_REPEATED_PRICE,
                 max_period_yield=MAX_PERIOD_YIELD, max_lag_days=None,
                 max_carry_forward_days=MAX_CARRY_FORWARD_DAYS,
                 yield_denominator="start"):
        if yield_denominator not in ("start", "end"):
            raise ValueError("yield_denominator must be 'start' or 'end', got %r"
                             % (yield_denominator,))
        self.anchors = list(anchors) if anchors is not None else list(rc.DEFAULT_ANCHORS)
        self._idx = {a: i for i, a in enumerate(self.anchors)}
        self.strict_currency = bool(strict_currency)
        self.require_listing_currency_match = bool(require_listing_currency_match)
        self.reject_repeated_price = bool(reject_repeated_price)
        #  'end' exists ONLY to reproduce the pre-fix defect for the guard progression.
        #  It is not a supported configuration -- see MAX_PERIOD_YIELD.
        self.yield_denominator = yield_denominator
        self.max_period_yield = float(max_period_yield)
        self.max_lag_days = max_lag_days
        self.max_carry_forward_days = max_carry_forward_days
        self._benchmark_source = benchmark_source

        cdx = _load_cdx(panel)
        missing = [c for c in REQUIRED_COLS if c not in cdx.columns]
        if missing:
            raise KeyError("panel cdx_df missing required column(s): %s" % (missing,))

        #  WHY EACH SOURCE WAS LOST, not just how many.  The guards below are RESTRICTIONS,
        #  so a name this leg cannot price is a name a composite hands back to the real leg
        #  -- and when the real leg is empty too, that is a REFUSAL that has to be
        #  publishable with a reason rather than appearing as a silent absence.  An
        #  aggregate count cannot say "Oslo refuses because its issuers report in USD on
        #  NOK-listed lines", and that sentence is the whole output of the gap-fill route
        #  on Oslo.
        self._drop_reason = {}
        self._n_panel_sources = int(cdx["source"].nunique())
        self._stage_remaining = set(cdx["source"].dropna().unique())

        df = self._stage("unusable_rows", self._clean(cdx))
        df, self._n_currency_switch_sources = self._guard_currency(df, strict_currency)
        df = self._stage("reporting_currency_switch", df)
        df, self._n_listing_mismatch_rows, self._n_listing_mismatch_sources =             self._guard_listing_currency(df, self.require_listing_currency_match)
        df = self._stage("currency_mismatch", df)
        df, self._n_backfill_rows = self._guard_repeated_price(df,
                                                              self.reject_repeated_price)
        df = self._stage("prelisting_backfill", df)
        df, self._n_yield_rejected = self._build_level(df)
        self._lut, self._lag = self._pick_anchors(df)
        self._n_rows = len(df)
        self._n_sources = int(df["source"].nunique())
        #  Survived every guard and still has no usable level at any anchor: an all-NaN or
        #  non-positive level, or every candidate pick past the staleness / carry-forward cap.
        for src in self._stage_remaining - {t for (t, _a) in self._lut}:
            self._drop_reason.setdefault(src, "no_anchor_pick")

    def _stage(self, reason, frame):
        """Record every source that disappeared at this guard, then advance the frontier."""
        now = set(frame["source"].dropna().unique())
        for src in self._stage_remaining - now:
            self._drop_reason[src] = reason
        self._stage_remaining = now
        return frame

    def drop_reason(self, ticker):
        """Why this leg cannot price `ticker` -- None if it can.

        'unusable_rows' | 'reporting_currency_switch' | 'currency_mismatch' |
        'prelisting_backfill' | 'no_anchor_pick' | 'not_in_panel'.
        """
        if any((ticker, a) in self._lut for a in self.anchors):
            return None
        return self._drop_reason.get(ticker, "not_in_panel")

    def drop_reason_counts(self):
        """{reason: n_sources} over the panel, biggest first."""
        out = {}
        for r in self._drop_reason.values():
            out[r] = out.get(r, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    # ----------------------------------------------------------------- build #
    @staticmethod
    def _clean(cdx, extra_cols=()):
        """Coerce, drop unusable rows, and dedup -- the ONE definition of those rules.

        `extra_cols` carries additional columns (e.g. the validation harness' signal
        inputs) through the same cleaning, so a caller never has to re-implement the dedup
        and risk drifting from what the shipped source actually did.
        """
        cols = (list(REQUIRED_COLS)
                + [c for c in extra_cols if c in cdx.columns and c not in REQUIRED_COLS])
        cols = list(dict.fromkeys(cols))
        df = cdx.loc[:, cols].copy()
        df["periodEndDate"] = pd.to_datetime(df["periodEndDate"], errors="coerce")
        for c in cols:
            if c not in ("source", "reportedCurrency", "periodEndDate"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        #  price <= 0 is not a price.  On the 2026-08-20 panel there are none, so this is
        #  a guard rather than a filter -- but a zero would make the yield term explode.
        df = df[df["price"].notna() & (df["price"] > 0) & df["periodEndDate"].notna()]
        #  DETERMINISTIC dedup.  12 (source, periodEndDate) pairs are duplicated on the
        #  2026-08-20 panel (52/53-week fiscal drift + restatements); keep the LAST row so
        #  a restatement wins, and so the result does not depend on panel row order.
        return (df.sort_values(["source", "periodEndDate"], kind="mergesort")
                  .drop_duplicates(["source", "periodEndDate"], keep="last")
                  .reset_index(drop=True))

    @staticmethod
    def _guard_currency(df, strict):
        """Restrict each source to its LATEST contiguous reportedCurrency run.

        A source that redenominates (61 of 2,645) has a price series whose ratio across
        the switch carries the FX rate, not a return.  `price` is marketCap/shares and
        marketCap is in the REPORTING currency (verified: marketCap/marketCap_usd
        reproduces the FX rate per currency -- KRW 1388, JPY 158, EUR 0.856), so the
        contamination is real rather than theoretical.
        """
        #  A MISSING currency must not look like a redenomination.  1,700 panel rows carry
        #  a NaN reportedCurrency, and `NaN != NaN` is True, so a raw comparison would open
        #  a spurious new run at every gap and (under strict) silently discard the history
        #  before it.  Fill within the source first, so only a REAL currency change splits.
        #  A source with NO currency anywhere still must not self-split, hence the sentinel.
        cur = (df.groupby("source", sort=False)["reportedCurrency"]
                 .transform(lambda s: s.ffill().bfill()).fillna("__NA__"))
        changed = (cur != cur.shift()) & (df["source"] == df["source"].shift())
        run = changed.groupby(df["source"]).cumsum()
        n_switch = int((run.groupby(df["source"]).max() > 0).sum())
        if strict:
            last_run = run.groupby(df["source"]).transform("max")
            df = df[run == last_run].reset_index(drop=True)
        return df, n_switch

    @staticmethod
    def _guard_listing_currency(df, require_match):
        """Keep only rows whose reportedCurrency == the LISTING currency of the ticker.

        The derived price is marketCap/shares in the REPORTING currency; adjClose is in the
        LISTING currency.  Where they differ, the derived-vs-real gap IS the FX move over
        the window -- measured mean log gap -0.1234 on the 22.2% mismatched names against
        +0.0067 on the matched ones.  An UNKNOWN suffix is treated as a mismatch: this is a
        restriction, and the mismatched name is handed to the real leg by the composite, so
        the conservative direction is to drop.
        """
        if not require_match:
            return df, 0, 0
        listing = df["source"].map(_listing_currency)
        bad = listing.isna() | (listing != df["reportedCurrency"])
        n_rows = int(bad.sum())
        n_src = int(df.loc[bad, "source"].nunique())
        return df[~bad].reset_index(drop=True), n_rows, n_src

    @staticmethod
    def _guard_repeated_price(df, reject):
        """Drop a row whose price EXACTLY repeats the previous period's (backfill).

Drops the ENTIRE run, head included -- see REJECT_REPEATED_PRICE for why keeping
        the head does not work (TORO stays priced at a pre-listing 7.97) and for the
        TORO/HAFN evidence and the 4.06% prevalence.
        """
        if not reject:
            return df, 0
        price, src = df["price"], df["source"]

        def _same(shift):
            other = price.shift(shift)
            close = (price - other).abs() <= REPEATED_PRICE_REL_TOL * other.abs()
            return close.fillna(False) & (src == src.shift(shift))

        in_run = _same(1) | _same(-1)        # a repeat, or the head of a run of repeats
        return df[~in_run].reset_index(drop=True), int(in_run.sum())

    def _build_level(self, df):
        """price * cumprod(1 + y) -- the chain-linked total-return level."""
        dp = df["dividendsPaid"]
        #  dividendsPaid is a cash OUTFLOW, so a payment is NEGATIVE (63,550 rows negative
        #  vs 398 positive on the 2026-08-20 panel).  The 398 positives are a vendor sign
        #  defect, not a dividend received, so they contribute nothing.
        #
        #  The yield is written over marketCap, not per-share, because the share divisor
        #  CANCELS EXACTLY here: (D/shares) / (marketCap/shares) == D/marketCap (verified,
        #  max abs difference 3e-11 across the panel).  This is NOT the marketCap-ratio
        #  shortcut `test_derived_prices` pins against -- that one is about the PRICE leg,
        #  where the divisor emphatically does not cancel (Spearman 0.9675 per-share vs
        #  0.8265 marketCap-only, because a quarter of the universe moves its share count
        #  by >25% over 36 months).
        #  DENOMINATOR IS PERIOD-**START** MARKET CAP, not period-end.  The dividend accrues
        #  to whoever held the company when it was declared, so the yield the holder earned
        #  is over the cap they owned at the START of the period.  Dividing by the END cap
        #  overstates the yield of any payer whose price fell during the period -- ZIM Q2-22
        #  reads 0.42 on the end cap and 0.27 on the start cap -- and it was that inflation,
        #  not real mega-dividends, that the old 0.25 ceiling was really suppressing.
        #
        #  The first row of each source has no prior cap and falls back to its own.  That is
        #  INERT for every window return: `F_eval / F_buy` cancels every period at or before
        #  the buy leg, and the first row can never be later than the buy leg.
        if self.yield_denominator == "end":
            denom = df["marketCap"].to_numpy()          # DEFECT REPRODUCTION ONLY
        else:
            mc_start = df.groupby("source", sort=False)["marketCap"].shift(1)
            denom = mc_start.where(mc_start > 0).fillna(df["marketCap"]).to_numpy()
        y = pd.Series(np.where(dp < 0, -dp.to_numpy(), 0.0) / denom,
                      index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rejected = int((y > self.max_period_yield).sum())
        y = y.where(y <= self.max_period_yield, 0.0)          # REJECT, do not clip
        df = df.assign(period_yield=y)
        df["div_factor"] = (df.groupby("source", sort=False)["period_yield"]
                              .transform(lambda s: (1.0 + s).cumprod()))
        #  NO CURRENCY CONVERSION HAPPENS HERE, deliberately.  There used to be a
        #  currency_mode='usd' that multiplied by marketCap_usd/marketCap; it was DELETED
        #  (2026-08-21) because that ratio is a SINGLE SPOT RATE per currency applied to all
        #  history (CAD 0.724390 on all 31,316 CAD rows, CNY 0.148610, JPY 0.006317), and a
        #  per-name constant CANCELS in a return ratio.  The mode therefore produced returns
        #  bit-identical to local while its docstring claimed it "INCLUDES FX" -- a mode that
        #  lied.  Currency coherence is handled by RESTRICTION instead
        #  (_guard_listing_currency); a real conversion needs a point-in-time FX series this
        #  repo does not have.
        df["level"] = df["price"] * df["div_factor"]
        return df, rejected

    def _pick_anchors(self, df):
        """Per anchor: the newest period end at or before Dec-31 of the anchor's YEAR."""
        lut, lag, n_dropped = {}, {}, [0]
        for a in self.anchors:
            cut = self._anchor_cutoff(a)
            pk = df[df["periodEndDate"] <= cut].groupby("source", sort=False).tail(1)
            days = (cut - pk["periodEndDate"]).dt.days
            for src, lvl, d in zip(pk["source"].to_numpy(), pk["level"].to_numpy(),
                                   days.to_numpy()):
                if not np.isfinite(lvl) or lvl <= 0:
                    continue
                if self.max_lag_days is not None and d > self.max_lag_days:
                    continue
                if (self.max_carry_forward_days is not None
                        and d > self.max_carry_forward_days):
                    n_dropped[0] += 1          # gone, not stale -> let terminal fire
                    continue
                lut[(src, a)] = float(lvl)
                lag[(src, a)] = int(d)
        self._n_carry_forward_dropped = n_dropped[0]
        return lut, lag

    @staticmethod
    def _anchor_cutoff(anchor):
        """Dec-31 of the anchor's calendar year -- NEVER the anchor's own trading date.

        See the module docstring: against the 2022-12-30 / 2023-12-29 grid dates a naive
        "period end <= anchor" rule pushes every December filer back a full quarter
        (median lag 0 -> 91 days; share >45d stale 9.3% -> 97.4%).
        """
        return pd.Timestamp(int(str(anchor)[:4]), 12, 31)

    # -------------------------------------------------------------- protocol #
    def price(self, ticker, anchor):
        """The TOTAL-RETURN LEVEL for `ticker` at `anchor` (None if unavailable).

        Named `price` because that is the injected-price-source protocol
        `returns_core.compute_returns` calls.  The VALUE is a dividend-adjusted level, so
        `price(eval)/price(buy) - 1` is a total return -- the same contract adjClose has.
        """
        return self._lut.get((ticker, anchor))

    def last_before(self, ticker, anchor):
        """Latest (anchor, level) strictly before `anchor`, for the terminal-value policy."""
        j = self._idx.get(anchor)
        if j is None:
            return None
        for k in range(j - 1, -1, -1):
            a = self.anchors[k]
            v = self._lut.get((ticker, a))
            if v is not None:
                return a, v
        return None

    def benchmark_series(self, symbol=rc.BENCHMARK_SYMBOL):
        """DELEGATED to `benchmark_source`.  The panel holds operating companies only --
        URTH is an ETF and files no statements, so the benchmark can never be derived."""
        if self._benchmark_source is None:
            raise RuntimeError(
                "DerivedPriceSource has no benchmark_source: the panel contains no ETF "
                "rows, so %r cannot be derived. Pass benchmark_source=<a real "
                "returns_core.PriceSource> to use benchmark_return with this leg."
                % (symbol,))
        return self._benchmark_source.benchmark_series(symbol)

    # ----------------------------------------------------------- diagnostics #
    def lag_days(self, ticker, anchor):
        """Days between the picked period end and the anchor's Dec-31 (None if no pick)."""
        return self._lag.get((ticker, anchor))

    def picked_period_end(self, ticker, anchor):
        """The STATEMENT DATE this leg priced `anchor` off (None if no pick).

        Exposed because "which filing did you use" is a different question from "how stale is
        it", and one consumer needs the identity rather than the distance: the anchor rule is
        "newest period end <= Dec-31", so a name that skipped a filing is priced at TWO
        anchors off the SAME statement.  The growth between those anchors is then exactly
        1.0 -- not a measurement that the price did not move, but no measurement at all.
        `HoleFilledPriceSource` refuses on that identity, which needs no threshold.
        """
        lag = self._lag.get((ticker, anchor))
        if lag is None:
            return None
        return self._anchor_cutoff(anchor) - pd.Timedelta(days=int(lag))

    def coverage(self):
        """DataFrame[anchor, n_names] -- the priced universe this leg offers per anchor."""
        counts = {a: 0 for a in self.anchors}
        for (_, an) in self._lut:
            counts[an] = counts.get(an, 0) + 1
        return pd.DataFrame([{"anchor": a, "n_names": counts[a]} for a in self.anchors])

    def timing_report(self):
        """Per-anchor staleness of the picked period end.  SURFACED, not hidden: ~9-10% of
        names sit more than STALE_LAG_DAYS before the anchor at every anchor."""
        per = {a: [] for a in self.anchors}
        for (_, an), v in self._lag.items():
            per.setdefault(an, []).append(v)
        rows = []
        for a in self.anchors:
            d = np.asarray(per[a], dtype=float)
            if d.size == 0:
                rows.append({"anchor": a, "n": 0})
                continue
            rows.append({"anchor": a, "n": int(d.size),
                         "median_lag": float(np.median(d)),
                         "mean_lag": float(d.mean()),
                         "p90_lag": float(np.percentile(d, 90)),
                         "max_lag": float(d.max()),
                         "pct_over_%dd" % STALE_LAG_DAYS:
                             float(100.0 * (d > STALE_LAG_DAYS).mean())})
        return pd.DataFrame(rows)

    def diagnostics(self):
        """Everything a report needs to state what this leg did, in one dict."""
        return {
            "n_rows": self._n_rows,
            "n_sources": self._n_sources,
            "strict_currency": self.strict_currency,
            "n_currency_switch_sources": self._n_currency_switch_sources,
            "require_listing_currency_match": self.require_listing_currency_match,
            "n_listing_mismatch_rows_dropped": self._n_listing_mismatch_rows,
            "n_listing_mismatch_sources": self._n_listing_mismatch_sources,
            "reject_repeated_price": self.reject_repeated_price,
            "n_backfill_rows_dropped": self._n_backfill_rows,
            "n_yield_rows_rejected": self._n_yield_rejected,
            "max_period_yield": self.max_period_yield,
            "yield_denominator": self.yield_denominator,
            "max_lag_days": self.max_lag_days,
            "max_carry_forward_days": self.max_carry_forward_days,
            "n_picks_dropped_carry_forward": self._n_carry_forward_dropped,
            "carry_forward_cap_status": "UNVERIFIED -- inert on this panel (0 of 216 grid "
                                        "cells differ); kept on reasoning, not evidence",
            "ic_caveat": IC_CAVEAT,
        }


class CompositePriceSource:
    """Derived leg where it exists, real leg everywhere else -- PER TICKER.

    WHY THIS IS THE ROUTE A BACKTEST ACTUALLY WANTS
    ----------------------------------------------
    The fundamentals panel is SURVIVORS-ONLY, and the backtest universe is not.  On the
    2026-08-20 grid roughly half of every scored anchor is delisted names (buy2018:
    1,153 live + 1,230 dead), and only 4 of the delisted registry's 9,277 symbols appear in
    `cdx_df` -- while 3,948 of them ARE priced by real_prices.csv.  So the pure derived
    route sends the entire dead half to `no_buy`, and every derived view EXCLUDES no_buy.
    That silently drops the losers: measured on the 36-month top-20 windows it moved the
    average return from +0.34 to +0.75 at buy2018.  The derived leg is not wrong there --
    it simply cannot see those names, and using it alone REINTRODUCES SURVIVORSHIP BIAS.

    THE SELECTION IS PER TICKER, NEVER PER ANCHOR -- and that is a correctness
    requirement, not a preference.  Taking a buy leg from one source and an eval leg from
    the other DOUBLE-COUNTS dividends: the real leg's adjClose is already back-adjusted,
    so mixing it with a derived level measured an IC residual of +0.0554 against +0.0028
    for the consistent construction (re-measured 2026-08-21; the gap WIDENED once the
    derived leg became currency-coherent, because the real buy leg then brings a currency
    of its own on top of the double-counted dividend).  A ticker with ANY derived pick therefore uses ONLY
    the derived source, and never falls back at another anchor; every other ticker uses
    only the real source.  Mixing within a window is impossible by construction.

    Mixing ACROSS names is what remains, and it is a real seam that belongs in any report
    built on this route.  The two legs are both total returns and agree at Spearman ~0.98
    after the guards, but "agree on the median" was NOT good enough -- see the correction
    below, which is the single most important thing on this class.

    WHAT THE COMPOSITE'S FIRST NUMBER ACTUALLY WAS -- A CORRECTED RECORD (2026-08-21)
    -------------------------------------------------------------------------------
    The composite initially moved the pooled-clean 36-month top-20 EXCESS from +0.263 (real
    route) to +0.141, and this docstring used to imply that was survivorship correction
    finally showing the losers -- i.e. that +0.141 was the better number.  AN INDEPENDENT
    DECOMPOSITION REFUTED THAT.  The move partitioned exactly, with no residual, into

        revaluation on the 37 shared names   -0.0671   (55%)
        mix-add, from 2 names (not 63)       -0.0550   (45%)
        mix-drop                              empty

    and BOTH channels were DEFECTS IN THIS LEG, not corrections:

      * the revaluation channel was overwhelmingly the CURRENCY MISMATCH -- a
        reporting-currency return compared against a listing-currency adjClose on ~20% of
        names.  Currency-matched routing alone takes it -0.0671 -> -0.0093.
      * the mix-add channel was PRE-LISTING BACKFILL: two names whose "buy price" was a
        carried-forward value for a quarter they did not trade.  The real route's `no_buy`
        on them was correct, and the backfill guard removes them.

    So +0.263 was NOT flattered by survivorship and +0.141 was NOT the better number.  The
    survivorship argument above is still sound as a reason the composite must exist -- the
    dead half genuinely cannot be priced by the derived leg -- but it was NOT what drove
    that particular delta, and the delta was not evidence for it.

    THE METHOD FAILURE WORTH REMEMBERING: the wrong read rested on "the median log gap is
    0.0000", which was true and blind.  The MEAN was -0.0671, and five name-windows carried
    the entire channel (CPH.TO buy2021, TSM buy2022, ZIM buy2021, YRD buy2022, DNG.TO
    buy2021 sum to -2.49 of -2.48).  A median cannot see that by construction, and the
    acceptance test asserted the median.

    THE CHANNEL, MEASURED CLOSING (pooled-clean 36-month top-20 excess, one guard at a time;
    reproduce with `--price-route derived+real` plus the `--derived-no-*` flags):

        real route (baseline)                        +0.2630
        composite, ALL THREE DEFECTS present         +0.1410   <- the number that misled us
          + period-START yield denominator           +0.1383
          + currency-matched routing                 +0.1946
          + yield ceiling 1.0                        +0.2033
          + backfill rejection            SHIPPED    +0.2627   <- residual -0.0003

    So the entire -0.122 was defect, with nothing left over: the corrected composite agrees
    with the real route to 0.0003 on this statistic.  The decisive window is buy2022, where
    the backfill guard takes the pick count from 20 to 18 -- exactly matching the real
    route -- and the return from +0.883 to +0.993 against the real route's +0.988.  Those are
    the two backfilled names.

    AND THAT AGREEMENT IS THE POINT, NOT A NULL RESULT.  `adjClose` is already a total
    return, so a CORRECT derived total return SHOULD agree with it wherever both legs price a
    name.  This leg's value was never a different answer on the overlap; it is coverage the
    real file does not have (see the coverage note in the module docstring, and read it
    honestly -- on the deployed top-100 grid that coverage advantage is now roughly nil).
    """

    def __init__(self, derived, real):
        self.derived, self.real = derived, real
        self.anchors = list(real.anchors)
        #  A ticker "belongs to" the derived leg if the derived leg can price it anywhere.
        self._derived_tickers = {t for (t, _) in derived._lut}

    def _for(self, ticker):
        return self.derived if ticker in self._derived_tickers else self.real

    def price(self, ticker, anchor):
        return self._for(ticker).price(ticker, anchor)

    def last_before(self, ticker, anchor):
        return self._for(ticker).last_before(ticker, anchor)

    def benchmark_series(self, symbol=rc.BENCHMARK_SYMBOL):
        """Always the REAL leg: URTH is an ETF and files no statements."""
        return self.real.benchmark_series(symbol)

    def diagnostics(self):
        d = dict(self.derived.diagnostics())
        d["route"] = "derived+real composite"
        d["n_tickers_on_derived_leg"] = len(self._derived_tickers)
        d["leg_selection"] = ("per TICKER (all-or-nothing); a window never mixes legs, "
                              "but different names in one average may use different legs")
        return d

    def timing_report(self):
        return self.derived.timing_report()


def _venue_of(symbol):
    """`'092730.KQ' -> '.KQ'`, `'META' -> '(none)'`.  The same venue key
    `price_grid_audit.suffix_of` uses, so the two modules' venue tables line up."""
    t = str(symbol)
    return "." + t.rsplit(".", 1)[1] if "." in t else "(none)"


def venue_listing_currency(venue):
    """Listing currency for a VENUE KEY rather than a symbol: `'.ST' -> 'SEK'`,
    `'(none)' -> 'USD'`, an unrecognised suffix -> None.

    A thin wrapper over `_listing_currency` so callers reporting per venue do not have to
    fabricate a symbol to look one up -- which is what the venue tables were doing, and it
    read as a typo rather than as a lookup.
    """
    return _listing_currency("X" if venue == "(none)" else "X" + venue)


class HoleFilledPriceSource:
    """The REAL leg with INTERIOR holes imputed from the derived leg -- as a GROWTH RATIO,
    never as a level.  Default OFF; it moves measured numbers.

    WHAT IT IS FOR (CEO, 2026-08-22).  A name priced at the buy anchor and missing at the
    eval anchor fires the terminal policy: `total_return_floor` reads -100% and the default
    beat-rate policy scores it a miss.  For an INTERIOR hole -- priced before AND after the
    gap -- a later price PROVES the company did not die, so that -100% is not a punishment
    for missing data, it is a FALSE READING that sandbags our own measured track record.
    177 names / 291 anchor-cells are in that state on the run machine's grid.

    WHY IT CANNOT INSERT THE DERIVED PRICE, WHICH IS WHAT WAS ASKED FOR
    ------------------------------------------------------------------
    A return is a RATIO, so both legs of it must be on ONE scale.  They are not.  `adjClose`
    is BACK-adjusted -- its level is today's price scaled backwards through the split and
    dividend chain -- while the derived level is `price * cumprod(1+y)`, accumulated FORWARD
    from the start of the panel.  Both are legitimate total-return levels and neither level
    means anything on its own.  MEASURED on the 18,108 (ticker, anchor) cells both legs
    price:

        derived_level / adjClose     p1 0.0100   p25 1.0000   p50 1.1690
                                     p75 1.5900  p99 9.4600
                                     min 0.0037  max 338.53
        share within +/-10% of 1.0:  36.06%

    So splicing a derived LEVEL into a real series manufactures a return of up to ~338x at
    the splice, on the very window the fix is meant to rescue.  That is the same defect class
    as the dividend double-counting `CompositePriceSource` had to be built around, and it is
    why this class exists in this shape instead.

    WHAT IT DOES INSTEAD -- THE DERIVED LEG SUPPLIES A RATIO
    -------------------------------------------------------
    For a hole at anchor H, with P = the latest real-priced anchor before H:

        g          = derived(H) / derived(P)          <- scale-free, both from ONE leg
        imputed(H) = real(P) * g                      <- expressed on the REAL scale

    A ratio of two derived levels cancels the derived scale exactly, so `g` is a pure
    total-return growth factor and `imputed(H)` lands on the real leg's own scale.

    THREE PROPERTIES, AND THE SECOND IS THE CORRECTNESS PROOF
    --------------------------------------------------------
    1. r(P -> H) == g - 1 exactly: the window that used to read -100% now reads the derived
       total return, which is what was asked for.
    2. RETURN-PRESERVING ACROSS THE HOLE.  For any later real anchor N,

           (1 + r(P->H)) * (1 + r(H->N)) == 1 + r(P->N)

       identically, because r(H->N) = real(N)/(real(P)*g) - 1.  So a window SPANNING the
       hole is bit-for-bit unchanged, and the derived leg only decides how the outer return
       is SPLIT between the two sub-windows.  It can therefore never INJECT return into the
       sample -- any error is a reallocation between adjacent sub-periods, bounded by the
       outer return.  Had any basis mixing crept in, this identity would fail; the test
       suite asserts it numerically rather than trusting the algebra written here.
    3. Both sub-window returns are ratios of same-scale quantities, so no dividend is
       counted twice: each factor covers a DISJOINT interval -- dividends in (P,H] land in
       `g`, dividends in (H,N] land in the complement.

    THE DISJOINTNESS ARGUMENT FOR GapFillPriceSource SURVIVES, and it had to be re-derived
    rather than assumed.  That class assigns whole TICKERS: derived only where the real leg
    prices the name at NO anchor.  This class only ever touches a ticker the real leg prices
    at TWO OR MORE anchors.  The two populations are disjoint by construction, so per-ticker
    exclusivity is untouched -- and inside this class every value is on the real scale, so
    there is no second basis for a window to mix.  Compose in this order: hole-fill the real
    leg FIRST, then hand the result to `GapFillPriceSource` as its real leg.

    WHAT IT DELIBERATELY DOES NOT TOUCH
    -----------------------------------
      * A name with NO real price at all -> no P to scale from -> stays `no_buy`.
      * A TRAILING gap (the series genuinely ends) -> no later real price, so it is not an
        interior hole -> stays `terminal`.  That distinction is the whole point of
        `terminal` and this must not erase it.
      * A LEADING gap (not yet listed) -> same, untouched.
      * A hole the derived leg cannot bridge (it must price BOTH P and H) -> left as a hole
        and counted as a refusal.

    WHAT IT CANNOT SEE
    ------------------
      * Whether the derived `g` is RIGHT.  Property 2 bounds the damage to a reallocation,
        not to zero.
      * A hole whose true return is dominated by something the panel misses entirely -- a
        delisting-and-relisting, or a reverse split the vendor mis-stamped.
      * Anything on the seven venues the real leg cannot price at all: those names have no
        real price anywhere, so they have no INTERIOR holes and are out of scope here by
        construction.  They are `GapFillPriceSource`'s business.
    """

    def __init__(self, real, derived):
        self.real, self.derived = real, derived
        self.anchors = list(real.anchors)
        self._idx = {a: i for i, a in enumerate(self.anchors)}
        self._lut = dict(real._lut)
        self._imputed = {}      # (ticker, anchor) -> dict(from_anchor, g, real_prev)
        self._refused = {}      # (ticker, anchor) -> reason
        self._fill()

    # ------------------------------------------------------------------ build #
    def _real_holes(self):
        """{ticker: [interior hole anchors]} -- priced BEFORE and AFTER the gap.

        Leading and trailing gaps are excluded HERE, at construction, rather than filtered
        downstream, so the `no_buy` / `terminal` distinction cannot be lost by an oversight
        somewhere else.
        """
        by_ticker = {}
        for (t, a) in self.real._lut:
            by_ticker.setdefault(t, set()).add(a)
        holes = {}
        for t, priced in by_ticker.items():
            idxs = sorted(self._idx[a] for a in priced if a in self._idx)
            if len(idxs) < 2:
                continue
            gap = [self.anchors[i] for i in range(idxs[0], idxs[-1] + 1)
                   if self.anchors[i] not in priced]
            if gap:
                holes[t] = gap
        return holes

    def _prev_real(self, ticker, anchor):
        """Latest anchor strictly before `anchor` where the REAL leg has a price."""
        j = self._idx[anchor]
        for k in range(j - 1, -1, -1):
            a = self.anchors[k]
            if (ticker, a) in self.real._lut:
                return a
        return None

    def _fill(self):
        for t, gaps in self._real_holes().items():
            for h in gaps:
                p = self._prev_real(t, h)
                if p is None:               # unreachable for an interior hole; belt
                    self._refused[(t, h)] = "no_prior_real_anchor"
                    continue
                d_h, d_p = self.derived.price(t, h), self.derived.price(t, p)
                if d_h is None or d_p is None:
                    self._refused[(t, h)] = "derived_leg_cannot_bridge"
                    continue
                if not (d_p > 0 and d_h > 0):
                    self._refused[(t, h)] = "non_positive_derived_level"
                    continue
                g = d_h / d_p
                if not np.isfinite(g) or g <= 0:
                    self._refused[(t, h)] = "non_finite_growth"
                    continue
                #  SAME-FILING REFUSAL.  The derived anchor rule is "newest period end <=
                #  Dec-31", so a name that skipped a filing is priced at BOTH anchors off the
                #  SAME statement and g comes out exactly 1.0.  That is not a measurement
                #  that the price held flat, it is NO measurement -- and imputing it would
                #  hand the hole a fabricated 0% return wearing the same clothes as a real
                #  one.  Caught on the identity of the filing, so there is no threshold to
                #  argue about; found by test_a_hole_the_derived_leg_cannot_bridge, which
                #  read 10.0 where it expected None.
                pe_h = getattr(self.derived, "picked_period_end", lambda *_a: None)(t, h)
                pe_p = getattr(self.derived, "picked_period_end", lambda *_a: None)(t, p)
                if pe_h is not None and pe_p is not None and pe_h == pe_p:
                    self._refused[(t, h)] = "same_filing_as_base_anchor"
                    continue
                base = self.real._lut[(t, p)]
                self._lut[(t, h)] = float(base * g)
                self._imputed[(t, h)] = {"from_anchor": p, "g": float(g),
                                         "real_prev": float(base)}

    # --------------------------------------------------------------- protocol #
    def price(self, ticker, anchor):
        return self._lut.get((ticker, anchor))

    def last_before(self, ticker, anchor):
        j = self._idx.get(anchor)
        if j is None:
            return None
        for k in range(j - 1, -1, -1):
            a = self.anchors[k]
            v = self._lut.get((ticker, a))
            if v is not None:
                return a, v
        return None

    def benchmark_series(self, symbol=rc.BENCHMARK_SYMBOL):
        """Always the REAL leg.  URTH is an ETF and is never hole-filled."""
        return self.real.benchmark_series(symbol)

    # ------------------------------------------------------------ diagnostics #
    def is_imputed(self, ticker, anchor):
        return (ticker, anchor) in self._imputed

    def imputation_report(self):
        """Per venue: holes filled, holes refused, and the growth factors applied.

        The `g` distribution is the thing to read.  A `g` far from 1 is not automatically
        wrong -- these are 12-month steps and some names really do triple -- but the tails
        are where an unmeasurable derived leg would show up, so they are reported rather
        than summarised away.
        """
        import pandas as pd
        rows = {}

        def _slot(sym):
            v = _venue_of(sym)
            return rows.setdefault(v, {"venue": v, "filled": 0, "refused": 0, "g": []})

        for (t, _a), rec in self._imputed.items():
            r = _slot(t)
            r["filled"] += 1
            r["g"].append(rec["g"])
        for (t, _a) in self._refused:
            _slot(t)["refused"] += 1
        out = []
        nan = float("nan")
        for v, r in rows.items():
            g = np.asarray(r["g"], dtype=float)
            out.append({"venue": v, "filled": r["filled"], "refused": r["refused"],
                        "g_median": round(float(np.median(g)), 4) if g.size else nan,
                        "g_p05": round(float(np.percentile(g, 5)), 4) if g.size else nan,
                        "g_p95": round(float(np.percentile(g, 95)), 4) if g.size else nan,
                        "g_max": round(float(g.max()), 4) if g.size else nan})
        return (pd.DataFrame(out).sort_values("filled", ascending=False)
                  .reset_index(drop=True))

    def refusal_counts(self):
        out = {}
        for reason in self._refused.values():
            out[reason] = out.get(reason, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def diagnostics(self):
        return {
            "route": "real, interior holes imputed from the derived GROWTH RATIO",
            "n_holes_filled": len(self._imputed),
            "n_holes_refused": len(self._refused),
            "refusal_reasons": self.refusal_counts(),
            "basis": ("imputed = real(prev_anchor) * derived(hole)/derived(prev_anchor); a "
                      "RATIO of two derived levels, so the derived scale cancels and every "
                      "value in this source is on the REAL scale"),
            "invariant": ("return-preserving across the hole: a window SPANNING a filled "
                          "hole is bit-for-bit unchanged, so this cannot inject return into "
                          "the sample -- only reallocate it between sub-periods"),
            "untouched": ("no_buy names (no prior real anchor) and TRAILING gaps (the series "
                          "genuinely ends) are deliberately left alone"),
        }


class GapFillPriceSource:
    """REAL leg everywhere it has a price; derived leg ONLY where the real leg is EMPTY.

    THIS IS NOT `CompositePriceSource`.  That class prefers the DERIVED leg wherever the
    derived leg exists, which reassigns ~2,000 names the real file prices perfectly well --
    and is why its headline had to be reconciled to -0.0003 against the real route before
    anyone could trust it.  This class is the CEO's instruction ("wire it in where real is
    empty") read literally: the real route is untouched on every name it can price, and the
    derived leg is a GAP FILL.

    THE ASSIGNMENT IS PER TICKER AND THE TWO SETS ARE DISJOINT BY CONSTRUCTION.  A ticker
    goes to the derived leg only if the real leg prices it at NO anchor.  So a window can
    never take one leg from each source -- which matters because the real leg's `adjClose`
    is already back-adjusted for dividends while the derived level chain-links them again,
    so a mixed window DOUBLE-COUNTS the dividend (measured IC residual +0.0554 mixed against
    +0.0028 consistent).  `CompositePriceSource` has to enforce that with an all-or-nothing
    rule; here it is free, because the derived-assigned set and the real-priceable set do
    not intersect at all.

    THE REFUSAL IS THE OUTPUT, NOT A FAILURE.  A real-empty ticker the derived leg also
    cannot price is REFUSED, and `assignment_report` names the reason.  The dominant reason
    is the listing-currency guard, and it stays on: it fixed a measured -0.1234 mean log gap
    on the 22.2% currency-mismatched subpopulation.  Oslo is full of USD reporters on
    NOK-listed lines, so Oslo refuses.  Substituting a number there would reintroduce
    exactly the defect that guard exists to remove.

    WHAT NOBODY CAN CHECK, AND IT IS THE HONEST LIMIT OF THE WHOLE ROUTE.  The derived leg's
    bias against the real leg is measurable ONLY on the OVERLAP -- names both legs price.
    This route uses the derived leg ONLY where the real leg is empty, i.e. exactly OFF that
    overlap.  The population where the bias is MEASURED and the population where the leg is
    USED are DISJOINT, so any bias correction carried across is an EXTRAPOLATION and must be
    labelled one.  `derived_price_validate.venue_currency_bias_table` exists to make that
    extrapolation inspectable: if the bias is flat across every measurable venue sharing a
    currency, carrying it to an unmeasured venue in that same currency is at least a
    supported guess; if it varies venue to venue, it is not.  NO CORRECTION IS APPLIED HERE.
    """

    def __init__(self, real, derived):
        self.real, self.derived = real, derived
        self.anchors = list(real.anchors)
        self._real_tickers = {t for (t, _a) in real._lut}
        self._derived_tickers = {t for (t, _a) in derived._lut}
        #  DISJOINT from _real_tickers by construction -- see the class docstring.
        self._gapfilled = self._derived_tickers - self._real_tickers

    def _for(self, ticker):
        return self.derived if ticker in self._gapfilled else self.real

    def price(self, ticker, anchor):
        return self._for(ticker).price(ticker, anchor)

    def last_before(self, ticker, anchor):
        return self._for(ticker).last_before(ticker, anchor)

    def benchmark_series(self, symbol=rc.BENCHMARK_SYMBOL):
        """Always the REAL leg: URTH is an ETF and files no statements."""
        return self.real.benchmark_series(symbol)

    def assignment_report(self, tickers):
        """Per ticker: which leg serves it, or WHY it is refused.

        DataFrame[ticker, venue, leg, reason]; `leg` is 'real' | 'derived_fill' | 'REFUSED'.
        `tickers` is the caller's universe, because a price source does not have one -- pass
        the panel's sources or the scored universe and the answer is about that population.
        """
        rows = []
        for t in tickers:
            if t in self._real_tickers:
                rows.append((t, _venue_of(t), "real", ""))
            elif t in self._gapfilled:
                rows.append((t, _venue_of(t), "derived_fill", ""))
            else:
                rows.append((t, _venue_of(t), "REFUSED",
                             self.derived.drop_reason(t) or "unknown"))
        return pd.DataFrame(rows, columns=["ticker", "venue", "leg", "reason"])

    def per_venue_counts(self, tickers):
        """The clear/refuse table per venue -- the number the CEO asked to see.

        `n_real_empty` is the only column this route can act on at all: where the real file
        has a price the route is inert by design, so a venue with n_real_empty == 0 tells you
        nothing about the derived leg either way.
        """
        rep = self.assignment_report(tickers)
        wide = (rep.pivot_table(index="venue", columns="leg", values="ticker",
                                aggfunc="count", fill_value=0)
                   .reindex(columns=["real", "derived_fill", "REFUSED"], fill_value=0)
                   .reset_index())
        wide["n_panel"] = wide[["real", "derived_fill", "REFUSED"]].sum(axis=1)
        wide["n_real_empty"] = wide["derived_fill"] + wide["REFUSED"]
        wide["pct_cleared_of_real_empty"] = [
            round(100.0 * d / e, 1) if e else float("nan")
            for d, e in zip(wide["derived_fill"], wide["n_real_empty"])]
        return wide.sort_values("n_real_empty", ascending=False).reset_index(drop=True)

    def refusal_reasons(self, tickers):
        """Refusal counts by (venue, reason).  A PUBLISHED refusal, not a substitution."""
        rep = self.assignment_report(tickers)
        ref = rep[rep["leg"] == "REFUSED"]
        if ref.empty:
            return pd.DataFrame(columns=["venue", "reason", "n"])
        return (ref.groupby(["venue", "reason"]).size().rename("n").reset_index()
                   .sort_values("n", ascending=False).reset_index(drop=True))

    def diagnostics(self):
        d = dict(self.derived.diagnostics())
        d["route"] = "real+derived gap-fill"
        d["n_tickers_real_priceable"] = len(self._real_tickers)
        d["n_tickers_derived_priceable"] = len(self._derived_tickers)
        d["n_tickers_gapfilled"] = len(self._gapfilled)
        d["leg_selection"] = ("REAL wherever the real leg prices the ticker at ANY anchor; "
                              "derived ONLY where it does not.  The two sets are disjoint, "
                              "so no window can mix legs.")
        d["bias_measurability"] = ("the derived-vs-real bias is measurable only on the "
                                   "OVERLAP, and this route uses the derived leg only OFF "
                                   "that overlap -- any correction carried across is an "
                                   "EXTRAPOLATION, never a measurement.  None is applied.")
        return d

    def timing_report(self):
        return self.derived.timing_report()


# --------------------------------------------------------------------------- #
#  LEVEL-BREAK REFEREE -- the derived leg as a second opinion on the real leg   #
# --------------------------------------------------------------------------- #
#
#  WHAT IT IS FOR (CEO, 2026-08-22, chosen over extending `contam_return_cap`, over
#  abstention, and over leaving it).  `SIMINN.IC` adjClose steps 0.2256 -> 9.23 across one
#  anchor and reads a +36,787% 36-month return; the derived leg reads +25.4% and is the sane
#  number.  A cap needs an invented threshold on the RETURN; a second leg DETECTS instead.
#
#  IT DETECTS AND PUBLISHES.  IT DOES NOT CORRECT.  That is a deliberate departure from the
#  instruction ("treat the real leg as suspect"), and it is measured, not cautious:
#
#   * OF THE 17 FLAGGED CELLS, 3 ARE THE DERIVED LEG'S FAULT.  `SNYR` 2021->2022,
#     `MAXENT-B.ST` 2023->2024 and `OBD.L` 2024->2025 all show g_derived == 1.000000
#     EXACTLY -- the derived anchor rule priced both ends off the SAME filing, so the derived
#     leg has no opinion at all.  Overriding the real leg there would corrupt a good price
#     with a fabricated flat.  Those three are separated out exactly, on the identity of the
#     filing (see `DerivedPriceSource.picked_period_end`), not by a threshold.
#   * AND AMONG THE 14 SURVIVORS IT IS STILL NOT ALWAYS THE REAL LEG.  `GDHG` 2023->2024 is
#     g_real 0.0557 against g_derived 0.0069: both legs collapse and they disagree only about
#     how far.  Nothing here can say which is right.
#
#  So a verdict column says which leg looks suspect and WHY, and the caller decides.  A
#  silent auto-override would have shipped three known-wrong corrections.
#
#  THE THRESHOLDS ARE A CHOICE, NOT A DISCOVERED BAND -- and this is where the method that
#  produced MIN_PAYLOAD_ROWS does NOT transfer.  There, two populations were genuinely
#  separated: bad bodies topped out at 15,441 and good ones started at 45,662, a factor of
#  3.0 with nothing in between.  Here the disagreement distribution is CONTINUOUS.  Measured
#  on 15,530 single-period cells (2,555 symbols) where both legs price both ends:
#
#      |log(g_real) - log(g_derived)|   p50 0.0156  p90 0.1358  p99 0.6611
#                                       p99.5 0.8630  p99.9 1.9693  max 5.5114
#      widest jump in the top 0.5%      0.78, between 4.06 and 4.84  <- density thinning,
#                                                                        not a band
#
#  A big REAL move is usually real: of the 392 cells whose real step is >= 5x, 354 (90.3%)
#  have a log gap below 0.5, i.e. the derived leg corroborates them.  That is what makes the
#  CONJUNCTION the right instrument -- extremeness alone flags 392 cells, almost all of them
#  genuine volatility.  But within the >= 5x population the gap is continuous too, so where
#  the line falls is a judgement.  It is set where the flagged set stays small enough to read
#  by hand and still contains the case that motivated the work:
#
#      step >= 5 AND |log gap| >= 1.0  ->  17 cells / 15 symbols / 0.109% of the corpus
#      the same rule at |log gap| >= 0.5 ->  272 cells / 150 symbols / 1.751%   (too many
#                                            to inspect, and mostly corroborated moves)
#
#  `SIMINN.IC` 2022->2023 sits at 3.783, comfortably inside.
#
#  THE CURRENCY GUARD STAYS ON, and the warning that it must not was checked and refuted.
#  The concern was that the guard is a MATCH test rather than a MAGNITUDE test, so it could
#  hide the case that motivated the referee.  It does not: `SIMINN.IC` reports ISK on an
#  ISK-listed line, so it is currency-MATCHED and survives the guard.  Measured both ways --
#  guard ON flags 17 cells / 15 symbols; guard OFF flags 30 / 28, and ALL 13 of the added
#  names are currency-MISMATCHED, i.e. their "disagreement" is the FX move over the period
#  and not a level break at all.  Un-gating the referee would therefore add 13 false
#  positives to catch nothing.
LEVEL_BREAK_MIN_REAL_STEP = 5.0
LEVEL_BREAK_MIN_LOG_GAP = 1.0


def level_break_candidates(real, derived, anchors=None,
                           min_real_step=LEVEL_BREAK_MIN_REAL_STEP,
                           min_log_gap=LEVEL_BREAK_MIN_LOG_GAP):
    """Single-period cells where the REAL leg makes an extreme move the derived leg does not
    corroborate.  A DETECTOR: it returns evidence and a verdict, and corrects nothing.

    Returns DataFrame[symbol, venue, from_anchor, to_anchor, g_real, g_derived, log_gap,
    real_step, derived_same_filing, verdict] sorted by |log_gap| descending.

    verdict is one of:
      'derived_uninformative' -- the derived leg priced BOTH ends off the same filing, so
                                 g_derived is spuriously ~1.0 and it has no opinion.  Do NOT
                                 act on the real leg here.
      'legs_disagree'         -- a genuine two-leg disagreement.  The real leg is the usual
                                 suspect (that is the motivating case) but this does not
                                 establish it: among these, GDHG 2023->2024 has both legs
                                 collapsing and disagreeing only about how far.

    WHAT IT CANNOT SEE
    ------------------
      * ANY name only one leg prices -- which is most of the seven venues the real grid
        cannot price at all.  The corpus is 2,555 of the panel's 4,954 sources, and
        `.PA/.KS/.OL/.KQ/.BR/.AS/.LS` contribute ZERO cells.  A referee needs two opinions;
        there is only one there.
      * WHICH leg is wrong.  It reports a disagreement.  3 of the 17 flagged cells are the
        derived leg's fault and are labelled as such; the rest are not adjudicated.
      * A break both legs share -- a vendor error that hits marketCap and adjClose together
        moves them in lockstep and reads as agreement.
      * A break SPREAD over more than one period: this compares ADJACENT anchors only, so a
        two-step break is diluted below the threshold at each step.
      * Anything below the thresholds, which are a choice on a continuous distribution and
        not a separating band.  The count at the shipped cut is 0.109% of cells; nothing
        makes that the right number rather than a readable one.
    """
    import pandas as pd
    anchors = list(anchors if anchors is not None else real.anchors)
    rows = []
    for i in range(len(anchors) - 1):
        a, b = anchors[i], anchors[i + 1]
        for t in {t for (t, an) in derived._lut if an == a}:
            r_a, r_b = real.price(t, a), real.price(t, b)
            d_a, d_b = derived.price(t, a), derived.price(t, b)
            if None in (r_a, r_b, d_a, d_b):
                continue
            if min(r_a, r_b, d_a, d_b) <= 0:
                continue
            g_real, g_der = r_b / r_a, d_b / d_a
            step = max(g_real, 1.0 / g_real)
            gap = abs(np.log(g_real) - np.log(g_der))
            if step < min_real_step or gap < min_log_gap:
                continue
            pe_a = getattr(derived, "picked_period_end", lambda *_x: None)(t, a)
            pe_b = getattr(derived, "picked_period_end", lambda *_x: None)(t, b)
            stale = bool(pe_a is not None and pe_b is not None and pe_a == pe_b)
            rows.append({
                "symbol": t, "venue": _venue_of(t), "from_anchor": a, "to_anchor": b,
                "g_real": round(float(g_real), 6), "g_derived": round(float(g_der), 6),
                "log_gap": round(float(gap), 4), "real_step": round(float(step), 3),
                "derived_same_filing": stale,
                "verdict": "derived_uninformative" if stale else "legs_disagree",
            })
    df = pd.DataFrame(rows, columns=["symbol", "venue", "from_anchor", "to_anchor",
                                     "g_real", "g_derived", "log_gap", "real_step",
                                     "derived_same_filing", "verdict"])
    if df.empty:
        return df
    return df.sort_values("log_gap", ascending=False).reset_index(drop=True)


def level_break_report(real, derived, anchors=None, **kw):
    """The referee's headline: how many cells, how many are actionable, and the corpus it
    could see at all.  Reported so a small flagged count is never read as a clean bill --
    it may just mean the referee had no second opinion to offer."""
    cand = level_break_candidates(real, derived, anchors=anchors, **kw)
    corpus = {t for (t, _a) in derived._lut} & {t for (t, _a) in real._lut}
    return {
        "n_cells_flagged": int(len(cand)),
        "n_symbols_flagged": int(cand["symbol"].nunique()) if len(cand) else 0,
        "n_legs_disagree": int((cand["verdict"] == "legs_disagree").sum()) if len(cand) else 0,
        "n_derived_uninformative": (int((cand["verdict"] == "derived_uninformative").sum())
                                    if len(cand) else 0),
        "n_symbols_both_legs_price": len(corpus),
        "min_real_step": kw.get("min_real_step", LEVEL_BREAK_MIN_REAL_STEP),
        "min_log_gap": kw.get("min_log_gap", LEVEL_BREAK_MIN_LOG_GAP),
        "action": ("DETECT AND PUBLISH ONLY.  No price is overridden: 3 of the 17 cells on "
                   "the run machine grid are the DERIVED leg's fault (g_derived == 1.0 "
                   "exactly, same filing both ends), so an automatic 'real is suspect' rule "
                   "would ship known-wrong corrections."),
        "blind_to": ("any name only ONE leg prices -- including all seven venues the real "
                     "grid cannot price, which contribute zero cells; a break both legs "
                     "share; a break spread across more than one period; and everything "
                     "below a threshold that is a choice on a continuous distribution, not "
                     "a separating band."),
    }


# --------------------------------------------------------------------------- #
#  ROUTE SELECTOR -- the side-by-side switch                                  #
# --------------------------------------------------------------------------- #
#  'derived+real' is the derived-PREFERRED composite; bare 'derived' is the clean single-leg
#  route for auditing the derived leg itself, and is survivors-only.  'real+derived' is the
#  GAP-FILL route -- real wherever real has a price, derived only where it does not.  The
#  LEADING leg is the preferred one in both composite names.
#
#  'real' REMAINS THE DEFAULT EVERYWHERE.  A second route existing does not change which one
#  runs; it has to be selected explicitly (`--price-route`, or configdic['price_route']).
PRICE_ROUTES = ("real", "derived", "derived+real", "real+derived")


def build_price_source(route, prices_csv=None, supp_csv=None, panel=None, anchors=None,
                       fill_interior_holes=False, **derived_kw):
    """Build the outcome-variable price source for `route`.

    'real'    -> returns_core.PriceSource over real_prices*.csv.  UNCHANGED and still the
                 default everywhere, so the certified numbers stay bit-for-bit.
    'derived' -> DerivedPriceSource over the fundamentals panel, with the real source
                 injected as its benchmark provider (the panel has no ETF rows).
                 SURVIVORS-ONLY -- see CompositePriceSource before using it on a universe
                 that contains delisted names.
    'derived+real'
              -> CompositePriceSource: the derived leg per ticker where it exists, the real
                 leg for everything else (i.e. the delisted half).  This is the route for a
                 universe-wide backtest.
    'real+derived'
              -> GapFillPriceSource: the REAL leg wherever it prices the ticker, the derived
                 leg ONLY where the real leg is empty.  Strictly less invasive than
                 'derived+real' -- it cannot move a name the real file already prices -- and
                 it is the route for "the refetch is deferred, fill what can be filled".
                 Read GapFillPriceSource on why its REFUSALS are the output.

    `fill_interior_holes` (default FALSE, because it MOVES MEASURED NUMBERS) wraps the real
    leg in HoleFilledPriceSource first: an anchor the real leg is missing BETWEEN two it has
    is imputed from the derived leg's GROWTH RATIO, so a name a later price proves was alive
    stops reading -100%.  It applies on 'real' and on 'real+derived'; the composition order
    is hole-fill FIRST, then gap-fill, and the two populations are disjoint by construction
    (hole-fill needs >= 2 real anchors, gap-fill needs ZERO), so per-ticker exclusivity
    holds.  On the derived-preferred routes it is refused rather than silently ignored --
    there the derived leg already owns those names and a hole fill would be meaningless.

    Keeping all routes selectable is the point: it is what makes the derived leg's fidelity
    against the real leg auditable instead of asserted.
    """
    if route not in PRICE_ROUTES:
        raise ValueError("route must be one of %s, got %r" % (PRICE_ROUTES, route))
    real = None
    if prices_csv is not None:
        real = rc.PriceSource(prices_csv, anchors=anchors, supp_csv=supp_csv)
    if fill_interior_holes and route in ("derived", "derived+real"):
        #  REFUSED, not ignored.  On a derived-preferred route the derived leg already owns
        #  every ticker it can price, so "fill the real leg's holes" is either a no-op or an
        #  instruction whose meaning nobody has defined.  Silently dropping the flag would
        #  let a caller believe holes were filled when they were not.
        raise ValueError("fill_interior_holes is only meaningful on a REAL-preferred route "
                         "('real' or 'real+derived'), got %r" % (route,))
    if route == "real" and not fill_interior_holes:
        if real is None:
            raise ValueError("route='real' needs prices_csv")
        return real
    if panel is None:
        raise ValueError("route=%r needs panel" % (route,))
    derived = DerivedPriceSource(panel, anchors=anchors, benchmark_source=real,
                                **derived_kw)
    if route == "derived":
        return derived
    if real is None:
        raise ValueError("route=%r needs prices_csv for the other leg" % (route,))
    #  HOLE-FILL FIRST.  Gap-fill assigns whole tickers off the real leg's coverage, so it
    #  must see the POST-fill real leg or it would judge coverage on a stale picture.  (In
    #  fact hole-fill never changes WHICH tickers the real leg prices, only at which
    #  anchors, so the order is belt-and-braces -- but the dependency runs this way and
    #  writing it in the other order would be a latent bug the day that changes.)
    if fill_interior_holes:
        real = HoleFilledPriceSource(real, derived)
    if route == "real":
        return real
    if route == "real+derived":
        return GapFillPriceSource(real, derived)
    return CompositePriceSource(derived, real)
