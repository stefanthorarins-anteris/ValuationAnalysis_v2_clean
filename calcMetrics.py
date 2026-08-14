import pandas as pd
import numpy as np

import nan_policy as npol
import reporting_period as rp

# `rpy` = this source's rows per year (4 quarterly / 2 semi-annual, reporting_period).
# It scales the moving-average window so it spans the same CALENDAR time, and it is the
# divisor for any ANNUAL rate expressed per period.  rpy defaults to 4 -> unchanged.

# =========================================================================== #
#  STAGE-1 DOMAIN GUARDS (sign-inversion fix, 2026-08-04)                      #
# =========================================================================== #
# A ratio whose ADVERSE quantity sits in the DENOMINATOR does not fail -- it INVERTS SIGN and
# scores as the best possible value.  Where the ratio can be rewritten in yield form that is
# the better fix and no guard is needed (see createDicts's module comment); where it cannot,
# the out-of-domain rows are REFUSED here.
#
# A guard is a predicate on the RAW statement frame returning the ADMISSIBLE rows.  Refused
# rows become NaN, and `calcScore.calcByTier` already scores NaN as a fail (`Sign * NaN > 0`
# is False), so no new scoring state is introduced -- which is the reason this is a mask and
# not a new branch in the scorer.
#
# WHY REFUSAL RATHER THAN A TRANSFORM OF THE PERVERSE VALUE.  Three candidates were
# considered -- negate the value, impute the column's floor, or refuse:
#   * NEGATION IS NOT SIGN-SAFE.  It flips the value across zero, which lands on the FAIL side
#     for a Sign +1 (higher-is-better) criterion but on the PASS side for a Sign -1 one.
#     MEASURED on the head(8) window: negating `returnOnEquity`'s 2,426 double-negative rows
#     takes them from 2,255 passes to 2 (correct), while negating `uNetDebtToEBITDA`'s 9,200
#     takes them from 5,280 passes to 9,200 -- i.e. it makes every one of them PASS.  A single
#     blanket transform cannot serve both signs.
#   * A BOUNDARY/FLOOR IMPUTATION DOES NOT APPLY.  The project's boundary rule imputes the
#     LIMIT of the metric as the input approaches its domain edge, and admits it only where
#     that limit is FINITE.  For every ratio here the limit is +/-infinity as the denominator
#     -> 0, so that rule's own escape clause applies: REFUSE, do not impute.
#   * AND AT STAGE-1 THE CHOICE IS MOOT ANYWAY.  calcByTier returns `w if pass else 0` -- a
#     BINARY outcome, with no ranking for a magnitude to inform.  Any treatment that lands on
#     the fail side is behaviour-IDENTICAL.  So refusal is the cheapest correct option and adds
#     no tuned constant.  (The choice would only become live in Stage-2, which RANKS; the one
#     Stage-2 metric in this family is `returnOnEquity` -- see stage2_metrics.postbm_metric.)
#
# A NaN in a guard's own input makes the row INADMISSIBLE: an undetermined domain is not a
# licence to score.  That is `fillna(False)` in apply_domain_guard, not an accident.
STAGE1_DOMAIN_GUARDS = {
    # Book equity must be POSITIVE for any ratio that divides by it, or leverage and return
    # ratios inverse-scale and change sign.  `> 0` and not `>= 0`: zero equity is division by
    # zero, and the pre-existing ruling on it (createDicts freeCashFlowToEquity) is that NaN
    # describes a technically-insolvent balance sheet better than either infinity.
    'equity_positive':
        lambda df: pd.to_numeric(df['totalStockholdersEquity'], errors='coerce') > 0,
    # EBITDA PROXY = operatingIncome + depreciationAndAmortization.  FMP publishes
    # `netDebtToEBITDA` but not the EBITDA behind it, so the guard reconstructs the sign from
    # the two statement lines it does publish.  A PROXY, stated as one: near zero the proxy and
    # the true EBITDA can straddle, so individual rows at the boundary may be classified
    # differently than FMP would.  The guard's DIRECTION does not depend on the proxy.
    'ebitda_positive':
        lambda df: (pd.to_numeric(df['operatingIncome'], errors='coerce')
                    + pd.to_numeric(df['depreciationAndAmortization'], errors='coerce')) > 0,
    # A tax RATE's admissible domain is >= 0.  Negative means the denominator (pre-tax income)
    # was non-positive or the expense was a credit; either way "tax efficiency" is not what the
    # number is measuring.  `>= 0` and not `> 0`: a zero rate is a real, defined answer (no tax
    # paid on positive income), and 9,326 head(8) rows are exactly zero -- refusing those would
    # be a much larger and unrelated change.
    'tax_rate_nonnegative':
        lambda df: pd.to_numeric(df['effectiveTaxRate'], errors='coerce') >= 0,
    # `uInterestCoverage` = operatingIncome / interestExpense (new criterion, 2026-08-05).
    # FMP reports `interestExpense` as 0 for a DEBT-FREE name, and a debt-free company does not
    # HAVE a coverage ratio -- the quantity does not exist rather than being large.  0 in the
    # denominator would give +/-inf -> NaN anyway; the guard states the domain instead of
    # relying on that accident, and it also refuses the (rare) NEGATIVE reported interest
    # expense, where the ratio inverts sign exactly like the rest of this family.
    # `> 0` and not `>= 0`: zero is the debt-free case, which is REFUSED here on purpose --
    # the leverage question for a debt-free name is carried by `netDebtToEBITDA`'s net-cash
    # branch, which passes it on an explicit operand condition.  Refusing never rewards.
    'interest_expense_positive':
        lambda df: pd.to_numeric(df['interestExpense'], errors='coerce') > 0,
    # `producerEbitdaPositive` (VETO column, 2026-08-07) asks whether a PRODUCING miner earns
    # anything at the EBITDA line.  With zero revenue there is nothing to earn it on: the name is
    # a PRE-PRODUCTION explorer and the cost-curve question this column asks does not exist yet,
    # so the row is REFUSED rather than read as a producer failing.  Counting a refusal as a
    # non-pass here would eject the exploration half of the Mining cohort FOR BEING EXPLORERS --
    # the `interest_expense_positive`-on-a-debt-free-name defect (measured: 1,668 sources, 21.5%
    # of the universe) in a new field.  Those names are judged by `cashRunwayOneYear` instead.
    # `> 0` and not `>= 0`: zero revenue IS the pre-production case, which is what is refused.
    # NOT A DENOMINATOR GUARD -- `producerEbitdaPositive` is a level, not a ratio -- so it cannot
    # be inverting a sign; it states the population the column is defined on.
    'revenue_positive':
        lambda df: pd.to_numeric(df['revenue'], errors='coerce') > 0,
    #  NO PEG ENTRY, DELIBERATELY -- see the `PEG` note in createDicts.BoMetric_special_dict and
    #  `peg_local` below.  PEG's domain is INTRINSIC TO ITS FORMULA (without a positive trailing
    #  EPS there is no P/E for a growth rate to be compared against, so the value does not exist
    #  rather than being inadmissible), so it is applied inside `peg_local` itself.  A `Guard`
    #  entry restating the same condition would be a SECOND registry of one fact -- and a fact
    #  stated twice, once with `rpy` in hand and once without it, is this repo's worst bug class.
}


# =========================================================================== #
#  PEG, COMPUTED LOCALLY  (CEO ruling, 2026-08-04; built 2026-08-05)           #
# =========================================================================== #
#  "In general, we should compute things we can rather than using the FMP."
#
#  WHAT THE VENDOR'S FIELD WAS, established arithmetically (no vendor docs exist), matched to
#  all printed digits on nine deliberately-seasonal quarters:
#
#      PEG_fmp = [ price / (4 * eps_t) ]  /  [ 100 * (eps_t / eps_{t-1} - 1) ]
#
#  Three things wrong with it, in the order that matters:
#
#   1. THE HORIZON IS THE NOISIEST ONE AVAILABLE, AND IT IS THE VENDOR'S CHOICE, NOT OURS.  The
#      growth leg is SEQUENTIAL quarter-over-quarter, so it is dominated by seasonality: a
#      retailer's Q4-vs-Q3 EPS move is the calendar, not growth.  For a filter that holds for 36
#      months that is close to indefensible.
#   2. IT IS DIMENSIONALLY INCOHERENT.  The PE leg is a quarter ANNUALISED (x4) while the growth
#      leg is a QUARTERLY percentage.  PEG is defined as "P/E divided by the annual growth RATE
#      in percentage points"; the vendor divides an annual P/E by a quarterly percentage.
#   3. IT IS QUANTISATION NOISE FOR SMALL CHANGES.  The growth leg differences TWO 2-DECIMAL-
#      ROUNDED `eps` figures, so for a small sequential change the denominator is mostly
#      rounding; 179 of 2,969 rows carry an exact-zero growth artifact.
#
#  WHAT IS COMPUTED HERE INSTEAD -- both legs TRAILING-YEAR, one horizon, full precision:
#
#      eps_ttm[t] = sum of the `rpy` most recent per-period EPS (t .. t+rpy-1, newest-first)
#      PE[t]      = price[t] / eps_ttm[t]
#      g[t]       = 100 * (eps_ttm[t] - eps_ttm[t - PEG_GROWTH_YEARS yr]) / |eps_ttm[t - ...]|
#                     / PEG_GROWTH_YEARS                       # percent PER YEAR
#      PEG[t]     = PE[t] / g[t]
#
#  and the Stage-1 criterion `1/PEG - 1 > 0` (i.e. 0 < PEG < 1) is UNCHANGED -- no threshold,
#  weight or tier moves.
#
#  THE HORIZON, AND WHY ONE YEAR (this is the real decision, so the reasoning is here and not
#  in a report).  Measured over the 61,832 newest-8 rows of the panel, overall criterion pass
#  rate, WITH the sign-crossing nerf in place: vendor 0.2050 -> 1y 0.2149 -> 2y 0.1754 ->
#  3y 0.1491.  (Without the nerf the same three read 0.2669 / 0.2410 / 0.2147 -- the whole
#  difference is the crossing cell, which is what the nerf is for.)
#    * TTM-vs-TTM removes SEASONALITY outright, which QoQ cannot, and averages four quarters on
#      EACH leg, so it is ~2x quieter than even a single-quarter YoY.
#    * It is the quantity PEG is DEFINED over: a growth rate PER YEAR against an annual P/E.
#    * IT KEEPS THE MOST ROWS MEASURABLE.  The in-domain cell holds 34,574 rows at 1y against
#      31,434 at 3y: a longer horizon silently shrinks the measurable universe by ~9% of rows
#      and biases it toward long-history names.
#    * IT STAYS RESPONSIVE, and that is not a side benefit -- it is the second prize of this
#      whole change.  A turnaround has to be visible while the 36-month hold is still ahead; a
#      3-year growth rate averages a fresh recovery back into three years of losses.
#    * AND THE NERF STRENGTHENS THE CHOICE rather than being orthogonal to it.  Crossing rows
#      are substituted with the POOL MEDIAN growth, and that median FALLS with the horizon
#      (6.72%/yr at 1y, 5.14% at 2y, 4.23% at 3y on this panel), so a longer horizon hands
#      every crossing row a harsher bar for a reason that is a property of the WINDOW, not of
#      the company.  The crossing cell tracks it: 0.1722 -> 0.1284 -> 0.0975.
#  THE COUNTER-ARGUMENT, STATED BECAUSE IT IS REAL: for a 36-month hold a 3-year growth rate is
#  closer to the horizon of the bet and is quieter still.  It loses on two grounds.  (a) Stage-1
#  already averages this criterion over 8 rows (`calcByTier` returns the head(8) MEAN of a
#  pass/fail indicator), so lengthening the window buys much less variance reduction than it
#  appears to.  (b) A longer LOOKBACK is not a better estimate of the forward three years for a
#  company whose trajectory has just changed -- which is exactly the population the filter is
#  trying to find.
#
#  THE GROWTH DENOMINATOR IS |eps_ttm_prev|, AND THAT ONE CHOICE IS WHAT MAKES A TURNAROUND
#  EXPRESSIBLE.  It reduces to the ordinary growth rate whenever the base is positive, so there
#  is NO case split, and when the base is NEGATIVE it still returns a positive, finite,
#  monotone-in-improvement number.  Under the vendor's `eps_t/eps_{t-1} - 1` the same situation
#  produces a NEGATIVE growth rate and the company FAILS a criterion for having recovered.
PEG_GROWTH_YEARS = 1

#  EPS BASIS.  The vendor's PEG is built from `eps` (income available to COMMON);
#  `netIncomePerShare` is a near-perfect proxy but NOT identical -- measured sign agreement
#  92.8%, median absolute error 2.5%.  `eps` / `epsdiluted` are now captured at ingest
#  (createDicts.preReq_dict 'inc') but are ABSENT FROM EVERY SAVED PANEL and only populate on
#  the next full fetch, so the local computation reads the PROXY and says so.
#  SWITCHING TO `eps` MUST BE A DELIBERATE EDIT, never an `eps if present else proxy` fallback:
#  the first fetch that carried the column would silently change the basis of a scored
#  criterion with nothing in the run to say it had.  Pinned by
#  test_nan_policy.test_peg_eps_basis_is_pinned_and_cannot_switch_silently.
_PEG_EPS_FIELD = 'netIncomePerShare'

#  `epsTTM` IS DELIBERATELY NOT USED, and the reason is not laziness.  `stamp_frequency_and_graham`
#  exposes a trailing-year EPS on a DIFFERENT basis (TTM netIncome / current-row shares) and it
#  is ABSENT from every panel written before 2026-07-29, including the one every number above is
#  measured on.  Using it would make PEG unmeasurable offline AND would put a second definition
#  of trailing EPS into a criterion whose basis the comment above pins to one named field.  If
#  the pipeline should carry ONE canonical trailing EPS -- it probably should -- that is a
#  deliberate unification, not a side effect of this change.


#  THE SIGN-CROSSING NERF (CEO, 2026-08-05): "We can also just set the going from negative to
#  positive arbitrarily a higher number in the peg calc, so it is nerfed."  Governing principle,
#  carried over from the return-on-equity case: THE CRITERION MUST NOT TREAT AN UNASSESSABLE
#  STATE POSITIVELY.
#
#  WHY A SUBSTITUTION AND NOT A CAP.  A percentage growth rate computed from a NEGATIVE base is
#  not a growth rate -- it is an artifact of the base's SIGN.  (E_now - E_prev)/|E_prev| for
#  E_prev < 0 saturates near +100% for a marginal recovery and grows with |E_prev|, i.e. with how
#  bad the prior year was, so the measure rewards the depth of the previous loss.  Measured
#  consequence: the turnaround cell passed at 0.890 against 0.362 in the normal cell -- PEG < 1
#  becomes "P/E under ~100" there where a steady 10%/yr grower faces "P/E under 10".  An order of
#  magnitude more valuation headroom for having had a bad prior period.
#
#  SO THE CROSSING ROW TAKES THE POOL'S MEDIAN GROWTH RATE, and its P/E then has to stand on its
#  own.  The crossing confers NEITHER CREDIT NOR PENALTY, which is the ROE ruling's requirement,
#  and it introduces NO TUNED CONSTANT -- the same ground on which a relative floor was refused
#  further down this module.  A chosen "arbitrarily higher number" would have been the only tuned
#  constant on this path.
#
#  IT IS A CROSS-SECTIONAL BASELINE, SO IT LIVES WHERE THE OTHER ONE DOES.  The median is a
#  property of the POOL, and `calc_special` sees ONE SOURCE at a time -- on the production fetch
#  path the panel does not even exist yet when it runs.  So the crossing rows come out of the
#  build as NaN and are filled by `substitute_peg_crossing`, called from `postBo.postBoWrapper`
#  immediately before Stage-1 scoring: the same position, and for the same reason, as
#  `calcScore.getAves2`'s `BoMetric_ave` (audit H-1 -- recompute the cross-sectional baseline on
#  the frame you actually score, never carry a stale one, and never freeze it into the panel).
#  The SAVED panel therefore keeps the honest per-source pre-substitution column.
#
#  THE COST, STATED: the criterion's bar for a crossing row is now PANEL-DEPENDENT, where the rest
#  of PEG is absolute.  That is a real property, not a hidden one -- `substitute_peg_crossing`
#  returns the median it used and the run prints it, so a ranking can never be read without it.
PEG_CROSSING_SUBSTITUTION = 'pool_median_growth'

#  =========================================================================== #
#  THE POOL WINDOW -- WHY THE MEDIAN IS NOT TAKEN OVER THE WHOLE PANEL          #
#  (fetch-depth audit, 2026-08-14)                                              #
#  =========================================================================== #
#  `peg_pool_median_growth` used to pool growth rates over EVERY ROW OF EVERY SOURCE in the
#  panel, so the bar a crossing row faces was a function of HOW MANY QUARTERS WE FETCHED.
#  That is the defect `meanBars` (register C-12) removed from the Stage-1 `mean` family, in
#  the one place it survived, and the deep fetch is what makes it bite:
#
#    * MEASURED, on the 2026-08-13 CUR3K panel (baseline_tools/depth_sensitivity.py, arm A):
#      re-scoring the SAME panel deepened from 24 to 80 rows MOVED THE POOLED MEDIAN, and it
#      moved a SCORED criterion -- `PEG` is Tier C (w = 0.30), so one flipped row inside the
#      head(8) window is worth 0.0375 of BoScore.  7.0% of sources changed score through this
#      channel alone once the history bonus was held fixed, by up to 0.15 (four flipped rows).
#      It is the ONLY channel through which a windowed Stage-1 criterion moved with depth.
#    * AND IT IS A REGIME-MIXING CHANNEL INDEPENDENT OF THE DEPTH QUESTION.  At `-nrperiods 80`
#      the panel reaches back to 2006, so a 2026 crossing row would have been judged against a
#      median blended out of the GFC and the 2021 boom.  "The typical growth rate" is a
#      statement about NOW; a two-decade pool is not that statement.
#
#  THE WINDOW IS STAGE-1'S OWN SCORING WINDOW, AND THAT CHOICE IS THE ARGUMENT.  The criterion
#  is scored as `calcByTier`'s head(n) mean over the newest n rows, so the population the bar
#  should describe is the population the criterion is actually confronted with -- exactly the
#  reasoning `meanBars._newest_window` states for the mean-bar pass rates ("the SAME population
#  Stage-1 scores over").  Any longer window would put rows into the RULER that never reach the
#  SCORE.
#
#  IT IS `rp.scale_window`-SCALED, WHICH STAGE-1's OWN head(n) DELIBERATELY IS NOT, and the
#  divergence is intentional rather than an oversight.  Ruling Q2 (2026-07-26) leaves Stage-1's
#  window unscaled because there it counts BERNOULLI TRIALS behind one company's pass rate --
#  a property of that company, which halving n only makes noisier.  This median is a
#  CROSS-SECTIONAL quantity pooled ACROSS companies, so an unscaled window would put TWICE the
#  calendar span of the semi-annual cohort (~14% of the universe) into a bar the quarterly
#  cohort also faces.  That is precisely the defect CYCLEHEAT_BASE_NQ exists to prevent, and it
#  is why the two windows are scaled differently.
#
#  BIT-IDENTICAL AT NO DEPTH -- this CHANGES the shipped number, and that is the point: the old
#  number was a function of `-nrperiods`.  The run prints both the median and the row count it
#  was taken over (see `substitute_peg_crossing`), so the change is visible in the log.
#
#  THE ON-PANEL COST, MEASURED AT TODAY'S DEPTH (2026-08-13 CUR3K, 2,629 sources, nothing else
#  varied).  Pool median 6.4718% -> 7.4457% (over 10,150 in-domain rows instead of 26,471 --
#  a slightly STRICTER bar, because the recent window is growthier than the 2020-2026 pool):
#      44 sources (1.67%) move BoScore; max |delta| 0.075000, median 0.037500 (= 0.30/8, i.e.
#      exactly one PEG row flipping inside the head(8) window)
#      top-20  : membership IDENTICAL **and order identical**
#      top-100 : membership IDENTICAL, **ORDER CHANGED on 4 of 100 positions** -- a local
#                rotation at ranks 47-50 (DDI 50->47, 0HQU.L 47->48, CF 48->49, PEY.TO 49->50)
#  THE ORDER CHANGE IS STATED BECAUSE THE CEO READS THE TOP-100 AS RANKED.  "Membership
#  unchanged" is the weaker claim and quoting it alone would imply the list is untouched; it is
#  not -- four adjacent names rotate.  Nothing enters or leaves either list.
PEG_POOL_WINDOW_NQ = 8


def peg_criterion(peg):
    """`1/PEG - 1`, the scored Stage-1 quantity, from a PEG Series.  ONE expression, shared by the
    build and by the crossing substitution so the two cannot drift.

    `0` for a PEG of exactly 0 is kept verbatim from the pre-2026-08-05 line -- it then becomes
    -1, i.e. a FAIL, which is right (a PEG of exactly 0 is not "infinitely cheap").  A NaN PEG
    stays NaN rather than becoming -1: both read as a FAIL at Stage-1 (`Sign * NaN > 0` and
    `-1 > 0` are both False), so this is not a scoring difference -- it keeps "not computable"
    and "computed, and it failed" DISTINCT in the panel, which the Stage-1 NaN-accounting readout
    and the fill report both read.
    """
    out = pd.Series(np.where(peg.notna() & (peg != 0), 1.0 / peg, 0.0) - 1.0, index=peg.index)
    return out.where(peg.notna())


def peg_local(df, rpy=rp.DEFAULT_ROWS_PER_YEAR, years=None, crossing_growth=None):
    """PEG computed locally from full-precision inputs.  Returns (peg, eps_ttm_now, eps_ttm_prev).

    `df` is ONE source, NEWEST-FIRST (the `build_bometric_rows` contract), so a positive shift
    in `.shift(-k)` reaches OLDER rows -- the same convention `calc_diff` uses.

    THE DOMAIN IS APPLIED HERE, AND ONLY HERE.  NaN wherever the quantity does not exist: no
    trailing year, no prior trailing year, a zero prior base (the growth rate would be division
    by zero), or a NON-POSITIVE CURRENT TRAILING EPS.  Nothing is imputed and no sentinel is
    returned.

    THE DOMAIN MOVED WHEN THE COMPUTATION DID, and this is the half of the change a reviewer
    should press on.  The shipped guard (`peg_growth_defined`, da79aee) required
    `eps_t > 0` AND `eps_{t-1} > 0`, because under the vendor's RATIO-form growth leg a base
    crossing zero made the growth rate meaningless.  The local growth leg divides by |base|, so a
    NEGATIVE base is fine -- it is the recovery case, and expressing it is the point.  What
    remains inadmissible is a non-positive CURRENT trailing EPS: with no earnings there is no
    P/E, and that one condition removes BOTH of the old false-pass cells (where PE < 0 cancelled
    against growth < 0 into a positive PEG).

    NOTE THIS IS NOT A `Guard` ENTRY, and that is deliberate rather than an omission.  A guard is
    a predicate on the RAW frame, applied to a ratio the loop in `build_bometric_rows` has
    already computed; its signature carries no `rpy`, so a PEG guard would have to re-derive the
    filer's frequency from the stamp while this function receives it from the caller.  Two
    statements of one domain, resolved from two different places, is exactly the
    silently-divergent pair this repo keeps getting bitten by -- so there is one statement, and
    it is the one that has `rpy`.

    THE FOUR SIGN CELLS, MEASURED on the 61,832 newest-8 rows [panel = resdic_2026-07-17_
    CORRECTED], BEFORE (vendor field + the shipped two-sided guard, single-period eps legs) and
    AFTER (local, TTM legs, 1-year horizon, crossing rows on the POOL MEDIAN growth):

        cell                    BEFORE rows / passes        AFTER rows / passes
        eps_now>0 prev>0        34,500 / 12,673  (0.367)    34,569 / 12,513  (0.362)
        eps_now<=0 prev>0        5,177 /      0  refused     4,225 /      0  refused
        eps_now<=0 prev<=0      17,035 /      0  refused    17,321 /      0  refused
        eps_now>0 prev<=0        5,089 /      0  FAILED      4,489 /    773  (0.172)  <-- the fix

    (Row counts move between the two because the legs themselves change from single-period to
    trailing-year -- they are not the same partition of the same rows.)  Overall criterion pass
    rate 0.2050 -> 0.2149.  NOTE WHERE THAT COMES FROM: the normal cell is essentially unchanged
    (-0.53pp), so the fixed 0<PEG<1 bar has NOT been loosened by the horizon change; the whole
    movement is the crossing rows going from AUTO-FAILED to SCORED-ON-THEIR-OWN-P/E.

    READ THE TWO CELL RATES AGAINST EACH OTHER, because that is the property the nerf is judged
    on.  Crossing 0.172 against normal 0.362 = 0.48x.  BEFORE the nerf it was 0.890 against 0.362
    = 2.46x, i.e. roughly an order of magnitude more P/E headroom for having had a bad prior year.
    Comparable-but-LOWER is the expected shape and the mechanism is not a mystery: a company whose
    trailing-year EPS has only just crossed zero has a tiny denominator, so its trailing P/E is
    genuinely high, and a high P/E fails against a 6.72%/yr median growth rate.  So the honest
    summary of what this buys is "those 5,089 rows are now MEASURED instead of auto-failed", NOT
    "they now pass".

    THE TINY-POSITIVE-BASE ROWS ARE NOT REACHED, DELIBERATELY.  222 rows panel-wide land at
    |PEG| < 1e-3 with a prior base that is positive but tiny -- a real, enormous growth rate, not
    a sign artifact, so the crossing substitution does not and must not touch them.  (Panel-wide
    over every row; the head(8) SCORING-window figure is ~76, and the two reconcile at the ~35% of
    panel rows that window covers.)  The only ways to reach them are a relative floor on |base| or
    a cap on the growth rate, and BOTH are tuned constants -- this path deliberately carries none,
    which is the same ground on which the pool median was chosen over a picked number.  Recorded
    for the CEO as a threshold question, not silently absorbed.
    """
    years = PEG_GROWTH_YEARS if years is None else years
    e = pd.to_numeric(df[_PEG_EPS_FIELD], errors='coerce')
    price = pd.to_numeric(df['price'], errors='coerce')
    #  Newest-first, so the rolling sum runs on the REVERSED series: each row then carries
    #  itself plus the (rpy-1) OLDER rows -- the same idiom stamp_frequency_and_graham uses for
    #  its TTM netIncome, deliberately, so the two trailing-year windows are built the one way.
    eps_ttm = e.iloc[::-1].rolling(int(rpy)).sum().iloc[::-1]
    lag = int(rpy) * int(years)
    prev = eps_ttm.shift(-lag)
    g = 100.0 * (eps_ttm - prev) / prev.abs() / float(years)
    #  THE SIGN-CROSSING NERF.  Where the base is NON-POSITIVE the ratio above is not a growth
    #  rate (see PEG_CROSSING_SUBSTITUTION), so it is replaced by the POOL's median growth rate
    #  when one is supplied and made NaN when it is not.  NaN is the build-time answer: the median
    #  is cross-sectional and `calc_special` sees one source, so the crossing rows are filled later
    #  by `substitute_peg_crossing`.  Nothing is imputed here and no sentinel is used.
    crossing = (prev <= 0) & prev.notna() & eps_ttm.notna()
    if crossing.any():
        g = g.mask(crossing, float(crossing_growth) if crossing_growth is not None else np.nan)
    pe = price / eps_ttm
    peg = (pe / g).replace([np.inf, -np.inf], np.nan)
    #  THE DOMAIN, applied here and nowhere else.  `eps_ttm > 0` is "there IS a P/E";
    #  `prev.notna() & (prev != 0)` is "there is a prior trailing year, and it is not a zero
    #  base".  A NEGATIVE prior base is admissible -- that is the turnaround.  These comparisons
    #  are False on NaN, so an undetermined domain is inadmissible without a fillna.
    peg = peg.where((eps_ttm > 0) & prev.notna() & (prev != 0))
    return peg, eps_ttm, prev


#  The row key the substitution aligns on.  (source, date) ALONE IS NOT ENOUGH: 282 sources carry
#  DUPLICATE snapped quarters (1,639 of 5,501 rows), so a pair-merge fans out many-to-many and can
#  pair a row with the wrong counterpart -- the defect that made my own first measurement of the
#  Graham boundary report 70 changed rows when the true answer was 0.  Adding the OCCURRENCE index
#  within (source, date) makes the key unique and is strictly stronger than the (source, date)
#  matching `data_quality.apply_data_quality_filter` already relies on, which rests on the same
#  documented invariant: both frames are built per ticker from the SAME date vector and both go
#  through `utils.setDatesToQuarterly`.
_PEG_KEY = ('source', '_peg_date', '_peg_occ')


def _peg_row_key(df, date_col='date'):
    """The three key columns for `df` (a small COPY; `df` itself is untouched).

    THE DATE IS SNAPPED ON BOTH SIDES BEFORE KEYING, and that is load-bearing rather than
    defensive.  `build_bometric_rows` writes `utils.setDatesToQuarterly(tempMetric_df)`, so
    BoMetric_df carries SNAPPED period ends, while cdx_df carries snapped ones on a SAVED panel and
    RAW ones on the live fetch path.  Keyed on the raw date, the live path matches nothing --
    measured on a synthetic pair: 2 crossing rows found, 0 filled, and the only symptom was the
    `n_unmatched` counter, which is exactly why that counter is returned and printed rather than
    swallowed.  `setDatesToQuarterly` is idempotent on already-snapped dates, so applying it here
    is a no-op on the saved-panel path.
    """
    d = pd.to_datetime(df[date_col], errors='coerce')
    #  utils.setDatesToQuarterly is `PeriodIndex(freq='Q').to_timestamp()`; inlined rather than
    #  imported to keep calcMetrics free of a utils dependency (utils imports nothing from here,
    #  but the one-way edge is worth preserving).  NaT-safe: PeriodIndex carries NaT through.
    d = pd.Series(pd.PeriodIndex(d, freq='Q').to_timestamp(), index=d.index)
    return pd.DataFrame({'source': df['source'].values, '_peg_date': d.values,
                         '_peg_occ': d.groupby([df['source'].values, d.values]).cumcount().values},
                        index=df.index)


def peg_pool_median_growth(cdx_df, freq_map=None, years=None, window_nq=None):
    """The POOL's median annual growth rate, over the rows the PEG criterion can actually score.

    THE POPULATION IS THE IN-DOMAIN ROWS -- positive trailing EPS now AND a positive prior
    trailing year.  That is deliberately narrower than "every row where the arithmetic works":
    it is "the typical growth rate among the companies this criterion is able to assess", which
    is exactly the bar a crossing row should be made to face.  Rows whose base is non-positive
    are excluded BY CONSTRUCTION -- they are the rows being substituted, so including them would
    let the artifact define its own replacement.

    ...AND THEY ARE RESTRICTED TO EACH SOURCE'S NEWEST `window_nq` ROWS, rp.scale_window-scaled
    (`PEG_POOL_WINDOW_NQ`, 8 quarters = 2 years either frequency).  Without that restriction the
    bar is a function of `-nrperiods`, which is the whole reason the constant exists -- the full
    reasoning, the measurement and why this window is scaled where Stage-1's is not are at
    PEG_POOL_WINDOW_NQ.  `window_nq=0` restores the whole-panel pool; it exists for the
    depth-sensitivity experiment and for nothing else, and it is NOT a production option.

    Returns (median, n_rows).  `median` is NaN when the pool has no in-domain row, in which case
    `substitute_peg_crossing` leaves every crossing row NaN -- i.e. refused, which is the honest
    answer and is what the criterion did before this change.
    """
    if freq_map is None:
        freq_map = rp.frequency_by_source(cdx_df)
    window_nq = PEG_POOL_WINDOW_NQ if window_nq is None else int(window_nq)
    vals = []
    for src, g in cdx_df.groupby('source', sort=False):
        rpy = rp.rows_per_year(freq_map, src)
        tf = g.iloc[::-1] if _is_oldest_first(g) else g
        e = pd.to_numeric(tf[_PEG_EPS_FIELD], errors='coerce')
        ttm = e.iloc[::-1].rolling(int(rpy)).sum().iloc[::-1]
        lag = int(rpy) * int(PEG_GROWTH_YEARS if years is None else years)
        base = ttm.shift(-lag)
        gr = (100.0 * (ttm - base) / base.abs()
              / float(PEG_GROWTH_YEARS if years is None else years))
        gr = gr.replace([np.inf, -np.inf], np.nan).where((ttm > 0) & (base > 0))
        #  `tf` is NEWEST-FIRST, so head(w) is the RECENT window.  Taken on the ROW series
        #  BEFORE the dropna -- a dropna'd series would reach back past the window to find w
        #  computable rows, which is the same uncapped-window shape this restriction removes
        #  (`stage2_metrics.cycleheat` states the identical distinction for its own gate).
        if window_nq > 0:
            gr = gr.head(rp.scale_window(window_nq, rpy))
        vals.append(gr.dropna())
    if not vals:
        return float('nan'), 0
    allg = pd.concat(vals, ignore_index=True)
    return (float(allg.median()) if len(allg) else float('nan')), int(len(allg))


def _is_oldest_first(g, date_col='date'):
    """True when this source's frame is OLDEST-first.  `peg_local` requires NEWEST-first (its
    `.shift(-k)` reaches older rows), and the saved panel is stored oldest-first while the live
    build hands over the raw newest-first FMP order -- so the orientation is DETECTED rather than
    assumed.  A frame with unparseable or tied dates falls through as newest-first, i.e. the
    build-time convention."""
    d = pd.to_datetime(g[date_col], errors='coerce').dropna()
    return len(d) >= 2 and d.iloc[0] < d.iloc[-1]


def substitute_peg_crossing(bm_df, cdx_df, freq_map=None, verbose=True):
    """Fill `bm_df['PEG']`'s sign-crossing rows with the pool-median-growth criterion value.

    Returns (bm_df COPY, stats).  `bm_df` is never mutated: the caller replaces its own local, so
    the SAVED panel keeps the honest per-source pre-substitution column and the SCORED frame
    carries the cross-sectional one -- the `BoMetric_ave` pattern (audit H-1).

    Called from ONE place, `postBo.postBoWrapper`, immediately before Stage-1 scoring.  It is not
    called from the per-source builders because the median does not exist there.
    """
    stats = {'median_growth': float('nan'), 'n_pool_rows': 0, 'n_crossing_rows': 0,
             'n_filled': 0, 'n_unmatched': 0}
    if bm_df is None or 'PEG' not in getattr(bm_df, 'columns', []) or cdx_df is None:
        return bm_df, stats
    if freq_map is None:
        freq_map = rp.frequency_by_source(cdx_df)
    med, n_pool = peg_pool_median_growth(cdx_df, freq_map=freq_map)
    stats['median_growth'], stats['n_pool_rows'] = med, n_pool
    if not np.isfinite(med):
        if verbose:
            print('PEG CROSSING SUBSTITUTION: the pool has no in-domain row, so no median exists '
                  '-- every sign-crossing row stays REFUSED (the pre-2026-08-05 behaviour).',
                  flush=True)
        return bm_df, stats

    rows = []
    for src, g in cdx_df.groupby('source', sort=False):
        rpy = rp.rows_per_year(freq_map, src)
        tf = (g.iloc[::-1] if _is_oldest_first(g) else g)
        peg, ttm, base = peg_local(tf, rpy=rpy, crossing_growth=med)
        crossing = (base <= 0) & base.notna() & ttm.notna() & (ttm > 0)
        if not crossing.any():
            continue
        k = _peg_row_key(tf)
        sub = pd.DataFrame({'source': k['source'], '_peg_date': k['_peg_date'],
                            '_peg_occ': k['_peg_occ'],
                            '_peg_new': peg_criterion(peg).values})[crossing.values]
        rows.append(sub)
    stats['n_crossing_rows'] = int(sum(len(r) for r in rows))
    if not rows:
        return bm_df, stats

    fill = pd.concat(rows, ignore_index=True).dropna(subset=['_peg_new'])
    out = bm_df.copy()
    key = _peg_row_key(out)
    idx = pd.MultiIndex.from_arrays([key['source'], key['_peg_date'], key['_peg_occ']])
    m = pd.Series(fill['_peg_new'].values,
                  index=pd.MultiIndex.from_arrays([fill['source'], fill['_peg_date'],
                                                   fill['_peg_occ']]))
    #  Duplicate keys would make the reindex ambiguous.  They cannot occur -- the occurrence index
    #  is what removes them -- so assert rather than silently take the first.
    if m.index.has_duplicates:
        raise ValueError(
            'calcMetrics.substitute_peg_crossing: the (source, date, occurrence) key is not '
            'unique on the substitution frame, so the fill would be ambiguous. That key exists '
            'precisely to remove the duplicate-snapped-quarter ambiguity -- if it is duplicated, '
            'the two frames were not built from the same date vector.')
    new = m.reindex(idx)
    take = new.notna().to_numpy()
    stats['n_filled'] = int(take.sum())
    stats['n_unmatched'] = int(len(fill) - take.sum())
    vals = pd.to_numeric(out['PEG'], errors='coerce').to_numpy(dtype='float64')
    vals[take] = new.to_numpy(dtype='float64')[take]
    out['PEG'] = vals
    if verbose:
        print('PEG CROSSING SUBSTITUTION: pool median annual growth = %.4f%% over %d in-domain '
              'row(s) [pooled over each source\'s NEWEST %d quarter(s), rpy-scaled -- '
              'PEG_POOL_WINDOW_NQ, so the bar does NOT move with `-nrperiods`]; %d '
              'sign-crossing row(s) found, %d filled, %d unmatched.'
              % (med, n_pool, PEG_POOL_WINDOW_NQ, stats['n_crossing_rows'],
                 stats['n_filled'], stats['n_unmatched']), flush=True)
        if stats['n_unmatched']:
            print('  NOTE %d substitution row(s) had no counterpart in BoMetric_df -- expected '
                  'for the oldest `rpy` rows, which build_bometric_rows trims.'
                  % stats['n_unmatched'], flush=True)
    return out, stats


def apply_domain_guard(df, values, guard):
    """`values` with every row the named guard rejects replaced by NaN.

    `df` is the RAW statement frame the ratio was built from (positional index, newest-first);
    `values` is the ratio as `calc_simpleRatio` returns it -- a list, positionally aligned with
    `df`.  Returns a list, so the caller's downstream handling is unchanged.

    ORDER MATTERS AND IS THE CALLER'S RESPONSIBILITY: the guard must be applied to the LEVEL
    BEFORE `calc_diff` takes the difference.  Guarding after the diff would leave a diff
    computed ACROSS an out-of-domain row -- a change measured against a meaningless base.
    Guarding before makes the diff NaN whenever EITHER leg is inadmissible, which is the
    honest reading: there is no defined change in a quantity that was undefined.
    """
    if guard not in STAGE1_DOMAIN_GUARDS:
        raise KeyError(
            "calcMetrics.apply_domain_guard: no guard named %r (known: %r). A `Guard` key was "
            "added to a createDicts metric entry without a predicate here -- add it, or drop "
            "the key. Silently ignoring an unknown guard would leave the criterion scoring "
            "its perverse rows with nothing to say so." % (guard, sorted(STAGE1_DOMAIN_GUARDS)))
    admissible = STAGE1_DOMAIN_GUARDS[guard](df)
    v = pd.to_numeric(pd.Series(list(values)), errors='coerce').to_numpy(dtype='float64')
    ok = np.asarray(admissible.fillna(False), dtype=bool)
    if len(ok) != len(v):
        raise ValueError(
            "calcMetrics.apply_domain_guard: guard %r produced %d flags for %d values. The "
            "guard reads the raw frame and the values are positionally aligned to it, so a "
            "length mismatch means the caller passed a different frame than the ratio was "
            "built from." % (guard, len(ok), len(v)))
    return np.where(ok, v, np.nan).tolist()


# =========================================================================== #
#  STAGE-1 BOUNDARY IMPUTATION (nan-policy.md ADDENDUM A, 2026-08-05)          #
# =========================================================================== #
#  THE COMPANION OF A GUARD, AND THE OPPOSITE CASE.  A `Guard` refuses rows whose value is
#  PERVERSE (a sign inversion) -- the number exists and is wrong.  A `Boundary` fills rows whose
#  value is UNDEFINED BECAUSE AN INPUT IS ADVERSE -- the number does not exist, and the metric's
#  own limit at that input's domain edge is the honest stand-in.  The CEO's instruction: "For
#  metrics that are NaN because of adverse things I don't think we should punish them again for
#  it.  So just put it like earnings were close to 0."
#
#  Declared per metric in `createDicts` (`Boundary`), predicate + limit here, and the limit
#  VALUES with their derivations in `nan_policy.BOUNDARY_LIMIT` -- the same three-place split the
#  guards use, so a criterion's domain and its boundary sit beside the criterion.
#
#  {name: (mask_of_rows_where_the_boundary_applies, metric_key_whose_limit_to_use)}
STAGE1_BOUNDARY_IMPUTATIONS = {
    #  `uGrahamNumberToPrice`.  grahamNumber = sqrt(22.5 * EPS_ttm * BVPS) is undefined for
    #  EPS_ttm <= 0 or BVPS <= 0; as EPS_ttm -> 0+ the whole expression -> 0, so the criterion
    #  column grahamNumber/price -> 0.0.  `calcScore.calcByTier` then tests `metvec - 1 > 0`
    #  on a unity criterion, i.e. -1.0, i.e. a FAIL -- which is EXACTLY what it does with the
    #  NaN today.  BEHAVIOUR-IDENTICAL BY CONSTRUCTION on 23,212 of the 61,832 newest-8 rows,
    #  and that identity is the point: it makes the fail DERIVED ("there is no earnings-based
    #  valuation floor to compare this price against") instead of INCIDENTAL ("the number was
    #  missing"), so the criterion no longer depends on NaN-scores-as-a-fail to be right.
    #
    #  ONLY THE ADVERSE ROWS.  `graham_missing_inputs` (208 rows, 0.9%) is a genuine provider
    #  gap and is left NaN -- imputing a real value there would put an answer where there is
    #  none.  The discriminator is the per-row reason the ingest already stamps.
    'graham_adverse': (lambda df: npol.graham_adverse_mask(df), 'uGrahamNumberToPrice'),
}


def apply_boundary_imputation(df, values, boundary, admissible=None):
    """`values` with every row the named boundary applies to replaced by the metric's LIMIT.

    Only rows that are BOTH in the boundary's mask AND currently NaN are filled: a row that
    already has a value is a measurement and is never overwritten.  Returns a list, so the
    caller's downstream handling is unchanged (mirrors `apply_domain_guard`).

    `admissible` -- the GUARD's mask, when a guard also ran on this form.  ENFORCEMENT, not
    decoration (review finding, 2026-08-05): `build_bometric_rows` applies the guard first and
    then the boundary, and a comment there claimed a boundary could therefore never resurrect a
    guard-refused row.  IT COULD.  Both mechanisms express refusal as NaN, so with a guard and a
    boundary on the same form the boundary's "fill the NaNs" step would fill the rows the guard
    had just refused -- and the ordering the comment relied on is what makes that happen, not what
    prevents it.  NOT LIVE TODAY (one Boundary, six Guards, no criterion declares both), which is
    exactly why it is worth closing now rather than after the first criterion that does.
    """
    if boundary not in STAGE1_BOUNDARY_IMPUTATIONS:
        raise KeyError(
            "calcMetrics.apply_boundary_imputation: no boundary named %r (known: %r). A "
            "`Boundary` key was added to a createDicts metric entry without a predicate here "
            "-- add it, or drop the key. Silently ignoring it would leave the criterion "
            "failing adverse rows for the wrong reason with nothing to say so."
            % (boundary, sorted(STAGE1_BOUNDARY_IMPUTATIONS)))
    mask_fn, metric_key = STAGE1_BOUNDARY_IMPUTATIONS[boundary]
    limit = npol.BOUNDARY_LIMIT.get(metric_key)
    if limit is None:
        raise KeyError(
            "calcMetrics.apply_boundary_imputation: boundary %r names metric %r, which has no "
            "entry in nan_policy.BOUNDARY_LIMIT. The LIMIT and its derivation must be written "
            "there -- ADDENDUM A's rule is not automatable, so a boundary without a derived "
            "limit is a tuned constant in disguise." % (boundary, metric_key))
    v = pd.to_numeric(pd.Series(list(values)), errors='coerce').to_numpy(dtype='float64')
    ok = np.asarray(mask_fn(df).fillna(False), dtype=bool)
    if len(ok) != len(v):
        raise ValueError(
            "calcMetrics.apply_boundary_imputation: boundary %r produced %d flags for %d "
            "values -- the caller passed a different frame than the ratio was built from."
            % (boundary, len(ok), len(v)))
    if admissible is not None:
        adm = np.asarray(pd.Series(list(admissible)).fillna(False), dtype=bool)
        if len(adm) != len(v):
            raise ValueError(
                "calcMetrics.apply_boundary_imputation: the guard mask has %d flags for %d "
                "values." % (len(adm), len(v)))
        #  A GUARD-REFUSED ROW IS OUT OF DOMAIN AND STAYS OUT.  The boundary answers "the metric
        #  is undefined because an input was adverse"; the guard answers "this row must not be
        #  scored at all".  The second is the stronger statement and wins.
        ok = ok & adm
    return np.where(ok & np.isnan(v), float(limit[0]), v).tolist()


# =========================================================================== #
#  netDebtToEBITDA -- THE THREE-BRANCH LEVERAGE RULE  (CEO, 2026-08-05)         #
# =========================================================================== #
#  WHAT CHANGED.  Until now this criterion was a UNITY test on FMP's `netDebtToEBITDA` with a
#  `Guard: ebitda_positive`, i.e. TWO branches: EBITDA > 0 -> test the ratio < 1; otherwise
#  refuse (NaN -> fail).  That refused 9,200 head(8) rows -- NET CASH with non-positive EBITDA
#  -- which the CEO has now ruled must PASS: net cash means there is no leverage problem
#  whatever earnings do.  The four measured cells over the 61,481 head(8) rows of the
#  2026-07-17 CORRECTED panel and the required behaviour of each:
#
#      netDebt > 0, EBITDA > 0   33,615 rows   test netDebt/EBITDA < the BAR (1.0x when
#                                                     this table was measured; 3.0x since
#                                                     2026-08-10 -- see NET_DEBT_TO_EBITDA_BAR.
#                                                     The CELL is unchanged, the LEVEL moved.)
#      netDebt < 0, EBITDA > 0   11,844 rows   PASS  (already passed via a negative ratio)
#      netDebt > 0, EBITDA <= 0   6,324 rows   FAIL  (debt, no earnings -- correct today, kept
#                                                     as a REFUSAL so the NaN accounting still
#                                                     records it as non-computable)
#      netDebt < 0, EBITDA <= 0   9,200 rows   PASS  <- THE FIX
#
#  WHY THIS IS A `special` FORM AND NOT A GUARD OR A BOUNDARY.  Neither existing mechanism can
#  express a three-branch rule, and forcing one would be worse than adding a formula:
#    * A `Guard` is a REFUSAL MASK.  It can only shrink a domain, so it cannot ADMIT the
#      net-cash cell -- and relaxing the guard to let the ratio through is the one thing that
#      must not happen (see the next paragraph).
#    * A `Boundary` FILLS an undefined row with the metric's ANALYTIC LIMIT, and the project's
#      own rule admits it only where that limit is FINITE and is not the metric's BEST value.
#      Here netDebt/EBITDA -> -infinity as EBITDA -> 0+ with netDebt < 0, which is both
#      infinite and the best side, so the boundary rule's own escape clause says REFUSE.  Using
#      it anyway would put a tuned sentinel where a derived limit is required.
#    * `special` is where this repo already keeps criteria that are FORMULAS rather than
#      Upper/Lower ratios (`CFOlessEarnings`, `PEG`, `returnOnEquity`,
#      `capitalExpenditureCoverageRatio`), and PEG's note states the principle directly: a
#      domain that is INTRINSIC to the rule belongs inside the rule, stated once.
#
#  THE NET-CASH BRANCH NEVER COMPUTES THE RATIO, AND THAT IS THE WHOLE POINT.  negative divided
#  by negative is a POSITIVE ratio of ARBITRARY magnitude -- netDebt -100 / EBITDA -200 = 0.5,
#  which would clear a `< 1` bar on apparent merit.  That is exactly the defect class the eight
#  sign-inversion fixes closed, and re-opening it here would be a regression dressed as a
#  feature.  So branch 1 tests the SIGN OF THE OPERAND and returns a verdict; the ratio's
#  magnitude enters ONLY in branch 2, where both operands are admissible.
#
#  HOW sign(netDebt) IS OBTAINED, AND ITS ONE LIMITATION.  `netDebt` is NOT a fetched field
#  today, so the operand's sign is recovered the same way the measurement that produced the
#  table above recovered it:  sign(netDebt) = sign(ratio) x sign(EBITDA proxy),  with the proxy
#  = operatingIncome + depreciationAndAmortization (FMP publishes `netDebtToEBITDA` but not the
#  EBITDA behind it).  Consequences, stated rather than buried:
#    * Where the proxy is ZERO or NaN the operand's sign is NOT RECOVERABLE, so the row is
#      REFUSED (~229 zero-proxy head(8) rows, 0.37%).  Refusing never rewards, and it keeps the
#      expected post-change NaN rate at ~10.8% rather than manufacturing passes.
#    * ratio == 0 exactly means netDebt == 0 -- neither net debt nor net cash.  With EBITDA > 0
#      branch 2 scores it as a PASS (zero leverage); with EBITDA <= 0 it falls to branch 3 and
#      is refused.  A genuine zero net-debt position with no earnings is arguably a pass, but it
#      is not recoverable from a zero ratio without the operand itself.
#    * `totalDebt` and `cashAndCashEquivalents` are being CAPTURED for the next fetch (see
#      createDicts.preReq_dict) precisely so this recovery can be replaced by the real operand
#      `netDebt = totalDebt - cash`.  DO NOT rewire this function to them until a panel that
#      actually carries them exists -- they are absent from every saved pickle.
#
#  THE FLOW LEG IS THIS FUNCTION'S RESPONSIBILITY NOW.  `build_bometric_rows` applies
#  `rp.stage1_flow_factor` only inside its ratio loop, and a `special` never enters that loop.
#  `netDebtToEBITDA` is ('flow_den', 'annualize') -- a net-debt STOCK over ONE PERIOD's EBITDA
#  -- so the factor (0.25 quarterly, 0.5 semi-annual) is what makes the bar an ANNUAL one.
#  Dropping it here would have raised the bar 4x for every name.
#
#  THE EMITTED COLUMN IS A VERDICT-BEARING QUANTITY, NOT A LEVERAGE RATIO.  Sign +1, scored as
#  `value > 0`:
#      +1.0        branch 1 admission (net cash).  A SENTINEL -- the magnitude means nothing,
#                  and it is deliberately NOT re-based on the bar: nothing downstream reads the
#                  magnitude, and moving a sentinel with a threshold invites someone to.
#      bar - r     branch 2, the headroom below the bar; > 0 iff r < bar.  Informative.
#                  `bar` is NET_DEBT_TO_EBITDA_BAR = 3.0x since 2026-08-10 (was 1.0x).
#      NaN         branch 3 / unrecoverable, refused -> scores as a fail, as today.
#  Nothing downstream reads this column except `calcScore.calcByTier`'s sign test and the NaN
#  accounting, so a mixed sentinel/margin column is safe -- but do NOT start reading it as a
#  ratio.
#  --- THE BAR: 3.0x ANNUALISED NET DEBT / EBITDA  (CEO, 2026-08-10) -----------------------
#  RAISED FROM 1.0x.  The CEO chose this over the alternative on the table (relabelling the
#  flag as something other than a solvency test), so the flag KEEPS ITS SOLVENCY LABEL and the
#  level is what moves.
#
#  1.0x WAS NOT A SOLVENCY BAR, IT WAS A NEAR-DEBT-FREE BAR.  On the 2026-08-10 CUR3K panel it
#  ejected 549 of the 1,388 general names -- 80% of them on FULLY-POPULATED numbers at a median
#  3.02x -- so the modal ejected name was an ordinary company with ordinary leverage, not a
#  distressed one.  A veto flag whose modal catch is "normal" is measuring the wrong thing.
#
#  3.0x IS WHAT "SERVICEABLE LEVERAGE" MEANS IN CREDIT PRACTICE: leveraged-loan and
#  investment-grade maintenance covenants sit at ~3.0-3.5x net debt / EBITDA, and distress is
#  conventionally read at 5-6x.  So 3.0 is the CONSERVATIVE end of the covenant band -- the
#  flag still fails a name the credit market itself would call over-levered, and no longer
#  fails one it would call ordinary.  It is a bar with a stated referent, which is more than
#  1.0x ever had.
#
#  IT IS A `special` COLUMN, SO IT MOVES TWO THINGS AT ONCE -- SAY SO RATHER THAN DISCOVER IT.
#  `netDebtToEBITDA` is BOTH a `stage1_veto` flag AND a Stage-1 SCORING criterion (Tier A,
#  w = 0.75, `createDicts.BoMetric_special_dict`), and both read the SAME emitted column
#  through the same `value > 0` test.  Loosening the bar therefore (a) un-ejects names from the
#  veto and (b) raises the Stage-1 score of every name whose annualised ratio lies in
#  [1.0, 3.0), which changes WHICH names reach the top-100 at all.  That coupling is inherent
#  to a `special` and is not a defect -- but a reader who expects only (a) will mis-read the
#  churn, so it is stated here.
#
#  WHAT MUST NOT MOVE: THE `EBITDA <= 0` AND NET-DEBT BRANCH.  Branch 3 refuses a row that has
#  net DEBT and no EBITDA, and `stage1_veto.FIELD_EVIDENCE['netDebtToEBITDA'] = 'counts'` turns
#  that refusal into a NON-PASS.  It is untouched by construction: the bar appears ONLY in
#  branch 2's margin, and branch 2 is gated on `ebitda > 0`.  Do not "generalise" the bar into
#  branch 1 or branch 3.
#
#  ITS VALUE, MEASURED RATHER THAN ASSERTED (corrected 2026-08-10, reviewer).  An earlier
#  version of this note called the branch "the flag's most valuable catch" that "exists
#  nowhere else in the veto set (11 names on this panel)".  The 11 is right and the rest was
#  not.  MEASURED on the 2026-08-10 CUR3K general pool at the 3.0x bar: 11 sources have a
#  WHOLLY refused `netDebtToEBITDA` window (net debt, no EBITDA) -- and **0 of the 11 are
#  ejected by this flag alone**; every one of them also fails another flag.  So the branch's
#  MARGINAL contribution to the ejection set is currently ZERO, and the honest reason to keep
#  it is that it is the only flag that ASKS the question, not that it is catching names
#  nothing else catches.
#  SEPARATELY, and not to be confused with the above: across the WHOLE flag the bar move
#  un-ejects 350 names (549 -> 199 failures), of which **251 were ejected by this flag ALONE**
#  (unique catches 334 -> 83).  That is the real cost of the level change, it is much larger
#  than the branch figure, and the two must never be quoted interchangeably.
NET_DEBT_TO_EBITDA_BAR = 3.0


def net_debt_three_branch(df, rpy=rp.DEFAULT_ROWS_PER_YEAR,
                          bar=None):
    """The three-branch leverage verdict for one source's frame (newest-first).

    Returns a float Series positionally aligned to `df`: > 0 pass, <= 0 fail, NaN refused.

    `bar` -- the annualised net-debt/EBITDA level branch 2 tests against, defaulting to
    `NET_DEBT_TO_EBITDA_BAR` (3.0x since 2026-08-10).  A parameter ONLY so an offline A/B can
    measure a level without mutating module state, exactly as `stage1_veto.apply_veto` takes
    `enabled=`/`pools=`; the production path passes nothing.
    """
    bar = NET_DEBT_TO_EBITDA_BAR if bar is None else float(bar)
    r = pd.to_numeric(df['netDebtToEBITDA'], errors='coerce')
    ebitda = (pd.to_numeric(df['operatingIncome'], errors='coerce')
              + pd.to_numeric(df['depreciationAndAmortization'], errors='coerce'))

    #  BRANCH 1 -- NET CASH.  An OPERAND condition: sign(netDebt) < 0, recovered as
    #  sign(ratio) x sign(EBITDA).  np.sign is 0 for an exact zero and NaN for NaN, so a
    #  zero/NaN proxy or a zero/NaN ratio yields a product that is NOT < 0 and the row falls
    #  through -- which is the intended refusal, not an oversight.
    net_cash = (np.sign(r) * np.sign(ebitda)) < 0

    #  BRANCH 2 -- the ordinary debt-service test, on the ANNUALISED ratio.
    factor = rp.stage1_flow_factor('netDebtToEBITDA', rpy)
    margin = bar - (r * factor)

    #  BRANCH 3 is the fall-through: neither net cash nor admissible EBITDA -> NaN.
    out = pd.Series(np.nan, index=df.index, dtype='float64')
    out = out.mask(ebitda > 0, margin)
    out = out.mask(net_cash, 1.0)
    return out


#maybe check if denom is 0?
def calc_simpleRatio(df,strUp,strDn):
#    res = pd.DataFrame()
#    res[resultString] = df[strUp] / df[strDn]
    if strDn == 'Identity':
        tmpres = df[strUp]
    else:
        tmpres = df[strUp]/df[strDn]

    return tmpres.tolist()
#    return res

def calc_compRatio(df,strUp,strDn,metstr,n,rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """DEAD CODE -- NO CALLER, AND IT WOULD RAISE IF THERE WERE ONE (flagged 2026-08-02).

    Repo-wide there is no call site: Stage-1 builds its rolling means via
    calc_simpleRatio + calc_diff (getData_fmp.build_bometric_rows).  And the final line is
    `res.tolist()` on a DataFrame, which has no `.tolist()` -- so the first caller would get
    an AttributeError, not a metric.  NOT removed here: deletion is outside the mandate of the
    pass that found it (behaviour-preserving refactor), and it is harmless while unreachable.
    Do not adopt it without fixing the return.
    """
    res = pd.DataFrame()
    if strDn == 'Identity':
        res[metstr] = df[strUp]
    else:
        res[metstr] = df[strUp] / df[strDn]

    res = res.iloc[::-1]
    res[metstr] = res[metstr].rolling(rp.scale_window(n, rpy)).mean()
    res = res.iloc[::-1]

    return res.tolist()

def calc_diff(df,metstr,n,rpy=rp.DEFAULT_ROWS_PER_YEAR):
    res = pd.DataFrame()

    dstr = "d" + metstr[0].upper() + metstr[1:]
    # shift(-1) is ONE REPORTING PERIOD, already frequency-relative -- it needs no
    # rescaling.  Only the smoothing window does, so the average covers the same
    # calendar span for a semi-annual filer as for a quarterly one.
    res[dstr] = df[metstr] - df[metstr].shift(-1)
    res = res.iloc[::-1]
    res[dstr] = res[dstr].rolling(rp.scale_window(n, rpy)).mean()
    res = res.iloc[::-1]

    return res

#  Stage-1 keys this function knows how to compute.  It MUST stay in step with
#  createDicts.getDicts()'s BoMetric_special_dict, which is the dict the only caller
#  (getData_fmp.build_bometric_rows) iterates.
#
#  WHY IT IS ENFORCED RATHER THAN TRUSTED (2026-08-02).  Before this, an unrecognised
#  `metstr` fell through every branch and returned an EMPTY DataFrame, which the caller
#  assigned straight into `tempMetric_df[key1]` -- so a metric added to the dict without a
#  branch here became an all-NaN column that then scored as pool-neutral, silently.  That is
#  the same silent-default shape as an unregistered Stage-2 metric, and it is closed the same
#  way: fail loudly, naming the key.
_SPECIAL_KEYS = ('CFOlessEarnings', 'PEG', 'returnOnEquity',
                 'capitalExpenditureCoverageRatio', 'netDebtToEBITDA')


def calc_special(df,metstr,n,rpy=rp.DEFAULT_ROWS_PER_YEAR,guard=None):
    """`guard`: optional STAGE1_DOMAIN_GUARDS name, applied to the computed column.

    The special criteria are FORMULAS rather than Upper/Lower ratio specs, so they never pass
    through build_bometric_rows's ratio loop where the declared `Guard` is applied.  The caller
    forwards it here instead, and the guard is applied to the FINISHED column -- which is the
    same point in the computation (these formulas have no diff stage of their own)."""
    if metstr not in _SPECIAL_KEYS:
        raise KeyError(
            "calcMetrics.calc_special has no formula for %r (known: %r). It used to return an "
            "EMPTY frame for an unknown key, which the caller wrote into BoMetric_df as an "
            "all-NaN column that scored as pool-neutral -- add the branch, or remove the key "
            "from createDicts.BoMetric_special_dict." % (metstr, list(_SPECIAL_KEYS)))
    res = pd.DataFrame()
    #if str == 'dInvPEG':
    #    Fix. Needs to be Earnings per share. And needs to be higher than unity (annualized)
    #    temp = pd.DataFrame()
    #    temp['ep'] = df['netIncome']/df['price']
    #    temp['dep'] = temp['ep'] - temp['ep'].shift(-1)
    #    temp['de'] = df['netIncome'] - df['netIncome'].shift(-1)
    #    res[str] = temp['dep']*temp['de']
    #    res = res.iloc[::-1]
    #    res[str] = res[str].rolling(n).mean()
    #    res = res.iloc[::-1]
    if metstr == 'CFOlessEarnings':
        # Sign-SAFE accrual test: CFO - netIncome (a DIFFERENCE, so no sign-flipping
        # denominator).  Replaces the uIncomeQuality = CFO/NI > 1 unity test -- see
        # createDicts.BoMetric_special_dict for the measured inversion it fixes.
        # No window/rpy scaling: both legs are the SAME period's flow, so the ratio of
        # their difference to zero is frequency-invariant (a sign test on a difference of
        # two same-period flows). Sign +1, tested as `metvec > 0`.
        res[metstr] = (pd.to_numeric(df['netCashProvidedByOperatingActivities'],
                                     errors='coerce')
                       - pd.to_numeric(df['netIncome'], errors='coerce'))
    elif metstr == 'PEG':
        #  COMPUTED LOCALLY (2026-08-05).  `df['priceEarningsToGrowthRatio']` -- the vendor
        #  field -- is no longer read: see `peg_local` for the vendor formula that was
        #  reverse-engineered, the three defects in it, and why the horizon is one YEAR.  The
        #  criterion form `1/PEG - 1 > 0` (i.e. 0 < PEG < 1) is UNCHANGED, as is Tier C /
        #  Sign +1; only the quantity inside it is now ours.
        #  `0` for a zero PEG is kept verbatim from the previous line -- it then becomes -1,
        #  i.e. a FAIL, which is right (a PEG of exactly 0 is not "infinitely cheap").
        #  crossing_growth is NOT supplied here, by design: the sign-crossing rows need the POOL's
        #  median growth rate, which does not exist at build time (this function sees ONE source,
        #  and on the fetch path the panel is still being accumulated).  They come out NaN and are
        #  filled by `substitute_peg_crossing` from `postBo.postBoWrapper`.  See
        #  PEG_CROSSING_SUBSTITUTION.
        _peg, _e_now, _e_prev = peg_local(df, rpy=rpy)
        res[metstr] = peg_criterion(_peg).values
    #elif str == 'CFOlessEarnings':
    #    res[metstr] = df['netCashProvidedByOperatingActivities'] - df['netIncome']
    #    res = res.iloc[::-1]
    #    res[metstr] = res[metstr].rolling(n).mean()
    #    res = res.iloc[::-1]
    elif metstr == 'returnOnEquity':
        #res[metstr] = df['netIncome']/df['totalStockholdersEquity'] - 0.12/4
        # 0.12 is an ANNUAL 12% ROE hurdle spread over the periods in a year, so the
        # divisor is rows-per-year -- NOT a fixed 4.  A semi-annual filer's row covers
        # six months and was being compared against a 3-month hurdle, i.e. a bar half
        # as high as intended.  (Row-based site NOT on the audit's list -- found in the
        # 2026-07-25 sweep.)
        #
        # THE `equity_positive` GUARD (declared in createDicts.BoMetric_special_dict) is what
        # stops a NEGATIVE-equity, NEGATIVE-income company clearing this 12% hurdle on a
        # positive ROE built from two negatives.  It is applied below, from the declaration --
        # NOT hard-coded here -- so the domain condition sits beside the metric.
        res[metstr] = df['returnOnEquity'] - 0.12/float(rpy)
    # An 'EPStoEPSmean' branch used to sit here and was UNREACHABLE: 'EPStoEPSmean' is a
    # STAGE-2 metric key (createDicts postNewRankingDict, computed by
    # stage2_metrics.eps_to_eps_mean) and has never been in BoMetric_special_dict, which is
    # the only dict this function is called with. Removed 2026-08-02 -- verified against
    # getDicts(): the special dict holds exactly _SPECIAL_KEYS. It also computed a DIFFERENT
    # quantity from the Stage-2 metric of the same name (a raw EPS-minus-mean level, not the
    # dimensionless deviation), so leaving it in place invited exactly the Stage-1/Stage-2
    # same-name-different-basis confusion that the accruals divergence came from. Nothing
    # referenced it, so no stored artifact changes.
    elif metstr == 'netDebtToEBITDA':
        #  THE THREE-BRANCH LEVERAGE RULE (CEO, 2026-08-05) -- see `net_debt_three_branch`
        #  above for the four measured cells, why this is a `special` rather than a `Guard`,
        #  and why the net-cash branch must never compute the ratio.  It carries NO `Guard`
        #  key: the old `ebitda_positive` guard IS branch 2's condition, now stated once
        #  inside the rule (same reasoning as PEG's domain).  The flow factor is applied
        #  inside the rule, because a `special` never passes through the ratio loop where
        #  `build_bometric_rows` applies it.
        res[metstr] = net_debt_three_branch(df, rpy=rpy).values
    elif metstr == 'capitalExpenditureCoverageRatio':
        tempce2cr = df[metstr]
        ce2cr = -tempce2cr.fillna(0)
        #  THE BAR IS `CFO > |capex|`, NOT `CFO > 2 x |capex|` (CEO, 2026-08-05).
        #  DERIVED, WHICH IS THE POINT OF THE CHANGE.  `capitalExpenditureCoverageRatio` is
        #  CFO / capex, and capex is reported NEGATIVE, so `-ratio` is CFO / |capex|.  The
        #  criterion `-ratio - 1 > 0` is therefore exactly `CFO > |capex|` -- which IS the
        #  definition of SELF-FUNDING capital expenditure: the business pays for its own
        #  investment out of operating cash rather than out of financing.  So the bar is a
        #  definition, not a tuning parameter, and it needs no provenance beyond itself.
        #  THE OLD `- 2` (CFO > 2x capex) HAD NO RECORDED SOURCE anywhere in the repo or the
        #  design folder -- it demanded twice-covered capex, a materially stricter and
        #  UNDERIVED bar, and that missing provenance is the whole reason it moved.
        #  UNCHANGED and deliberately so: the `fillna(0)` above, which turns a missing ratio
        #  into 0 -> -1 -> a FAIL (defect D10, a separate open item -- do not fold a second
        #  change into this one).
        res[metstr] = ce2cr - 1

    if guard is not None:
        res[metstr] = apply_domain_guard(df, res[metstr].tolist(), guard)

    return res


# =========================================================================== #
#  THE VETO COLUMNS -- COMPUTED, CARRIED, NEVER SCORED  (CEO, 2026-08-07)      #
# =========================================================================== #
#  These four exist for `stage1_veto.POOL_FLAGS` and for nothing else.  They are NOT Stage-1
#  criteria: they are declared in `createDicts.BoMetric_veto_dict`, which carries NO `Tier` and
#  NO `Sign`, and `calcScore.simpleScore_fromDict` iterates the FIVE SCORING dicts by name and
#  never sees it.  `createDicts` asserts the two key sets are DISJOINT at import.
#
#  WHY A SEPARATE CHANNEL AT ALL, since a column is a column.  Putting them in
#  `BoMetric_special_dict` -- the obvious home, since they are formulas rather than Upper/Lower
#  ratios -- would have added FOUR WEIGHTED CRITERIA TO EVERY POOL'S STAGE-1 SCORE, general
#  included, because every entry in that dict carries a Tier and a Sign and `calcByTier` scores
#  it.  That is a much larger change than a veto and nobody ruled for it.  The separation is
#  what makes "a veto column cannot become a scoring criterion by accident" a structural
#  property rather than a convention.
#
#  EACH COLUMN CARRIES THE QUANTITY WHOSE SIGN (or unity comparison) IS THE VERDICT, not a
#  boolean.  `producerEbitdaPositive` and `equityPositive` are named for the TEST the veto
#  applies (`> 0`), and hold the level it is applied to -- the same shape as `uCurrentRatio`
#  holding the ratio for a `> 1` bar.  A boolean column would put NaN and bool in one column,
#  make the dtype `object`, and hide a refusal behind a falsy value.
#
#  SIGN-SAFETY.  Exactly ONE of the four is a ratio (`reitEbitdaInterestCoverage`), and its
#  denominator is restricted to `> 0` by the `interest_expense_positive` guard DECLARED in
#  createDicts and applied here -- so it cannot invert the way the eight criteria fixed in the
#  2026-08-04/05 sign passes did.  The other three are levels or a sum: there is no denominator
#  to change sign.  A row the guard refuses arrives at `stage1_veto` as NaN, so the ADMISSIBILITY
#  DECISION LIVES IN THE COLUMN and no condition in `stage1_veto` can invert it.
#
#  NOT ON ANY EXISTING PANEL -- BUT REBUILDABLE ON ONE (corrected 2026-08-09).  These four
#  DERIVED columns are absent from every saved `BoMetric_df`, so `stage1_veto.apply_veto`
#  degrades the affected POOL to `applies=False` with `missing_columns` set rather than raising
#  (see `_STALE_PANEL_NOT_APPLICABLE`), and that is still exactly what a LIVE run on a stale
#  panel does.
#  WHAT WAS WRONG was the sentence that used to end this block: "NOTHING HERE IS BACKTESTABLE".
#  The RAW INPUTS are a different question from the derived columns, and all six of them --
#  `ebitda`, `cashAndCashEquivalents`, `netCashProvidedByOperatingActivities`,
#  `totalStockholdersEquity`, `revenue`, `interestExpense` -- are present on the 2026-08-07
#  CUR3K panel's `cdx_df` (verified 2026-08-09).  So an offline caller can rebuild these columns
#  with THIS function and evaluate them; it was done, and the numbers are recorded at
#  `stage1_veto.POOL_FLAGS['Mining']`.  A missing derived column is a REBUILD, not a re-fetch.
#  Backtestable is not the same as shipped: the live path reads the columns off the panel and
#  does not rebuild them, so the `applies=False` degradation above is unchanged.
_VETO_KEYS = ('reitEbitdaInterestCoverage', 'producerEbitdaPositive',
              'cashRunwayOneYear', 'equityPositive')

#  THE RAW STATEMENT LINES EACH VETO COLUMN NEEDS, so a caller can ask BEFORE computing.
#  `ebitda` and `cashAndCashEquivalents` are CAPTURE-ONLY additions from 2026-08-05, so the
#  OFFLINE rebuild paths (baseline_tools/panel_upgrade, dead_merge) can be handed a saved
#  `cdx_df` that predates them -- and the live fetch path can too, if preReq_dict is ever pared.
#
#  THE ABSENT-INPUT ANSWER IS TO OMIT THE COLUMN, NOT TO EMIT AN ALL-NaN ONE, and the difference
#  is the whole reason this registry exists.  An all-NaN column is PRESENT, so
#  `stage1_veto.missing_columns` finds nothing missing, the pool reports `applies = True`, every
#  flag abstains for want of evidence and the cohort comes back with ZERO EJECTIONS -- a veto
#  that could not run, presenting as a veto that ran and found the cohort clean.  That is the one
#  outcome this layer is built to make impossible.  An ABSENT column instead trips
#  `_STALE_PANEL_NOT_APPLICABLE`, which declines that pool BY NAME, says which column is missing
#  and says RE-FETCH.  Loud and true beats quiet and wrong.
_VETO_INPUTS = {
    'reitEbitdaInterestCoverage': ('ebitda', 'interestExpense'),
    'producerEbitdaPositive':     ('ebitda', 'revenue'),
    'cashRunwayOneYear':          ('cashAndCashEquivalents',
                                   'netCashProvidedByOperatingActivities'),
    'equityPositive':             ('totalStockholdersEquity',),
}


def veto_missing_inputs(df, metstr):
    """The raw columns `metstr` needs that `df` does not carry, in declaration order.

    Includes the column named by the metric's `Guard`, because a guard whose own input is absent
    cannot refuse anything: `apply_domain_guard` would see an all-False (or raising) mask and the
    admissibility decision -- the thing that makes this column sign-safe -- would silently not
    happen.  A gate that cannot run is not a gate.
    """
    return [c for c in _VETO_INPUTS[metstr] if c not in df.columns]


def calc_veto(df, metstr, rpy=rp.DEFAULT_ROWS_PER_YEAR, guard=None):
    """One VETO column for ONE source's raw cdx-schema frame.  Same shape as `calc_special`.

    `guard`: optional STAGE1_DOMAIN_GUARDS name, declared in `createDicts.BoMetric_veto_dict`
    and applied to the FINISHED column -- these are formulas, so they never pass through
    `build_bometric_rows`'s ratio loop where a declared `Guard` is otherwise applied.

    An unknown key RAISES rather than returning an empty frame, for the reason `calc_special`
    does: the caller assigns the result straight into `BoMetric_df[key]`, so a key with no
    branch would become an ALL-NaN column -- and an all-NaN veto column does not present as a
    missing column, it presents as a cohort that abstains on everything, i.e. as a veto that
    ran and found nothing.  Fail loudly, naming the key.
    """
    if metstr not in _VETO_KEYS:
        raise KeyError(
            "calcMetrics.calc_veto has no formula for %r (known: %r). An unknown key would "
            "become an all-NaN column, which a veto reads as 'abstained on everything' rather "
            "than as 'missing' -- add the branch, or remove the key from "
            "createDicts.BoMetric_veto_dict." % (metstr, list(_VETO_KEYS)))
    res = pd.DataFrame()
    if metstr == 'reitEbitdaInterestCoverage':
        #  Does the rent cover the interest bill -- the ONE solvency question a rent-collecting
        #  leveraged vehicle answers.  Tested `> 1` (the column IS the coverage ratio, so the
        #  unity bar is the bar, not a chosen level).
        #  THE VENDOR'S OWN `ebitda`, not the `operatingIncome + D&A` proxy: the proxy exists
        #  because FMP would not give us EBITDA, and from the 2026-08-05 capture it does.  Using
        #  the proxy here would restate a quantity we now hold -- this repo's worst bug class.
        #  BOTH LEGS ARE THE SAME PERIOD'S FLOW, so the ratio is frequency-invariant and takes NO
        #  `rp.stage1_flow_factor`: a semi-annual filer's 6-month EBITDA is divided by its
        #  6-month interest bill, and the unity bar means the same thing for both filers.
        #  ADMISSIBILITY: `interestExpense > 0` (declared `Guard`), which is what makes this
        #  sign-safe; a REIT with no interest expense arrives as NaN and ABSTAINS
        #  (`FIELD_EVIDENCE['reitEbitdaInterestCoverage'] == 'not_evidence'`) rather than being
        #  read as unable to cover interest it does not owe.  A NaN `ebitda` propagates to NaN
        #  through the division, so the "ebitda notna" half of the gate needs no separate
        #  statement -- and must not get one, or the domain is stated twice.
        res[metstr] = (pd.to_numeric(df['ebitda'], errors='coerce')
                       / pd.to_numeric(df['interestExpense'], errors='coerce'))
    elif metstr == 'producerEbitdaPositive':
        #  Does a PRODUCING miner earn anything at the EBITDA line.  The column holds EBITDA
        #  itself and the veto tests `> 0`; the name states the test, not the contents.
        #  A LEVEL, NOT A RATIO, so it is frequency-invariant in the only sense that matters to a
        #  SIGN test: a 6-month EBITDA has the same sign as the 12-month one it is half of.  No
        #  flow factor, deliberately -- scaling a quantity by a positive constant cannot change
        #  the answer to `> 0`, so applying one would be arithmetic with no meaning.
        #  ADMISSIBILITY: `revenue > 0` (declared `Guard`) -- a pre-revenue explorer is refused
        #  and ABSTAINS; `cashRunwayOneYear` is the flag that judges exactly those names.
        res[metstr] = pd.to_numeric(df['ebitda'], errors='coerce')
    elif metstr == 'cashRunwayOneYear':
        #  Can the company fund TWELVE MONTHS at its current burn: `cash + CFO x rpy > 0`.
        #  THE HORIZON IS DERIVED, NOT CHOSEN, and that is what makes this the one column here
        #  with a bar rather than a sign test: IAS 1.25 and ASC 205-40 both require management to
        #  assess going concern over AT LEAST TWELVE MONTHS, so twelve months is the STATUTORY
        #  horizon and this column inherits it.  No percentile, no tuned level.
        #  `rpy` IS WHAT MAKES IT TWELVE MONTHS FOR EVERY FILER.  CFO is ONE PERIOD's flow, so
        #  `CFO x rpy` annualises it: x4 for a quarterly filer, x2 for a semi-annual one.  Without
        #  it the column would mean twelve months for one filer and SIX for another, which is
        #  precisely the frequency defect this pipeline keeps finding.  `rpy` is the ONE
        #  classification stamped by `fillPreReqdf` and is passed in, never re-derived here.
        #  NOT `rp.stage1_flow_factor`: that decides scaling for declared RATIO keys; this is an
        #  annualisation INSIDE a formula, stated once, beside the arithmetic.
        #  A SUM, so there is no denominator and no sign to invert.  Both operands measured 0.00%
        #  NaN on the panel, so a NaN here is never routine -- hence
        #  `FIELD_EVIDENCE['cashRunwayOneYear'] == 'counts'`: there is no benign channel into a
        #  refusal, and no `Guard` is declared.
        res[metstr] = (pd.to_numeric(df['cashAndCashEquivalents'], errors='coerce')
                       + pd.to_numeric(df['netCashProvidedByOperatingActivities'],
                                       errors='coerce') * float(rpy))
    elif metstr == 'equityPositive':
        #  Is book equity positive at all.  A STOCK, so no `rpy` and no flow factor; a level, so
        #  no denominator and nothing to invert.  ALWAYS ADMISSIBLE -- no `Guard` -- because
        #  `totalStockholdersEquity` is never absent and a degenerate one is adverse on any
        #  reading, which is the same shape as `returnOnAssets`'s ruling.
        res[metstr] = pd.to_numeric(df['totalStockholdersEquity'], errors='coerce')

    if guard is not None:
        res[metstr] = apply_domain_guard(df, res[metstr].tolist(), guard)

    return res