import pandas as pd
import numpy as np
import warnings

import nan_policy as npol
import reporting_period as rp

# Every window here is written for QUARTERLY rows and parameterised by `rpy` (rows per
# year: 4 quarterly, 2 semi-annual -- reporting_period).  A semi-annual filer's row is a
# SIX-month flow, so its TTM is 2 rows, its YoY shift is 2 rows, and its window-edge trim
# is 2 rows.  rpy defaults to 4, so quarterly names are bit-identical.

# Suppress FutureWarning about DataFrame concatenation with empty/all-NA entries
warnings.filterwarnings('ignore', message='.*concatenation with empty or all-NA entries.*')

# --- Montier C-score: ONE cutoff for the whole pipeline -----------------------
# THE single source of truth for "the C-score is high enough to surface for review".
# forensicFlags imports this constant, so the flag column, the presentation and the
# legacy problemlist below can never drift apart again.
#
# It used to be TWO different cutoffs: calcMontierC below used a strict `cscore > 4`
# for problemlist_Cscore while forensicFlags/the presentation used `C >= 4`.  The
# 2026-07-17 forensic CSV therefore shipped two columns that CONTRADICT each other on
# 12 of 90 names -- every name whose C_Score_mean is exactly 4.0 (ALFA.L, EVER, LSC.L,
# STNG, QLYS, FFIV, 0RJ6.L, BKE, CPA, 0RV0.L, 0QQN.L, 0QO1.L) is flagged True by
# `C_flag_ge_4` and False by `legacyProblemC_strict_gt4` in the same row.  A C_Score of
# exactly 4 is common (the score is a mean of two integer counts), so this was not an
# edge case.  Resolved to >= 4 everywhere (audit H7 / DA "uncaught internal
# inconsistency", 2026-07-19).
#
# THE CUTOFF IS UNCHANGED AT >= 4, NOW OUT OF **5** SCORED FLAGS (ruling, 2026-07-26).
# DAPPdec was REMOVED from the scored set (see C_FLAG_COLS), so the same 4 is now a
# STRICTER proportion than published Montier's 4-of-6.  That is deliberate and was NOT
# rescaled to >= 3/5: with a 5-flag median of 2.00 and positively-correlated flags, >= 3/5
# would fire on a large fraction of the shortlist, which is the "a HIGH flag firing on a
# third of the shortlist trains the reader to ignore it" failure.  It also matches this
# module's already-ratified posture a few lines down: "Under-counting is the safe
# direction for a review flag."
C_FLAG_CUTOFF = 4        # surface for review when C_Score_mean >= 4 (of 5 scored flags)

# The FIVE Montier red-flag columns that are SCORED, in the order they are counted.
#
# DAPPdec IS DELIBERATELY ABSENT (ruling, 2026-07-26).  It was POSITIVE BY CONSTRUCTION:
# `adTTM` is a running cumulative sum over the fetched window, so `gppeTTM` grows
# monotonically, `dappTTM` shrinks monotonically, and `-diff` is positive almost always.
# It fired on 67.8% of names (mean 0.650) against 0.271-0.472 for the other five, and
# 13 of the 16 top-100 C>=4 flags rested on it alone.
#
# It is REMOVED rather than repaired because there is no way to measure it correctly here:
#   * the correct denominator is GROSS PP&E, and gross PP&E / accumulated depreciation are
#     ABSENT from FMP's standard balance sheet (54 keys, identical US and non-US).  They
#     exist only on per-ticker `financial-statement-full-as-reported` -- ~8,000 extra calls,
#     about +20% on a 12-hour run, US-GAAP-only and ragged by filer and period.  That would
#     make the flag's population FILER-DEPENDENT, i.e. it swaps a construction bias for a
#     SELECTION bias on the very layer whose firing rate we are trying to make trustworthy.
#   * the cheap substitute `ddaTTM/nppeTTM` was rejected too: its YoY change is
#     capex-dominated, so it would fire on the INVESTING firm and clear the DIVESTING one,
#     and it would collide with flag `TAgr > 10%` -- two of five flags keying off asset
#     growth, which is the same by-construction cancellation the audit found in Beneish
#     DSRI, newly introduced instead of inherited.
# So the real choice was measure-it-wrongly vs do-not-measure-it, and not measuring it is
# the honest one.  The computation is DELETED, not demoted -- see the dated decision block
# in calcMontierC for the full reasoning and the two ways NOT to bring it back.
C_FLAG_COLS = ['NICFOdiv', 'DSOinc', 'DSIinc', 'OCARinc', 'TAgr']

# =========================================================================== #
#  THE BENEISH BASE FLOOR -- AQI and DEPI (undefended-denominator fix, 2026-08-14)
# =========================================================================== #
#  THE DEFECT.  Every Beneish component except TATA is an INDEX -- a quantity divided by its
#  own value one year earlier.  TATA's denominator was guarded in the 2026-07-19 audit fix
#  (`taLevel.where(taLevel > 0)`, a few lines into the M-score below); the index denominators
#  were not.  So a base that rounds to nothing produces an index of arbitrary size, which then
#  enters a LINEAR discriminant and carries the whole verdict.
#
#  IT IS LIVE, AND IT REACHES THE CEO.  On the SHIPPED 2026-08-13 CUR3K run
#  (`ForensicFlagsTop100-2026-08-13`):
#      WSE     rank 22   M_Score_mean = +25.4980   driver `AQI(+26.16)`  -- AQI = 67.41.
#              Its aqiTTM ROSE from 0.006326 (2025-01) to 0.426444 (2026-01), and AQI is
#              current/prior, so 0.426444 / 0.006326 = 67.41.  (An earlier version of this
#              note said it "fell from 0.4264 to 0.006326" -- that is the newest-first LIST
#              order read as if it were time, and it states the case backwards.)  The name is
#              flagged a Beneish manipulator 25 points past a cutoff of 0 on the strength of
#              ONE index sitting ~60x outside the range Beneish fitted the coefficients on.
#      HWX.TO  rank  5   M_Score_mean =  +0.9618   driver `AQI(+2.50)`   -- AQI = 10.93/13.07
#              on bases of 0.002417 / 0.002093.
#  Panel-wide (2,629 sources, 45,322 computable AQI rows) the index spans -2.42e8 to +6.53e7
#  and DEPI spans -1,334 to +455,627, i.e. a single DEPI row can move M by 52,397.
#
#  WHY A MAGNITUDE FLOOR IS EXPRESSIBLE FOR THESE TWO AND NOT FOR THE OTHER FIVE.  aqiTTM and
#  depiTTM are the only two Beneish bases that are DIMENSIONLESS SHARES -- "the fraction of
#  total assets that is neither current nor net PP&E" and "D&A as a fraction of D&A + net PP&E".
#  An absolute floor on a share is scale-free: it needs no per-currency, per-size or per-sector
#  calibration and it means the same thing for every filer.  DSRI's base is in DAYS and SGI's is
#  in CURRENCY, so the same constant would be meaningless there -- see the OPEN ITEM at the end.
#
#  THEY ARE *NOT* "BOUNDED IN [0,1] BY CONSTRUCTION" -- an earlier version of this block said so
#  and it is false (review F-2).  MEASURED, aqiTTM runs -14,550.37 to 1.0000: the bound holds
#  only where the balance-sheet identity holds, and this block's own argument is that it
#  sometimes does not.  That is why there is a SECOND, separate guard below.
#
#  ---- (a) aqiTTM: A MAGNITUDE FLOOR **AND** AN IDENTITY GUARD ------------------------------
#  THE FLOOR IS 0.01, AND IT IS READ OFF THE DENSITY (review F-3).  An earlier derivation here
#  argued from the size of the panel's identity violations; corrected, that argument points at
#  ~0.025 rather than 0.01, so it is replaced by the direct evidence.  The distribution of
#  aqiTTM in 0.005-wide bins [panel = HomeGDrive/pipeline Boresults_dic CUR3K 2026-08-13]:
#
#      [0.000,0.005) 2164   [0.015,0.020)  760   [0.035,0.040)  687
#      [0.005,0.010)  916   [0.020,0.025)  778   [0.040,0.045)  704
#      [0.010,0.015)  743   [0.025,0.030)  742   [0.045,0.050)  668
#
#  Flat at ~740 rows/bin from 0.010 upward, with ~1,600 rows of EXCESS piled below it.  That
#  excess is the degenerate population -- bases that are the residue of subtracting two
#  near-equal balance-sheet aggregates rather than a measured composition -- and it ENDS at
#  0.010.  The floor is placed where the excess ends, which is evidence the panel states
#  directly instead of an estimate from a selected tail.
#
#  THE IDENTITY GUARD IS SEPARATE AND DOES THE HEAVY LIFTING ON THE EXTREME.
#  `totalCurrentAssets + propertyPlantEquipmentNet <= totalAssets` is an identity, so aqiTTM
#  outside [0,1] is ARITHMETICALLY IMPOSSIBLE -- 160 of 54,957 rows (0.291%).  Those rows are
#  not small, they are WRONG, so no magnitude floor can reach them.  Measured effect on the
#  largest single-row |0.404 x AQI| contribution to one M row:
#
#      no guard                        97,671,954.40      282 rows contribute > 10
#      floor 0.01 alone                   136,644.81       43        <- SHIPPED FIRST, INADEQUATE
#      identity [0,1] alone            26,379,737.88      248
#      floor 0.01 + identity                   36.32       40        <- BOTH ARE NEEDED
#
#  The floor alone leaves 136,644 and is UNCHANGED at every floor from 0.001 to 0.02, because
#  the residual comes from a base that is large-magnitude and NEGATIVE.  The identity alone
#  leaves 26.4M.  Only together do they bound the term.  Cost of adding the identity: 160 rows
#  (0.291%) on top of the floor's 3,151 (5.734%).
#
#  ---- (b) depiTTM: ITS OWN FLOOR, ON ITS OWN EVIDENCE (review F-4) -------------------------
#  depiTTM DOES NOT INHERIT AQI's CONSTANT.  It was set to 0.01 by sharing a symbol, and that
#  refused 4,320 rows (8.0%) while changing ZERO verdicts on the shipped shortlist -- a large
#  coverage cost for nothing the CEO sees.  Its own density tells a different story:
#
#      [0.000,0.002) 1797   [0.006,0.008)  659   [0.012,0.014) 1253
#      [0.002,0.004)  340   [0.008,0.010) 1046   [0.014,0.016) 1365
#      [0.004,0.006)  422   [0.010,0.012) 1305   [0.016,0.018) 1254
#
#  A sharp, ISOLATED spike of 1,797 rows in [0, 0.002) against 340 in the very next bin, and
#  then a smooth rise to the body of the distribution.  depiTTM's degenerate population ends at
#  0.002, not 0.01.  Everything between is a real, low depreciation rate -- long-lived assets:
#  REITs, utilities, shipping -- and refusing those is refusing a measurable depreciation policy
#  for being unusual, which is not what this guard is for.
#  THE IMPLIED-LIFE ARGUMENT IS WITHDRAWN, not quietly dropped: 0.01 was justified here as a
#  99-year implied life (life = 1/depi - 1), but the panel's own p90 is already ~84 years, so
#  the cut sat INSIDE the population.  At 0.002 the implied life is ~500 years, which is
#  genuinely outside anything depreciable.
#  MEASURED TRADE, largest single-row |0.115 x DEPI| contribution and base rows refused:
#      floor 0      52,397.16  (128 rows > 10)   0.000%
#      floor 0.002      56.82  ( 19 rows > 10)   3.297%   <- CHOSEN
#      floor 0.005      29.13  (  8 rows > 10)   4.296%
#      floor 0.01        9.11  (  0 rows > 10)   7.843%
#  0.002 kills three orders of magnitude of the extreme for 3.3%; the last factor of six costs
#  2.4x that coverage and moves nothing on the shortlist.  The residual 56.82 is comparable to
#  the AQI limb's 36.32 and is disclosed rather than hidden.
#
#  ---- (c) COVERAGE: A PARTIALLY-REFUSED WINDOW IS NOT A TRAILING-YEAR MEAN (review F-1) -----
#  `M_Score_mean` is `M_Score.head(scale_window(4, rpy)).mean()`, and pandas' `mean()` SKIPS
#  NaN -- so refusing SOME rows of the window silently shortens it while the column keeps its
#  trailing-year label.  That is the same "number wearing a label it does not own" defect this
#  change fixes elsewhere, and the floor is what creates it.  The window is now gated on
#  `nan_policy.COVERAGE_MIN` (0.50), the repo's own rule for exactly this question.
#  AN EARLIER VERSION OF THIS BLOCK DEFERRED THAT FIX ON A REASON THAT WAS MEASURABLY WRONG --
#  it claimed the coverage rule "would also change the three names the floor never touched".
#  `nan_policy.py`'s test is a STRICT `<`, and those three (LOUP.PA, NEDAP.AS, MAU.PA) sit at
#  EXACTLY 0.50 and pass.  Measured on the shipped top-100 it changes exactly ONE name:
#      HWX.TO   w=4  present=4  computable=1  coverage 0.25  -> REFUSED
#  which is the point -- HWX.TO is RANK 5, and without this its manipulation flag is removed
#  (+0.9618 -> -1.3159) on the strength of a mean over ONE of its four quarters.
#
#  MEASURED COST OF THE WHOLE GUARD, same panel, against the shipped run.  The OFF arm
#  (both guards neutralised, COVERAGE_MIN = 0) reproduces `ForensicFlagsTop100-2026-08-13` to
#  4.99e-05 -- its 4-dp rounding -- with an identical NaN pattern, so the before column is the
#  shipped artifact and not a re-derivation of it:
#      shortlist  97 scored / 9 flagged M>0 / 3 abstain  ->  92 / 7 / 8.
#                 Newly abstaining: WSE (was +25.4980, the largest flag on the deck) and
#                 CFX.L, PEY.TO, TOT.TO on the bases; HWX.TO on COVERAGE.
#                 TWO verdicts change, both from a flag to an abstention: WSE and HWX.TO.
#      panel      127 of 2,629 sources (4.83%) go from an M_Score_mean to an abstention, 0
#                 regain one.  (It was 193 = 7.34% while depiTTM shared AQI's 0.01 floor; two
#                 thirds of that cost was the DEPI limb buying nothing -- see (b).)
#  ABSTAINING DOES NOT HIDE A NAME.  `calcBeneishM` already appends a NaN M to `problemlist`,
#  and `forensicFlags` renders it as 'data-incomplete: dig-deeper' rather than as a red
#  manipulation flag -- the same treatment the 2026-07-26 domain review gave the D&A == 0 case
#  ("UNCOMPUTABLE != MAXIMALLY SUSPICIOUS").  The change is that a name stops carrying a
#  precise-looking number the model cannot support, not that it stops being reviewed.
#
#  THE FLOOR IS APPLIED TO THE WHOLE BASE SERIES, NOT ONLY TO THE DIVISOR LEG.  A near-zero
#  base in the NUMERATOR gives an index near 0, which is bounded and so looks harmless -- but
#  it is the same non-measurement, and it lands on the index's BEST side, where the 2026-07-26
#  ruling is that an uncomputable component must never score as the favourable one.  Flooring
#  the series makes BOTH legs honest and makes the guard symmetric with `apply_domain_guard`,
#  which likewise refuses the LEVEL before any difference or ratio is taken.
#
#  NOT A DEPTH FIX, AND THE DEPTH CLAIM IT WAS PROPOSED UNDER DOES NOT REPRODUCE.  The floor
#  was requested partly to remove a reported "out-of-window rows move M_Score_mean by up to
#  4.98".  MEASURED on this panel, arm A = full panel vs arm B = each source truncated to its
#  newest 16 rows (mathematically a superset of every input the head(4) mean depends on, since
#  the deepest row read is i + 2*rpy - 1 <= 11): 392 of 2,312 computable sources move, median
#  2.22e-16, p99 6.40e-14, MAX 6.49e-12, and ZERO verdict flips.  Nothing of the 4.98 class
#  exists here.  The floor does NOT remove that residual either -- it is ordinary floating-point
#  noise in the other six components' rolling sums and is unchanged at every floor tested.  So
#  this fix is justified by the UNDEFENDED DENOMINATOR alone, which is real and live; the depth
#  story is not evidence for it and is not claimed as such.
#
#  OPEN, AND ONLY HALF CLOSED BY THE INPUT-SANITY GUARD (nan_policy section 5, 2026-08-14).
#  17 names carried |M| > 100 after the AQI/DEPI guard and 3 exceeded 1,000, the largest being
#  RDZN at -67,865 with JHX at -50,323.  An earlier version of this note said "those are NOT
#  near-zero-base cases" and generalised RDZN's scale corruption to all 17.  THAT WAS WRONG,
#  and the corrected decomposition is the reason the residual is what it is:
#
#      3 of 17 are RAW-INPUT SCALE CORRUPTION -- RDZN, JHX and SSRM.  RDZN's are TWO SINGLE
#        CELLS (SG&A 1.090059e13 at 2025-10-01; totalCurrentLiabilities 6.136344e13 at
#        2024-10-01), each smeared over four windows by `invrollsumTTM` -- the claim that its
#        SG&A was corrupt "across four rows" was the TTM sum being read as the raw panel.
#      3 of 17 are driven by TATA, i.e. by a DEGENERATE `totalAssets` LEVEL DENOMINATOR --
#        ALDAR.PA (-142.5), EBON (-391.6) and SSRM (-137.7).  TATA is not one of the five
#        indices, so this is a SIXTH defect class that neither the base floor nor the input
#        guard addresses, and it is named here rather than left inside a miscount.
#      the remaining 11 ARE near-zero-base cases on the five STILL-UNGUARDED index
#        denominators (DSRI/GMI/SGI/SGAI/LVGI) -- e.g. SAP.TO's SGAI base falls to 5.2e-5,
#        and BLDP's gross margin crosses zero (-1.383 -> +0.000996, GMI = -1,389).
#
#  MEASURED AFTER THE INPUT-SANITY GUARD: 17 -> 14 names at |M| > 100, 3 -> 1 above 1,000.
#  RDZN, JHX and CCAP drop out.  So the guard closes the scale-corruption class and leaves the
#  near-zero-base class on the other five denominators OPEN -- that is the next instrument, and
#  it is a base floor like AQI's, NOT another input check.  DSRI's base is in DAYS and SGI's in
#  CURRENCY, so neither takes AQI's dimensionless constant; each needs its own derivation.
#  None of the 14 is in the shipped top-100, so none is on the CEO's desk today.
BENEISH_AQI_SHARE_FLOOR = 0.01
BENEISH_DEPI_SHARE_FLOOR = 0.002


def _floor_share_base(share, floor, unit_interval=False):
    """`share` with every row the guard refuses replaced by NaN.

    `share` is a dimensionless balance-sheet share (aqiTTM or depiTTM), newest-first.  Returns
    (guarded series, number of rows refused) -- the count so `calcBeneishM` can report what the
    guard did in the run log, in the same style as the Montier C-score summary.

    Two independent conditions, because they catch different failures (review F-2):
      floor         : |share| < floor -- the base is too SMALL to be a measurement.
      unit_interval : share outside [0, 1] -- the base is arithmetically IMPOSSIBLE.  Those
                      rows are not small, they are WRONG, so no magnitude floor reaches them;
                      on this panel they are what leaves a 136,644 AQI term standing.

    NaN in, NaN out: a row that was already non-computable is not counted as a refusal, so the
    printed number is what THESE rules removed and not the column's pre-existing gaps.

    THE FLOOR IS PASSED IN, NOT READ FROM A DEFAULT ARGUMENT.  A default is evaluated ONCE at
    import, so `detectManipulation.BENEISH_AQI_SHARE_FLOOR = x` -- which is how every constant
    in this repo is overridden for a measurement or a test (cf. `sm.CYCLEHEAT_BASE_NQ` in
    baseline_tools/test_eps_mean_window) -- would have had NO EFFECT while appearing to work.
    That happened while this change was being measured, and the first before/after reported "no
    change" for a reason that had nothing to do with the data.  The two call sites read the
    module constants by name at call time.
    """
    v = pd.to_numeric(share, errors='coerce').replace([np.inf, -np.inf], np.nan)
    refused = v.notna() & (v.abs() < float(floor))
    if unit_interval:
        refused |= v.notna() & ((v < 0.0) | (v > 1.0))
    return v.where(~refused), int(refused.sum())


def detectManipulationWrapper(resdic):
    symblist = list(resdic['postRank']['source'])
    # ONE classification for both forensic models (the same map Stage-2 derives, since
    # both come from the same cdx_df).
    freq_map = rp.frequency_by_source(resdic['cdx_df'], verbose=True)
    mscore_df, SLmeanMscore, problemlist_Mscore = calcBeneishM(resdic, symblist, freq_map)

    cscore_df, SLmeanCscore, problemlist_Cscore = calcMontierC(resdic, symblist, freq_map)

    detmandic = {'mscore_df': mscore_df, 'SLmeanMscore': SLmeanMscore, 'problemlist_Mscore': problemlist_Mscore,
                 'cscore_df': cscore_df, 'SLmeanCscore': SLmeanCscore, 'problemlist_Cscore': problemlist_Cscore}

    return detmandic

def _toNewestFirst(df):
    """Normalize a per-symbol frame to NEWEST-FIRST (row 0 = most recent quarter).

    The forensic M/C formulas and the recency window (head(...)) below are all
    written to this single, explicit orientation. The upstream cdx_df is
    oldest-first and is left untouched (other modules depend on that ordering);
    the flip is local to the forensic computation. Sorting by parsed date makes
    the orientation robust to however the rows happen to arrive."""
    return (df.sort_values('date', key=lambda s: pd.to_datetime(s), ascending=False)
              .reset_index(drop=True))


def _yoyCurOverPrior(ttm, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """current / prior-year, on NEWEST-FIRST data.

    Row k is the current period; row k+rpy is the SAME period one year earlier (4 rows
    for a quarterly filer, 2 for a semi-annual one).  Beneish DSRI/AQI/SGI/SGAI/LVGI are
    all defined current/prior."""
    return ttm / ttm.shift(-int(rpy))


def _yoyPriorOverCur(ttm, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """prior-year / current, on NEWEST-FIRST data.

    Beneish GMI and DEPI are defined prior/current (a DECLINE in gross margin or
    in the depreciation rate pushes the index ABOVE 1.0, the suspicious side)."""
    return ttm.shift(-int(rpy)) / ttm


def calcMontierC(resdic, symblist, freq_map=None):
    cdx_df = resdic['cdx_df']
    if freq_map is None:
        freq_map = rp.frequency_by_source(cdx_df)
    SLmeanCscore = pd.DataFrame(columns=['source', 'C_Score_mean'])
    SLmeanCscore['source'] = symblist
    cdf = pd.DataFrame(columns=['date', 'symbol', 'NICFOdiv','DSOinc','DSIinc','OCARinc','TAgr','C_Score'])
    problemlist = []
    for symbol in symblist:
        tmpcdf = pd.DataFrame(columns=['date', 'symbol', 'NICFOdiv','DSOinc','DSIinc','OCARinc','TAgr','C_Score'])
        # C-score if NICFO > 0, DSOinc > 0 ...
        # Newest-first so diff(-4)/pct_change(-4) read current-minus-prior (a YoY
        # INCREASE is the suspicious side for every Montier flag) and head(...)
        # summarizes the MOST RECENT quarters.
        tempcdx_df = _toNewestFirst(cdx_df[cdx_df['source'] == symbol])

        # MISSING DATA MUST NOT FIRE A RED FLAG (audit H7 fix, 2026-07-19).
        # Every flag below used to end in `.fillna(99999)`, i.e. a period whose YoY
        # change could not be COMPUTED was scored as a maximal INCREASE -- the
        # suspicious side -- so absent data manufactured forensic red flags.  It was
        # also inconsistent: the since-deleted DAPPdec negated AFTER the fill (-99999), so
        # for that one flag missing data could never fire while for the others it always did.
        # NaN is now left as NaN and `> 0` treats it as NOT FIRED.
        #
        # Incidence note (measured, not assumed): on the shipped 2026-07-17 top-100 the
        # 99999 fill reached the 2-quarter scoring window for 0 of 90 names, so this is
        # a LATENT defect on that run -- the window sits 4+ rows away from the series
        # edge where the fill normally lands.  It fires whenever a CURRENT-period
        # component is genuinely absent (a name with no inventory line, a blank
        # statement-quarter -- ~7.7% of sources carry at least one, audit M-4).
        #
        # KNOWN RESIDUE (not fixed here, flagged): a name with a non-computable
        # component now scores LOWER and so reads as "cleaner" rather than as
        # "incomplete".  Under-counting is the safe direction for a review flag, but the
        # honest treatment is a per-name not-computable count surfaced beside the score.
        _rpy = rp.rows_per_year(freq_map, symbol)
        cfoTTM = invrollsumTTM(tempcdx_df['netCashProvidedByOperatingActivities'], _rpy)
        niTTM = invrollsumTTM(tempcdx_df['netIncome'], _rpy)
        NICFO = (niTTM - cfoTTM)/cfoTTM.abs()
        tmpcdf['NICFOdiv'] = NICFO.diff(periods=-_rpy)

        dsoTTM = invrollsumTTM(tempcdx_df['daysSalesOutstanding'], _rpy)
        tmpcdf['DSOinc'] = dsoTTM.diff(periods=-_rpy)

        dsiTTM = invrollsumTTM(tempcdx_df['daysOfInventoryOutstanding'], _rpy)
        tmpcdf['DSIinc'] = dsiTTM.diff(periods=-_rpy)

        ocarTTM = (invrollsumTTM(tempcdx_df['otherCurrentAssets'], _rpy)
                   / invrollsumTTM(tempcdx_df['revenue'], _rpy))
        tmpcdf['OCARinc'] = ocarTTM.diff(periods=-_rpy)

        # ==================== DAPPdec: DELETED 2026-07-26. DO NOT RESTORE. ============
        # Montier's 5th flag (declining depreciation rate) used to be computed here as
        #     adTTM   = ddaTTM cumulative-summed over the panel
        #     gppeTTM = nppeTTM - capex + adTTM        # a PROXY for gross PP&E
        #     DAPPdec = -(ddaTTM/gppeTTM).diff(-rpy)
        # It was POSITIVE BY CONSTRUCTION: `adTTM` is a cumsum from the panel's start, so
        # gppeTTM grows monotonically, dappTTM falls monotonically, and a test for "the
        # depreciation rate DECREASED" is pre-satisfied before any company data is read.
        # Measured fire rate 0.666 versus 0.271-0.472 for the other five flags, and
        # 13 of the 16 C>=4 flags on the 2026-07-17 top-100 depended on it alone.
        #
        # It is DELETED rather than demoted to a diagnostic: a demoted column still ships in
        # ForensicFlagsTop100-<date>.csv and can still reach problemlist_Cscore, so the
        # broken artifact would remain in the artifacts.  Deleting at source makes every
        # surface follow from the shared C_FLAG_COLS / C_FLAG_CUTOFF constants.
        #
        # THERE IS NOTHING TO RECALIBRATE IT AGAINST.  A correct denominator needs GROSS
        # PP&E or accumulated depreciation, and neither is on FMP's standard balance sheet
        # (54 keys, identical US and non-US).  They exist only on per-ticker
        # `financial-statement-full-as-reported`: ~8,000 extra calls (~+20% on a 12-hour
        # run), probably US-GAAP-only, ragged by filer and period -- which would make the
        # flag's POPULATION filer-dependent, i.e. a selection bias in place of a
        # construction bias, on the exact layer whose firing rate we are trying to trust.
        #
        # DO **NOT** REINSTATE IT FROM `propertyPlantEquipmentNet`.  ddaTTM/nppeTTM measures
        # CAPEX INTENSITY, not depreciation policy: its YoY change is capex-dominated, so it
        # would fire on the INVESTING firm and clear the HARVESTING one -- backwards -- and
        # it would triple-count asset growth alongside `TAgr` and Beneish `SGI`.
        #
        # NOT to be harmonised with Beneish DEPI, which makes the same net-PP&E
        # substitution and is deliberately LEFT ALONE: there it carries the smallest
        # coefficient (0.115) inside a continuous weighted sum with a measured-negligible
        # contribution, which is tolerable; here it was one binary vote of six against a
        # hard cutoff, which is not.
        # ==============================================================================

        taTTM = invrollsumTTM(tempcdx_df['totalAssets'], _rpy)
        tmpcdf['TAgr'] = taTTM.pct_change(-_rpy, fill_method=None) - 0.1

        # Count over the SIX flag columns explicitly (not `tmpcdf > 0` over every
        # column) so the score can never pick up a stray column; NaN > 0 is False, so a
        # non-computable flag simply does not fire.
        tmpcdf['C_Score'] = (tmpcdf[C_FLAG_COLS].apply(pd.to_numeric, errors='coerce')
                             > 0).sum(axis=1)

        tmpcdf['date'] = tempcdx_df['date']
        tmpcdf['symbol'] = tempcdx_df['source']

        # Window-edge trim: `rpy` rows, whose YoY diff has no prior-year counterpart
        # (a semi-annual filer only loses its oldest 2).
        symb_cscore = tmpcdf[0:len(tmpcdf)-_rpy]

        cdf = pd.concat([cdf, symb_cscore])
        # Newest-first: the most recent HALF-YEAR of periods for either frequency --
        # 2 quarters or 1 half (C_WINDOW scaled by rpy).
        cscore = symb_cscore['C_Score'].head(rp.scale_window(2, _rpy, minimum=1)).mean()
        SLmeanCscore.loc[SLmeanCscore['source']==symbol, 'C_Score_mean'] = cscore

        if np.isnan(cscore) or np.isinf(cscore):
            problemlist.append(symbol)
        elif cscore >= C_FLAG_CUTOFF:      # ONE cutoff (was a stricter `> 4` here)
            problemlist.append(symbol)

    # FRAME THE DROP AS A CORRECTION, ONCE, IN THE RUN LOG (ruling 2026-07-26).
    # Removing DAPPdec cuts the C>=4 count about 5x (16 -> 3 on the 07-17 reference pool,
    # median C 2.75 -> 2.00).  Printed with the reason attached so a reader does not
    # conclude "the forensic layer went quiet" -- the flags that disappeared were the ones
    # resting on a component that was positive by construction.
    try:
        _v = pd.to_numeric(SLmeanCscore['C_Score_mean'], errors='coerce')
        print('MONTIER C-SCORE: scored over %d flags %s. The 6th (declining depreciation '
              'rate) was DELETED 2026-07-26 -- positive by construction, and gross PP&E is '
              'not obtainable from the standard balance sheet. cutoff >= %d of %d. '
              'This pool: median C %.2f, C>=%d on %d of %d names. REFERENCE 2026-07-17 pool '
              'for scale: median 2.75 -> 2.00 and C>=4 count 16 -> 3. A LOWER flag count is '
              'a CORRECTION, not a quieter layer: under a binomial null P(X>=4) is 8.9%% at '
              '5 flags vs 24.9%% at 6, so the SAME cutoff is now STRICTER.'
              % (len(C_FLAG_COLS), C_FLAG_COLS, C_FLAG_CUTOFF, len(C_FLAG_COLS),
                 _v.median(), C_FLAG_CUTOFF, int((_v >= C_FLAG_CUTOFF).sum()),
                 int(_v.notna().sum())), flush=True)
    except Exception as _e:
        print('WARNING: C-score summary skipped (%s)' % _e, flush=True)

    return cdf, SLmeanCscore, problemlist

def calcBeneishM(resdic, symblist, freq_map=None):
    cdx_df = resdic['cdx_df']
    if freq_map is None:
        freq_map = rp.frequency_by_source(cdx_df)
    SLmeanMscore = pd.DataFrame(columns=['source', 'M_Score_mean'])
    SLmeanMscore['source'] = symblist
    mdf = pd.DataFrame(columns=['date', 'symbol', 'DSRI','GMI','AQI','SGI','DEPI','SGAI','LVGI','TATA','M_Score'])
    problemlist = []
    #  BENEISH_SHARE_FLOOR refusals, counted per run so the summary below states what the guard
    #  actually did on THIS pool.  Local, not a module accumulator: `nan_policy.reset_counts`
    #  documents what a module-level counter costs when a process scores twice.
    n_floor = {'AQI': 0, 'DEPI': 0}
    #  names whose M window fell below nan_policy.COVERAGE_MIN once the bases were guarded
    n_coverage = [0]
    for symbol in symblist:
        tmpmdf = pd.DataFrame(columns=['date', 'symbol', 'DSRI','GMI','AQI','SGI','DEPI','SGAI','LVGI','TATA','M_Score'])
        # Newest-first so invrollsumTTM's trailing-4-quarter sum, the YoY helpers
        # (prior = 4 rows older) and head(...) all read one known orientation.
        tempcdx_df = _toNewestFirst(cdx_df[cdx_df['source'] == symbol])
        # DSRI = DSO_t / DSO_{t-4}  -- the PUBLISHED index (audit H6 fix, 2026-07-19).
        # daysSalesOutstanding IS ALREADY the receivables-to-sales ratio (AR/Sales*365),
        # so published DSRI = (AR/Sales)_t / (AR/Sales)_{t-1} = DSO_t / DSO_{t-1}.
        # Dividing by salesTTM a SECOND time made this
        #     DSRI_coded = DSRI_true / SGI,
        # which is not an index of anything and, because SGI enters the M-score with the
        # other large positive coefficient (+0.892), meant Beneish's two biggest positive
        # terms partially CANCELLED by construction.  Measured on the shipped 2026-07-17
        # top-100: spearman(DSRI_coded, SGI) = -0.648 vs -0.180 for the published form,
        # and DSRI_coded * SGI reproduced the published DSRI to a median abs error of
        # 2.7e-03 -- i.e. the 1/SGI contamination was exact, not incidental.
        _rpy = rp.rows_per_year(freq_map, symbol)
        dsoTTM = invrollsumTTM(tempcdx_df['daysSalesOutstanding'], _rpy)
        salesTTM = invrollsumTTM(tempcdx_df['revenue'], _rpy)
        tmpmdf['DSRI'] = _yoyCurOverPrior(dsoTTM, _rpy)     # DSO_t / DSO_{t-1}

        gmiTTM = invrollsumTTM(tempcdx_df['grossProfitMargin'], _rpy)
        tmpmdf['GMI'] = _yoyPriorOverCur(gmiTTM, _rpy)      # GM_{t-1} / GM_t  (margin decline -> >1)

        tcaTTM = invrollsumTTM(tempcdx_df['totalCurrentAssets'], _rpy)
        ppenTTM = invrollsumTTM(tempcdx_df['propertyPlantEquipmentNet'], _rpy)
        taTTM = invrollsumTTM(tempcdx_df['totalAssets'], _rpy)
        aqiTTM = 1- (tcaTTM + ppenTTM)/taTTM
        #  UNDEFENDED DENOMINATOR (fix 2026-08-14) -- see BENEISH_SHARE_FLOOR.  aqiTTM is the
        #  share of total assets that is neither current assets nor net PP&E; below 1% of the
        #  balance sheet it is the same size as the panel's own measured inconsistency in those
        #  three lines, so the YoY index built on it is a ratio of error terms.  REFUSED, which
        #  makes AQI (and hence M) NaN -- 'data-incomplete: dig-deeper' downstream.
        aqiTTM, _nf = _floor_share_base(aqiTTM, BENEISH_AQI_SHARE_FLOOR,
                                        unit_interval=True)
        n_floor['AQI'] += _nf
        tmpmdf['AQI'] = _yoyCurOverPrior(aqiTTM, _rpy)      # AQI_t / AQI_{t-1}

        sgiTTM = invrollsumTTM(tempcdx_df['revenue'], _rpy)
        tmpmdf['SGI'] = _yoyCurOverPrior(sgiTTM, _rpy)      # Sales_t / Sales_{t-1}

        #z = x / (x + y) = 1 / ((x + y) / (x)) = 1 / (1 + (y / x))
        # Ef w = y / x, þá: z = 1 / (1 + w). x = depreciationAndAmortization, y = PP&Enet
        ddaTTM = invrollsumTTM(tempcdx_df['depreciationAndAmortization'], _rpy)
        w = ppenTTM/ddaTTM
        depiTTM = 1/(1+w)                                   # depreciation rate = Dep/(Dep+PPE)
        #  SAME GUARD, SAME REASON -- see BENEISH_SHARE_FLOOR.  depiTTM below 0.01 is an implied
        #  depreciation life over 99 years, i.e. D&A is not measuring the reported PP&E.  This
        #  ALSO subsumes the 2026-07-26 D&A == 0 case: there w = +inf gave depiTTM = 0 exactly
        #  and DEPI = +-inf, which the replace() below already converted to NaN.  The floor now
        #  refuses the CONTINUOUS neighbourhood of that point too, which is where the finite but
        #  meaningless indices lived (largest single-row DEPI contribution to M: 52,397 -> 9.11).
        depiTTM, _nf = _floor_share_base(depiTTM, BENEISH_DEPI_SHARE_FLOOR)
        n_floor['DEPI'] += _nf
        tmpmdf['DEPI'] = _yoyPriorOverCur(depiTTM, _rpy)    # rate_{t-1} / rate_t  (decline -> >1), full-year YoY

        sgaTTM = invrollsumTTM(tempcdx_df['sellingGeneralAndAdministrativeExpenses'], _rpy)
        sgaiTTM = sgaTTM/sgiTTM
        tmpmdf['SGAI'] = _yoyCurOverPrior(sgaiTTM, _rpy)    # (SGA/Sales)_t / (SGA/Sales)_{t-1}

        ltdTTM = invrollsumTTM(tempcdx_df['longTermDebt'], _rpy)
        clTTM = invrollsumTTM(tempcdx_df['totalCurrentLiabilities'], _rpy)
        lvgiTTM = (ltdTTM+clTTM)/taTTM
        tmpmdf['LVGI'] = _yoyCurOverPrior(lvgiTTM, _rpy)    # Leverage_t / Leverage_{t-1}

        # TATA = (NI_ttm - CFO_ttm) / TotalAssets_LEVEL  -- published Beneish
        # (audit H6 fix, 2026-07-19).  TWO errors, fixed TOGETHER on purpose:
        #
        #   1. DENOMINATOR SCALE.  taTTM is invrollsumTTM(totalAssets) -- a 4-quarter
        #      SUM of a STOCK, i.e. ~4x the actual asset base.  TTM flows over a 4x
        #      denominator put TATA at ~1/4 scale, which made the model's FLAGSHIP
        #      accruals term (largest coefficient, +4.679) its least influential input.
        #      Total assets is a stock and belongs in the denominator as a LEVEL.
        #   2. THE -cffTTM TERM.  Published TATA is (NI - CFO)/TA; the extra
        #      -financing-cash-flow term is not part of the model.  It is not neutral:
        #      on the shipped 2026-07-17 top-100 the term -cffTTM/taTTM was POSITIVE
        #      for 89 of 90 names, so it added a near-universal upward bias to the
        #      most heavily weighted component -- and the 1/4-scale denominator was
        #      SUPPRESSING that bias.  Fixing the denominator alone would have
        #      QUADRUPLED it, which is why both move in one change.
        #
        # Evidence the fixed form is the right quantity: forensicFlags.computeSloanAccruals
        # independently computes (NI_ttm - CFO_ttm)/avg(TA) for the same names.  Against
        # that reference, the old TATA scored spearman 0.290 (and had the WRONG SIGN on
        # the pool median: +0.0077 vs the reference's -0.0794); the fixed TATA scores
        # spearman 0.848 with a median level of -0.0750.
        #
        # TATA is a current-period LEVEL (higher accruals = more suspicious), so it
        # carries no YoY orientation to invert.
        niTTM = invrollsumTTM(tempcdx_df['netIncome'], _rpy)
        cfoTTM = invrollsumTTM(tempcdx_df['netCashProvidedByOperatingActivities'], _rpy)
        # totalAssets <= 0 is impossible (614 such rows exist in the panel, audit M-3);
        # as a LEVEL denominator it would produce +-inf rather than the ~harmless small
        # number a 4-quarter sum used to hide, so it is coerced to NaN = "not computable".
        taLevel = pd.to_numeric(tempcdx_df['totalAssets'], errors='coerce')
        taLevel = taLevel.where(taLevel > 0)
        tmpmdf['TATA'] = (niTTM - cfoTTM)/taLevel

        # +1.78 folds the -1.78 manipulator cutoff into the stored score, so the
        # stored M>0 is exactly the standard Beneish M>-1.78. Preserved verbatim;
        # it now folds correctly-directed components.
        # UNCOMPUTABLE != MAXIMALLY SUSPICIOUS (domain review S3, fixed 2026-07-26).
        # depreciationAndAmortization == 0 on 5.24% of panel rows (9,295/177,350) makes
        # w = PPE/D&A infinite, depiTTM = 1/(1+inf) = 0 and DEPI = 0/0-or-x/0 = +-inf, which
        # propagates straight through the linear sum: M_Score = +inf.  An infinity then
        # trips `mscore > 0`, lands the name in problemlist_Mscore and renders as a RED
        # manipulation flag (R3 HIGH) -- 1 of 90 top-100 names today (TUYA).  That is the
        # same defect the Montier fix removed when it deleted `.fillna(99999)`: a company
        # with no depreciation line is not a probable fraud, it is not measurable on this
        # component.  Replacing +-inf with NaN makes the M_Score NaN, which forensicFlags
        # reports as 'data-incomplete: dig-deeper' rather than as a red flag.
        for _c in ('DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA'):
            tmpmdf[_c] = pd.to_numeric(tmpmdf[_c], errors='coerce').replace(
                [np.inf, -np.inf], np.nan)
        tmpmdf['M_Score'] = - 4.84 + 0.92*tmpmdf.DSRI + 0.528*tmpmdf.GMI + 0.404*tmpmdf.AQI + 0.892*tmpmdf.SGI +\
                            0.115*tmpmdf.DEPI - 0.172*tmpmdf.SGAI + 4.679*tmpmdf.TATA - 0.327*tmpmdf.LVGI + 1.78
        tmpmdf['date'] = tempcdx_df['date']
        tmpmdf['symbol'] = tempcdx_df['source']
        # Window-edge trim: `rpy` rows, whose YoY has no prior-year counterpart.
        symb_mscore = tmpmdf[0:len(tmpmdf)-_rpy]

        mdf = pd.concat([mdf,symb_mscore])
        # Newest-first: the most recent YEAR of periods for either frequency --
        # 4 quarters or 2 halves (M_WINDOW scaled by rpy).
        #  COVERAGE (review F-1).  `mean()` skips NaN, so a partially-refused window would
        #  otherwise yield a mean over FEWER periods still labelled as the trailing year --
        #  the defect this whole change exists to remove, reintroduced by its own guard.
        #  `nan_policy.COVERAGE_MIN` is the repo's existing rule for this question and is
        #  reused rather than restated, so the two cannot drift apart.
        _win = pd.to_numeric(symb_mscore['M_Score'], errors='coerce').head(
            rp.scale_window(4, _rpy))
        _ok = int(_win.notna().sum())
        if len(_win) and (_ok / float(len(_win))) >= npol.COVERAGE_MIN:
            mscore = _win.mean()
        else:
            #  COUNT ONLY WHAT THIS RULE ACTUALLY CHANGED.  A window with NOTHING computable
            #  already produced NaN (an all-NaN `mean()` is NaN), so counting it here would
            #  report the rule as responsible for every pre-existing gap in the column -- 444
            #  names on the 2026-08-13 panel against the 5 it really converts.  An operator
            #  reading the run log has no way to tell those apart, so the counter does it.
            if _ok:
                n_coverage[0] += 1
            mscore = np.nan
        SLmeanMscore.loc[SLmeanMscore['source']==symbol, 'M_Score_mean'] = mscore

        if np.isnan(mscore) or np.isinf(mscore):
            problemlist.append(symbol)
        elif mscore > 0:
            problemlist.append(symbol)

    #  SAY WHAT THE GUARD DID, in the run log, beside the C-score's own summary.  A refusal that
    #  is invisible is indistinguishable from a name that never had the data -- and the whole
    #  point of the floor is that an ABSTENTION is a different statement from a LOW SCORE.
    try:
        _v = pd.to_numeric(SLmeanMscore['M_Score_mean'], errors='coerce')
        print('BENEISH M-SCORE: base guards refused %d AQI row(s) (|aqi| < %.4g or outside '
              '[0,1]) and %d DEPI row(s) (|depi| < %.4g) as non-measurements -- a share below '
              'the floor makes the year-over-year index a ratio of rounding error, and one '
              'outside [0,1] is arithmetically impossible. %d name(s) had SOME of their M '
              'window left but fell below nan_policy.COVERAGE_MIN = %.2f, so what remained '
              'was not a trailing-year mean. This pool: %d of %d name(s) score, %d flag M > 0, %d '
              'ABSTAIN (reported as data-incomplete, NOT as a manipulation flag).'
              % (n_floor['AQI'], BENEISH_AQI_SHARE_FLOOR, n_floor['DEPI'],
                 BENEISH_DEPI_SHARE_FLOOR, n_coverage[0], npol.COVERAGE_MIN,
                 int(_v.notna().sum()), len(SLmeanMscore), int((_v > 0).sum()),
                 int(_v.isna().sum())), flush=True)
    except Exception as _e:
        print('WARNING: Beneish M-score summary skipped (%s)' % _e, flush=True)

    return mdf, SLmeanMscore, problemlist

def invrollsumTTM(Svec, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """Trailing-twelve-month sum on NEWEST-FIRST data: `rpy` rows, not a fixed 4.

    A semi-annual filer's two rows already cover 12 months; summing four of them built a
    TWENTY-FOUR-month "TTM" and made every ratio built on it (Beneish TATA/SGI/AQI/LVGI,
    the Montier flags, Sloan accruals) ~2x its peers."""
    return (Svec.iloc[::-1].rolling(int(rpy)).sum()).iloc[::-1]
