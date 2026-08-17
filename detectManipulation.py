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
#  near-zero-base class on the other five denominators OPEN.
#  None of the 14 is in the shipped top-100, so none is on the CEO's desk today.
#
#  CLOSED 2026-08-15 BY THE BLOCK BELOW -- AND THE PRESCRIPTION IN THIS PARAGRAPH WAS WRONG.
#  It said the next instrument "is a base floor like AQI's, NOT another input check", and that
#  each of the five needs its own floor in its own units.  Measured, NONE of the five carries a
#  degenerate population a floor could cut off: three fail because a vendor SENTINEL ZERO is
#  summed into the trailing-year base as if it were a measured period, and two (SGI, LVGI) do
#  not fail at all.  The decomposition above also mis-assigns two names -- ALDAR.PA and EBON are
#  GMI-driven once the input-sanity guard is applied, not TATA-driven; SSRM is the only TATA
#  case and the only one of the 14 still standing.  See BENEISH_PERIOD_DOMAIN below.
BENEISH_AQI_SHARE_FLOOR = 0.01
BENEISH_DEPI_SHARE_FLOOR = 0.002

# =========================================================================== #
#  THE OTHER FIVE INDICES -- DSRI, GMI, SGI, SGAI, LVGI  (O-13, 2026-08-15)
# =========================================================================== #
#  THE PRESCRIPTION WAS "FIVE MORE FLOORS LIKE AQI's"; THE PANEL SAYS OTHERWISE, AND THAT
#  REFUSAL IS THE SUBSTANCE OF THIS BLOCK.  Measured on the same panel the AQI/DEPI floors were
#  derived on (HomeGDrive/pipeline Boresults_dic CUR3K 2026-08-13, 2,629 sources / 61,354 rows,
#  read AFTER `nan_policy.refuse_impossible_cells` so it is the state that ships today):
#  NOT ONE of the five carries a degenerate population that a magnitude floor would cut off.
#  Three of them fail for a different reason entirely, and two do not fail at all.
#
#  ---- WHAT THE FIVE ACTUALLY HAVE IN COMMON -----------------------------------------------
#  DSRI, GMI and SGAI are the three Beneish bases that are RATIOS TO SALES -- receivables per
#  day of sales, gross profit per unit of sales, SG&A per unit of sales.  FMP computes the
#  first two for us and emits a literal `0` for the period when it cannot; the third is a raw
#  expense line that a filer may not report separately at all.  `invrollsumTTM` then SUMS those
#  periods into a trailing-year base, so ONE unreported period is silently added as if it were
#  a measured zero -- and the year-over-year index is computed against it.  That, not a small
#  denominator, is what produced the O-13 residual.  The evidence is the density AT zero, where
#  a continuous quantity should have none (panel rows, per field):
#
#      daysSalesOutstanding == 0   7,174 rows | 0<v<0.01: 35   0<v<0.1: 95   0<v<1: 415
#      grossProfitMargin    == 0   4,263 rows | 0<v<1e-4:  0   1e-4<=v<1e-3: 20  <1e-2: 203
#      grossProfitMargin    == 1   2,497 rows | 0.999<v<1: 58  1<v<1.001: 17
#      SG&A                 == 0   3,964 rows | quarterly SG&A/sales <1e-6: 9  <1e-5: 12
#
#  A pile of thousands of rows at EXACTLY zero with a near-empty neighbourhood is a sentinel,
#  not a measurement.  THE CORROBORATION IS PER FIELD, AND IT IS NOT EQUALLY STRONG ON ALL
#  THREE -- said here rather than left for a reader to discover, because the guards' costs run
#  in the opposite order to their evidence.
#    * `grossProfitMargin == 0` -- THE STRONGEST, AND IT IS ARITHMETIC.  4,095 of the 4,263 zero
#      rows sit on a row whose REVENUE IS ALSO ZERO and a further 95 on a row whose revenue is
#      ABSENT: 4,190 of 4,263 (98.3%) are an undefined 0/0 or a missing input, and only 65 are a
#      zero margin on real sales.  (An earlier version of this note said all 4,190 had revenue
#      zero; 95 of them have no revenue figure at all -- review L-4.)
#    * `grossProfitMargin >= 1` -- ALSO ARITHMETIC, AND IT NEEDS NO SENTINEL ARGUMENT.  A margin
#      of 1 or more requires cost of revenue <= 0, which a cost cannot be: 2,339 of the 2,497
#      `== 1` rows carry `grossProfit == revenue` TO THE CENT, and 635 further rows exceed 1
#      outright.  Of the 323 sources with a `== 1` row exactly TWO report it in every period; the
#      rest alternate between 1 and a real margin (GXAI: 1, 0.9516, 1, 0.669, 1).
#    * `SG&A == 0` -- DENSITY ONLY, AND THE CONCRETE CASE.  3,964 exact zeros against 9 rows
#      whose quarterly SG&A/sales is below 1e-6 and 3 in [1e-6, 1e-5): the neighbourhood is
#      empty.  SAP.TO is what that looks like on one name (below).  Note the honest limit: 3,810
#      of the 3,964 sit on a row with revenue > 0, so unlike the margin case this is not an
#      undefined quantity -- it is a line the filer did not report separately.
#    * `daysSalesOutstanding == 0` -- THE WEAKEST OF THE THREE, and the reason is measurable:
#      2,906 of the 7,174 zero rows (40.5%, across 517 sources) sit on a row with revenue > 0,
#      i.e. a company with real sales reporting zero receivable-days, which is genuinely
#      possible for a cash business.  The revenue-corroboration that carries the gross-margin
#      limb DOES NOT carry this one.  What makes it acceptable is that it is also the CHEAPEST
#      limb: leave-one-out, the DSO guard costs 43 names at the margin (739 -> 696 without it)
#      and moves `|M| > 100` not at all.  The weakest evidence is attached to the smallest
#      effect: on the same leave-one-out the gross-margin guard costs 262 names and the SG&A
#      guard 169, and the gross-margin limb is the one whose evidence is arithmetic.
#  THE `longTermDebt` CONTROL IS WITHDRAWN (review M-1), not quietly dropped.  An earlier version
#  of this block argued that `longTermDebt`'s 4,869 exact zeros are all genuine and that the
#  "does the source report it in EVERY period" test "is the test `longTermDebt` would PASS".
#  MEASURED, IT FAILS ITS OWN TEST: only 40 of its 677 sources (5.9%) report zero in every
#  period, against 15.5% for `DSO == 0` and 29.2% for `GM == 0`.  On the sandwich form of the
#  same test (a sentinel row with non-sentinel rows on both sides) the reviewer measured it
#  alternating MORE than two of the three guarded fields.  So it discriminated nothing, and the
#  argument above rests on the arithmetic and the density instead -- which is where it always
#  actually rested.
#
#  AND IT MATTERS WHICH SIDE THE ERROR LANDS ON -- THE SIDE IS NOT THE SAME FOR ALL THREE, and
#  an earlier version of this block got that wrong (review M-3).  DSRI = 0 (coefficient +0.92)
#  lands on the FAVOURABLE side and GMI = 1 is dead NEUTRAL, which is the "UNCOMPUTABLE != the
#  clean value" case the 2026-07-26 domain review ruled on.  But SGAI's coefficient is NEGATIVE
#  (-0.172), so a sentinel `SGAI = 0` contributes 0 instead of a neutral -0.172 and pushes M
#  **UP, TOWARD THE FLAG**.
#  THAT IS NOT HYPOTHETICAL AND IT IS THE STRONGEST SINGLE PIECE OF EVIDENCE FOR THIS CHANGE.
#  `000660.KS` (SK hynix), RANK 63 ON THE SHIPPED 2026-08-13 SHORTLIST, reports SG&A = 0 in its
#  newest quarter against revenue of KRW 79.32tn, in a series whose other quarters carry KRW
#  407-547bn.  Measured, guards OFF vs ON:
#      SGAI (newest row)   0.2935  ->  NaN (the row abstains)
#      M_Score (that row)  +1.3385 ->  NaN
#      M_Score_mean        +0.0713 ->  -0.3511
#      forensic tag        `multi-flag: concern`  ->  `single-flag: dig-deeper`
#  So the sentinel MANUFACTURED a manipulation flag on a top-100 name and the guard removes it.
#  The coverage-cost discussion below is framed entirely as flags LOST; at least one of the
#  flags lost was fabricated by the defect this change fixes.
#
#  ---- SO THE INSTRUMENT IS A DOMAIN ON THE PERIOD, APPLIED BEFORE THE SUM ------------------
#  `BENEISH_PERIOD_DOMAIN` refuses the PERIOD, not the base, so `invrollsumTTM`'s rolling sum
#  propagates the NaN and the whole trailing-year window abstains.  That is the same rule the
#  2026-08-14 `ttm_sum` fix settled for the deck ("the newest rpy rows, every one present, else
#  NaN") applied to the forensic layer: a trailing-year base assembled out of periods the
#  vendor did not report is not a trailing-year measurement, whatever its magnitude.
#  The bounds are read off the arithmetic, not fitted:
#      daysSalesOutstanding  (0, inf)   0 = no receivable to age (or 0/0); < 0 impossible.
#      grossProfitMargin     (0, 1)     gross profit cannot exceed revenue (cost of revenue is
#                                       not negative), so >= 1 is either the missing-cost
#                                       sentinel or arithmetically impossible; <= 0 is outside
#                                       the domain of GMI -- see the BLDP counterexample below.
#      SG&A                  (0, inf)   0 = the line is not reported; < 0 is not an expense.
#  and three bases additionally require a POSITIVE base, which is arithmetic and not a floor:
#  SGI (trailing-year sales), SGAI (SG&A intensity), LVGI (leverage share).
#
#  ---- (1) GMI: A MAGNITUDE FLOOR IS THE WRONG INSTRUMENT, AND BLDP PROVES IT ---------------
#  The brief named BLDP as a base "crossing zero" (-1.383 -> +0.000996, GMI = -1,389) and asked
#  whether a floor is the right tool.  It is not, and the reason is stronger than "the base is
#  small".  BLDP's gross margin IMPROVED over that year -- the four quarters average -34.6% in
#  the prior window and +0.02% in the current one -- and the index whose entire job is to detect
#  DETERIORATING margins returned -1,389, which at +0.528 SUBTRACTS 733 points from M, i.e.
#  scores the name as maximally CLEAN.  A floor on |base| would refuse the 2025-10 row and leave
#  the 2025-07 row (-1.474 / -0.297 = GMI +4.96, both legs negative) reading as a 5x margin
#  deterioration when the margin again IMPROVED (-36.9% -> -7.4%).  The defect is not the
#  size of the base, it is that `GM_{t-1}/GM_t` is monotone in margin decline ONLY on the
#  positive domain.  So the instrument is the domain, and it is exact rather than calibrated.
#  Its cost is the largest of the three (10,680 periods refused: 7,548 at <= 0 and 3,132 at
#  >= 1) and its benefit is the largest too: the worst single-row |0.528 x GMI| contribution
#  to one M row falls from 4,903,000 to 37.97.
#
#  ---- (2) DSRI: THE SENTINEL, AND NO FLOOR IN EITHER DIRECTION -----------------------------
#  0HYP.L and CFG are ONE issuer (Citizens Financial, LSE line) and its DSO reads
#  0, 0, 5922.68, 0, 0, 0, 30.5453, 0 across eight quarters -- so the "base" the index divided
#  was a single spike surrounded by sentinels, smeared over four windows by the TTM sum.  The
#  sentinel rule refuses it.  NO magnitude floor is added on top, in either direction:
#    * LOW side: excluding the zeros, the whole first DAY of DSO holds 415 rows spread smoothly
#      (35 below 0.01 days, 95 below 0.1, 164 below 0.25, 415 below 1) against 7,174 at exactly
#      zero.  There is no excess population above zero to cut off, so any floor would be an
#      invented constant.
#    * HIGH side -- and this one was CHECKED because ALHPI.PA survives at DSRI = 9,753.  RAW
#      ROW COUNTS over the 53,477 positive-DSO rows, in days (review M-5: an earlier version
#      quoted these as a per-DECADE DENSITY -- count divided by the band's width in log10 -- with
#      no formula beside them, so the figures summed to 182,453 against a panel of 53,477 and a
#      reader could not check them.  Same measurement, stated as counts):
#          [0,1)      415   [1,10)     3,110   [10,30)    9,648   [30,60)   19,698
#          [60,90) 10,384   [90,120)   3,712   [120,180)  2,590   [180,365)  1,878
#          [365,730)  811   [730,1825)   559   [1825,3650)  257   [3650,1e4)   213
#          [1e4,1e5)  177   [1e5,1e6)     19   [1e6,1e7)      4   [1e7,1e9)      2
#      Monotone decay over four orders of magnitude from the [30,60) mode with NO break and no
#      isolated re-rise -- the signature the 500x input-sanity spike factor was chosen ON is
#      absent here.  A cut anywhere in that tail would be a number chosen to catch the names we
#      already know about.  ALHPI.PA is therefore NAMED as residual below, not floored away.
#
#  ---- (3) SGAI: THE SENTINEL ALREADY CATCHES IT; A FLOOR WOULD BE REDUNDANT ----------------
#  SAP.TO was the brief's near-zero-base exemplar (base 5.2e-5).  It is not a small measurement:
#  the window behind that base is SG&A = 0, 0, 1,000,000, 0 for the four quarters to 2025-01,
#  in a series whose reported quarters carry 4.8e8 to 1.1e9 -- THREE SENTINEL ZEROS AND A CRUMB,
#  over sales of 1.9e10.  The period domain refuses it without a magnitude constant, and the
#  panel offers no evidence for one either: outside the zeros
#  the quarterly SG&A/sales density is 9 rows below 1e-6, 3 in [1e-6,1e-5), 52 in [1e-5,1e-4),
#  355 in [1e-4,1e-3) -- a thin smooth tail, not a pile.
#
#  ---- (4) SGI: NO FLOOR AT ALL, ON PRINCIPLE AND ON EVIDENCE -------------------------------
#  SGI's base is trailing-year sales -- a CURRENCY LEVEL.  There is no unit-free constant to
#  place on it (the panel spans KRW, JPY, USD and EUR filers over ten orders of magnitude), and
#  a SELF-relative test would refuse the very case that must survive: GXAI's revenue genuinely
#  goes 275 -> 6,005,051 over the window (SGI 30.2 / 134.1 / 480.0 / 233.1), a real 21,800x
#  ramp on a pre-revenue name.  The ONLY refusal applied is the arithmetic one -- trailing-year
#  sales <= 0 is not a level to index against -- and GXAI's SGI column comes through the change
#  UNTOUCHED.  (GXAI's M does end up abstaining, but on its GMI and DSRI: five of its ten most
#  recent quarters report grossProfitMargin exactly 1 and three report DSO exactly 0.  That is a
#  different statement from refusing its revenue growth, and the distinction is asserted by a
#  test rather than left to this comment.)
#
#  ---- (5) LVGI: MEASURED, AND IT HAS NO DEGENERATE POPULATION ------------------------------
#  The one place a floor was expected and the density flatly refuses it.  (LTD+CL)/TA in
#  0.005-wide bins from zero: 258, 302, 569 (over 0.01-0.02), 487, 933 (0.03-0.05) -- i.e. a
#  per-0.001 density of 51.6, 60.4, 56.9, 48.7, 46.7 walking INTO zero.  Flat.  Compare AQI,
#  where the same measurement gave 2,164 rows in the first bin against a flat ~740 above 0.010.
#  There is nothing to cut.  LVGI gets the arithmetic refusal only (33 rows, 0.063%: a leverage
#  share cannot be negative and totalAssets is already guarded positive), and NO floor.
#
#  ---- MEASURED EFFECT, same panel, guards ON vs OFF through this same code path -------------
#      panel      2,175 scored / 332 flagged M>0 / 454 abstain  ->  1,436 / 128 / 1,193
#      shortlist     92 scored /   7 flagged     /   8 abstain  ->      79 /   5 /    21
#      |M| > 100     14 -> 1        |M| > 25   38 -> 3        |M| > 10   64 -> 8
#  IN THE ARTIFACT THE CEO READS, `ForensicFlagsTop100`: 14 of the 100 tags change, and the
#  baseline's FIVE `multi-flag: concern` rows -- the strongest label the deck carries -- become
#  two, MAU.PA and 200670.KQ dropping to `data-incomplete` and 000660.KS to `single-flag`.
#  Tag counts 67 clean / 20 single-flag / 5 multi-flag / 8 data-incomplete  ->  59 / 18 / 2 / 21.
#  (An earlier version of this block said 12 of 100 and 2 of 4.  That count came from an OFF arm
#  which left the POSITIVE-BASE guards enabled and so was not the shipped before-state; on the
#  true arm it is 14 and 3 of 5 -- review M-4.  MAU.PA is the row the wrong arm hid.)
#  THE COVERAGE COST IS LARGE AND IS NOT DRESSED UP: 739 names (34%) stop carrying an M.  Two
#  measurements say what that population is.  FIRST, the flags were concentrated in it: 26.8%
#  of the names that lose their M were flagged M > 0, against 9.3% of the names that keep one.
#  Two thirds of the entire Beneish flag population (332 -> 128) was resting on periods the
#  vendor did not report -- and at least one of those flags was MANUFACTURED by a sentinel
#  rather than merely unsupported by it (000660.KS, above).  SECOND, the extremes travel with
#  them: the |M| of the lost names has p90 7.95 and max 1,504, against p90 1.58 and max 161.8
#  for the kept ones.
#  THE COST IS ALSO UNEVEN BY MARKET, which is the part a reader must be told rather than left
#  to discover.  Abstention rate before -> after: .PA 10.7% -> 60.1%, .AS 16.7 -> 52.4,
#  .L 15.5 -> 46.5, .TO 25.4 -> 45.3, US 22.1 -> 43.3, .KQ 0.0 -> 36.8, .KS 1.4 -> 22.1.
#  The unevenness is not created here -- FMP's ratio coverage was always thinnest on Euronext
#  small caps; before this change it was papered over with a number.  A Korean name that
#  abstained 0% of the time was not better covered, it was scoring on sentinels.
#
#  ---- WHAT IS STILL NOT CAUGHT, AND IT IS NOW ONE CLASS RATHER THAN FIVE -------------------
#  Every survivor is a LEVEL that changes scale between adjacent periods without violating any
#  WITHIN-ROW identity, so neither the input-sanity relations nor any base guard can see it:
#    * SSRM (M = -161.8, the ONLY name left above |M| = 100).  Its newest row carries
#      totalAssets 4,279,290 against 5,946,730,000 the quarter before -- but liabilities
#      (887,716) and equity (3,391,570) are rescaled by the same ~1,390x, so L + E = TA holds
#      to within 4 currency units and `impossible_relation_hits` -- which fires at a factor of
#      100 -- sees a perfectly consistent balance sheet.  The isolated-
#      spike rule cannot reach it either: it needs a period on BOTH sides and this is the
#      ENDPOINT row -- the limitation `nan_policy` section 5 already declares.  It is the TATA
#      class (a degenerate totalAssets LEVEL), which is not one of the five indices.
#    * ICU leaves the worst surviving |0.327 x LVGI| at 594: totalAssets steps 106,203,000 ->
#      603,000 between two quarters.  A STEP, not a spike, so the two-sided rule cannot fire.
#    * ALDLS.PA leaves the worst |0.172 x SGAI| at 16,540: revenue reads 105,303,000 then
#      102,875 then 108,994,000.  That IS an isolated spike -- but `revenue` is not in
#      `nan_policy.SCALE_SPIKE_FIELDS`, which lists balance-sheet stocks and SG&A only.
#    * ALHPI.PA leaves the worst |0.92 x DSRI| at 8,973 on revenue of 1 and 3 currency units.
#  ONE INSTRUMENT WOULD REACH ALL FOUR -- a per-source TIME-CONTINUITY test on levels, extended
#  to `revenue` and to the endpoint row.  It is NOT done here for a scope reason that is worth
#  stating: those fields feed STAGE-1 metrics, so refusing them moves scores, while everything
#  in this block is merged after `getAggScore` and moves nothing.  Registered as the follow-up.
#  ALSO NOT DONE, same class, different consumer: `calcMontierC`'s `DSOinc` reads the SAME
#  sentinel-laden `daysSalesOutstanding` column with no domain guard at all.
BENEISH_PERIOD_DOMAIN = {
    'daysSalesOutstanding': (0.0, np.inf),
    'grossProfitMargin': (0.0, 1.0),
    'sellingGeneralAndAdministrativeExpenses': (0.0, np.inf),
}


def _domain_period_input(series, field):
    """A per-PERIOD vendor input with every out-of-domain period replaced by NaN.

    Returns (guarded series, rows refused).  The refusal is at the SUMMAND, not at the
    TTM base, so `invrollsumTTM`'s rolling sum propagates the NaN and the whole window
    abstains -- a trailing-year base built out of periods the vendor did not report is
    not a trailing-year measurement.  NaN in, NaN out, so the count is what THIS rule
    removed and not the column's pre-existing gaps.
    """
    lo, hi = BENEISH_PERIOD_DOMAIN[field]
    v = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)
    refused = v.notna() & ~((v > float(lo)) & (v < float(hi)))
    return v.where(~refused), int(refused.sum())


def _positive_base(base):
    """A base series with every non-positive row replaced by NaN; (series, rows refused)."""
    v = pd.to_numeric(base, errors='coerce').replace([np.inf, -np.inf], np.nan)
    refused = v.notna() & (v <= 0.0)
    return v.where(~refused), int(refused.sum())


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

# =========================================================================== #
#  THE FORENSIC DATA GAP REACHES THE RANKING  (CEO, 2026-08-16)
# =========================================================================== #
#  THIS MOVES BENEISH FROM DISPLAY-ONLY INTO THE SCORE, AND THAT IS INTENDED.  Every earlier
#  comment in this module -- and the O-13 report that preceded this block -- states that the
#  M-score is merged AFTER `getAggScore` and is weighted by nothing.  That is still true of
#  the M-score ITSELF: no verdict, flag or index value enters the score.  What enters is the
#  ABSENCE of one: a name whose Beneish assessment could not be made now carries ad-hoc penalty
#  points and therefore ranks below an otherwise identical name whose assessment was made.
#  A future reader who finds "display-only" above and this below is not looking at a
#  contradiction -- the two statements are about different things, and this one is the newer.
#
#  THE RULING.  CEO, 2026-08-16, on the 739 names the O-13 domain guards convert to an
#  abstention: *"We should not be rewarding lack of data. What we want is to have the top 20 be
#  INFORMED good, rather than perhaps good but we don't know because we lack data."*  And on
#  the instrument: *"we should probably punish it slightly. Straight to the ad-hoc penalty
#  bucket. But have it very slight."*
#
#  WHY THE CHARGE IS CORRECT UNDER BOTH READINGS OF THE VENDOR'S ZERO -- the point the CEO
#  asked about directly and the reason this does not depend on resolving it.  O-13 established
#  that an exact zero in these inputs is a SENTINEL on the panel-wide evidence (4,190 of the
#  4,263 zero gross margins sit on a zero-revenue row, i.e. 0/0; only 2 of 323 sources with a
#  `margin == 1` row report it in every period).  It does NOT resolve PER NAME: a cash-only
#  retailer can genuinely have no receivables, and a shell can genuinely report no SG&A.
#  It does not have to.  If the value is MISSING we lack the data; if it is LITERALLY ZERO the
#  index is 0/0 or a division by zero and there is still no usable measurement.  Either way the
#  honest statement is "we could not assess this company", and that must not score the same as
#  "we assessed it and it was clean".  So the penalty rests on the ambiguity, not on its
#  resolution.
#
#  THE AMOUNT, AND WHY IT IS SHAPED THIS WAY.  `adhoc_penalty` fixes the weight at 0.01 by
#  ruling and puts the scaling in the AMOUNT, so the amount is a COUNT of distinct data-gap
#  events, exactly as `stage1_veto` counts missing rows:
#      forensic_gap:beneish_components   min(components absent from the whole M window, 3)
#      forensic_gap:no_verdict           1, when the name ends with no M_Score_mean at all
#  Maximum 4 points = -0.04 AggScore.  Against the two scales that exist: it is 57% of the
#  largest charge ONE Stage-1 veto flag can raise (7 points = -0.07, refused rows on all but one
#  row of its 8-row window) and exactly half the worst per-SOURCE veto total measured on the
#  2026-08-13 run (-0.08, across several checks); and it is 10.8% of that run's top-100
#  median-to-rank-20 distance (0.3709).  That is the "very slight" the ruling asked for,
#  measured against the score geometry rather than asserted.
#  AN EARLIER VERSION OF THIS LINE SAID "HALF the largest bucket a Stage-1 veto flag can raise
#  (-0.08)", conflating the veto's single-FLAG maximum (-0.07) with the worst per-SOURCE total
#  observed on one run (-0.08).  Writing the assertion that checks this against
#  `stage1_veto.WINDOW_ROWS` is what exposed it; the test now pins the property that survives --
#  a missing forensic assessment must charge STRICTLY LESS than one persistent solvency flag.
#  THE CAP AT THREE IS A DECISION, not a rounding.  Beyond three absent components the
#  statement has already saturated -- "we could not assess this company" -- and the
#  `no_verdict` point is what says it.  Panel distribution of components absent (of 8):
#  0 -> 1,520 names, 1 -> 626, 2 -> 232, 3 -> 89, 4 -> 80, 5 -> 62, 6 -> 18, 7 -> 2, so the cap
#  binds on 162 names (6.2%) and leaves the graded 1/2/3 range doing the work the CEO asked for
#  ("one absent input is a different statement from three").
#  THE `no_verdict` POINT IS NOT A DOUBLE CHARGE FOR THE SAME FACT.  Measured: a component
#  absent across the whole window ALWAYS costs the verdict (0 names carry a verdict with
#  n_absent > 0), so for those names it is a second point on one cause -- but it is the ONLY
#  charge that reaches the 84 names whose components are all computable somewhere and which
#  still abstain on `nan_policy.COVERAGE_MIN`.  Without it that class -- an abstention with no
#  absent component -- would be free.
#
#  DOUBLE-CHARGING WITH THE STAGE-1 VETO BUCKET IS DELIBERATE, AND MEASURED.  200 of the 2,629
#  panel sources carry both a veto contribution and a forensic one; 4 of the shipped top-100 do
#  (WSE -0.10 combined, PSI.TO -0.08, 035420.KS -0.04, 200670.KQ -0.03).  They are DIFFERENT
#  assessments -- the veto could not run the solvency flags, this could not run the manipulation
#  model -- and a name missing both is worse informed than a name missing one, which is the
#  scaling the CEO's own ruling on this bucket puts in the amount.  The worst combined charge
#  (-0.10) is 27% of the median-to-rank-20 distance: still a nudge, never a removal.
#
#  WHAT IT ACTUALLY MOVES, and the honest reading of the one change it makes.  `head(100)` runs
#  BEFORE Stage-2, so this penalty CANNOT change top-100 membership -- it reorders within the
#  100 (verified: the name SET is identical) and therefore can change the top-20 cut.  On the
#  shipped 2026-08-13 top-100: 21 of 100 names charged, total -0.4600, worst -0.03; 47 of 100
#  change position, largest move 9 places; the TOP-20 changes by ONE name, CFX.L out, GMR.L in,
#  and every other top-20 move is one or two places.  Panel-wide (2,629 sources) 1,193 carry the
#  gap: 84 at 1 point (the coverage-only abstentions), 626 at 2, 232 at 3, 251 at 4.
#  THE SHIPPED FOOTPRINT IS 35 NAMES, NOT 21 (review L-6): the five carve-out side-lists ship
#  too, and 14 of their names are charged after the forensic-validity gate below (71 before it).
#  The general top-100's 21 are unchanged by that gate -- none of them was ever exempt.
#  THE FLIP IS A NEAR-TIE: CFX.L and GMR.L are 0.002301 AggScore apart -- 0.2301 POINTS at the
#  fixed weight of 0.01, under a quarter of the smallest whole point this bucket can charge.
#  THE FRONTIER, NOT A UNIVERSAL (review H-1, and the correction matters because the earlier
#  claim reached the CEO).  This block previously said "there is no 'very slight' value that
#  both charges CFX.L and leaves the top 20 alone; only charging nothing can".  THAT IS FALSE AS
#  STATED.  Every rule I had tested lives in {1,2,3,4} points, i.e. at least 4.3x the tie, so the
#  search could not have found a counterexample -- and the reviewer measured two:
#      shipped shape x 0.10   max -0.0040   21 charged   top-20 UNCHANGED   7 positions move
#      flat 0.23 point        max -0.0023   21 charged   top-20 UNCHANGED   9 positions move
#  The true statement is the arithmetic one: ANY per-name charge below 0.2301 points leaves the
#  cut alone, and no INTEGER point rule at weight 0.01 is that small.  So the choice is a real
#  one and it belongs to the CEO, not to this file.
#  WHY THE INTEGER RULE IS STILL WHAT SHIPS, stated as a recommendation rather than a necessity:
#  at 0.1x the maximum charge is 1.1% of the median-to-rank-20 distance and the bucket is close
#  to inert, which is the shape the ruling was issued against ("we should not be rewarding lack
#  of data"); and the flip itself is the ruling working -- CFX.L abstains, GMR.L is assessed and
#  uncharged, so the tiebreak goes to the name we could assess.
BENEISH_GAP_MAX_COMPONENT_POINTS = 3
CHECK_GAP_COMPONENTS = 'forensic_gap:beneish_components'
CHECK_GAP_NO_VERDICT = 'forensic_gap:no_verdict'

#  Component -> what a reader who does not know Beneish needs to be told it means.  Used for
#  the abstention REASON in `ForensicFlagsTop*.csv` and in the deck (CEO: an abstention must say
#  WHY -- "no margin data from the vendor" reads differently from an empty cell, and a blank
#  rate this high reads as a broken tool unless it explains itself).
_COMPONENT_ENGLISH = {
    'DSRI': 'receivables days (DSRI)',
    'GMI': 'gross margin (GMI)',
    'AQI': 'asset quality (AQI)',
    'SGI': 'sales growth (SGI)',
    'DEPI': 'depreciation rate (DEPI)',
    'SGAI': 'SG&A intensity (SGAI)',
    'LVGI': 'leverage (LVGI)',
    'TATA': 'accruals (TATA)',
}
M_COMPONENTS = ('DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA')


def absent_components(mscore_df, symbol, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """The Beneish components with NO computable value anywhere in `symbol`'s M window.

    The window is the one `M_Score_mean` averages -- `head(scale_window(4, rpy))` of the
    already-trimmed per-symbol rows -- so this counts what the SCORE was missing, not what the
    whole history was missing.  A source absent from `mscore_df` returns every component: it
    produced no forensic rows at all, which is the largest gap there is, not the smallest.
    """
    if mscore_df is None or len(mscore_df) == 0 or 'symbol' not in mscore_df.columns:
        return list(M_COMPONENTS)
    sub = mscore_df[mscore_df['symbol'] == symbol]
    if sub.empty:
        return list(M_COMPONENTS)
    win = sub.head(rp.scale_window(4, int(rpy)))
    return [c for c in M_COMPONENTS
            if c not in win.columns
            or not pd.to_numeric(win[c], errors='coerce').notna().any()]


def abstention_reason(absent, has_verdict, n_computable=None, n_window=None):
    """Plain-English WHY for a name with no M-score -- '' when it has one.

    THE CSV AND THE DECK BOTH READ THIS ONE FUNCTION so the shipped artifacts cannot disagree
    about why a name abstained, and so the wording is changed in one place.
    """
    if has_verdict:
        return ''
    if absent:
        return ('no usable vendor data for: %s'
                % ', '.join(_COMPONENT_ENGLISH.get(c, c) for c in absent))
    if n_window:
        return ('fewer than half the trailing-year periods are computable (%s of %s), so the '
                'mean would not be a trailing-year mean'
                % (n_computable if n_computable is not None else '?', n_window))
    return 'the trailing-year window could not be formed'


def forensic_gap_points(absent, has_verdict,
                        max_component_points=None):
    """(points, per-check contributions) for ONE name.  Points are the COUNT, never a rate.

    Returns `(total, [(check, points), ...])` so the caller can itemise each check separately
    into the evidence CSV -- a single blended number would make the bucket unarguable, which is
    the property `adhoc_penalty` exists to protect.
    """
    cap = (BENEISH_GAP_MAX_COMPONENT_POINTS if max_component_points is None
           else max_component_points)
    items = []
    n = min(len(absent), int(cap))
    if n > 0:
        items.append((CHECK_GAP_COMPONENTS, float(n)))
    if not has_verdict:
        items.append((CHECK_GAP_NO_VERDICT, 1.0))
    return float(sum(p for _c, p in items)), items


def contribute_forensic_gap_points(cdx_df, sources, penalty_book, freq_map=None, pool=None,
                                   verbose=True, sector_map=None):
    """Charge the ad-hoc bucket for every name whose Beneish assessment could not be made.

    Returns the per-source gap frame (source, n_absent, absent, has_verdict, points, reason)
    for reporting; charges nothing and returns an empty frame when `penalty_book` is None.

    IT CALLS `calcBeneishM` RATHER THAN RE-DERIVING THE GAP.  A second implementation of "which
    components are computable" is the classic way a penalty and the artifact it is supposed to
    explain drift apart -- the CSV would say the name has a gross margin while the bucket
    charged it for not having one.  The cost is that the pool's M is computed twice per run
    (~25s per 2,600 sources, against a 12-hour fetch), which is the cheap side of that trade.

    RAISED ONCE PER RUN, NOT ONCE PER POOL.  `PenaltyBook.penalty_series` sums a source's
    contributions across every pool tag, so raising the same finding for the general pool and
    again for a carve-out cohort would charge a name in both TWICE for one gap.  The carve IS a
    strict partition today -- measured on the shipped run, ZERO names sit in both the general
    pool and any cohort (review L-5, correcting the original justification, which cited an
    overlap that does not occur) -- so `pool=None` buys nothing on this panel.  It is kept
    because it is right if the partition ever breaks, and because it matches
    `penalty_series` being pool-agnostic.

    A NAME THE HOUSE RULES FORENSICALLY INVALID IS NOT CHARGED (review H-2, 2026-08-17).
    `forensicFlags._classify_financial` already declares Beneish structurally undefined for
    banks, insurers and REITs, and the forensic tag says so instead of scoring them.  Charging
    a REIT for not having a Beneish score is charging it for being a REIT, and the reason
    string written against it ("the manipulation assessment is ABSENT -- not clean") would be
    false: the assessment is INAPPLICABLE, which is a different fact.  Measured before the gate
    on the five shipped cohort side-lists: 71 charged, 57 of them forensically invalid -- REIT
    16 of 17, InvestmentVehicle 16 of 20, FinManager 13 of 16, BalanceSheetFin 12 of 13, Mining
    0 of 5.  The general top-100 was already clean (0 of its 21).
    THE CLASSIFIER IS THE SAME ONE AND THE MAP IS THE SAME MAP, deliberately: the charge and
    the tag the CEO reads must agree by construction rather than by coincidence.  The API
    sector fallback is fetched AFTER Stage-2 and so is not available here, which makes this
    gate the PICKLE-sector half of the two-source rule -- and therefore possibly narrower than
    the final tag, never wider (the fallback only ever ADDS financials).  On the shipped run
    the two agree exactly: the pickle-only classification exempts all 57.
    """
    cols = ['source', 'n_absent', 'absent', 'has_verdict', 'points', 'reason',
            'forensically_valid']
    if penalty_book is None or cdx_df is None or not len(cdx_df):
        return pd.DataFrame(columns=cols)
    symblist = [s for s in dict.fromkeys(sources)]
    if freq_map is None:
        freq_map = rp.frequency_by_source(cdx_df)
    #  IMPORTED LAZILY, and the circularity is the reason: `forensicFlags` imports this module
    #  at its top level, so a module-level import here would be a cycle.  By the time this runs
    #  both modules are loaded, and the alternative -- passing the classifier in from `postBo`
    #  -- would put the "which names is Beneish undefined for" policy in a third place.
    from forensicFlags import _classify_financial, _load_sector_map
    if sector_map is None:
        sector_map = _load_sector_map()
    mscore_df, SLmeanMscore, _problems = calcBeneishM(
        {'cdx_df': cdx_df}, symblist, freq_map, verbose=False)
    means = dict(zip(SLmeanMscore['source'],
                     pd.to_numeric(SLmeanMscore['M_Score_mean'], errors='coerce')))
    rows = []
    #  BUFFERED, COMMITTED TO THE BOOK ONLY AFTER THE LOOP COMPLETES (review L-1).  The caller
    #  wraps this in one `except` whose message says the gap "cost nothing"; if the loop raised
    #  partway, names 1..k-1 would ALREADY be in the book and would ship in the evidence CSV
    #  underneath that caveat -- a false statement in the one file that exists to be argued
    #  with.  All-or-nothing makes the caveat true.
    pending = []
    n_exempt = 0
    for symbol in symblist:
        _rpy = rp.rows_per_year(freq_map, symbol)
        m = means.get(symbol, np.nan)
        has_verdict = bool(pd.notna(m) and np.isfinite(m))
        absent = absent_components(mscore_df, symbol, _rpy)
        total, items = forensic_gap_points(absent, has_verdict)
        sub = mscore_df[mscore_df['symbol'] == symbol] if len(mscore_df) else mscore_df
        _win = (pd.to_numeric(sub['M_Score'], errors='coerce')
                .head(rp.scale_window(4, _rpy)) if len(sub) else pd.Series(dtype='float64'))
        reason = abstention_reason(absent, has_verdict, int(_win.notna().sum()), len(_win))
        _valid = not _classify_financial(sector_map.get(symbol, 'Unknown'), None)[0]
        if not _valid and total > 0:
            n_exempt += 1
            total, items = 0.0, []
        rows.append({'source': symbol, 'n_absent': len(absent),
                     'absent': ','.join(absent), 'has_verdict': has_verdict,
                     'points': total, 'reason': reason, 'forensically_valid': _valid})
        for check, pts in items:
            #  BOTH CHECKS CARRY THE SPECIFIC REASON, not just the component one.  A test
            #  caught the first version doing otherwise: the COVERAGE-ONLY abstention raises
            #  `no_verdict` ALONE, so its generic sentence was the only thing in the evidence
            #  CSV for that class -- a charge a reader could not argue with, which is the one
            #  property `adhoc_penalty` says every contribution must have.
            pending.append((symbol, check,
                            (reason if check == CHECK_GAP_COMPONENTS else
                             'the Beneish M-score could not be formed for this name, so the '
                             'manipulation assessment is ABSENT -- not clean: %s' % reason),
                            pts))
    for symbol, check, why, pts in pending:
        penalty_book.add(symbol, check, why, pts, pool=pool)
    #  THE EXEMPTION IS DECLARED, NOT SILENT (review H-2).  A name that is not charged and not
    #  named reads as a name with no data gap; these have one and it does not count against
    #  them.  Zero points, so it rides the CSV's NOT_MEASURED section beside the veto's.
    if n_exempt:
        penalty_book.declare_unmeasured(
            CHECK_GAP_COMPONENTS,
            'EXEMPT: %d name(s) had an absent Beneish assessment that was NOT charged because '
            '`forensicFlags._classify_financial` rules the model inapplicable to them (banks, '
            'insurers, REITs). For those names the honest statement is "the manipulation model '
            'does not apply", not "the assessment is absent". Pickle-sector basis: the API '
            'sector fallback is fetched after Stage-2 and can only ADD financials, so this '
            'exemption is never wider than the tag the shortlist ships.' % n_exempt,
            pool=pool)
    out = pd.DataFrame(rows, columns=cols)
    if verbose:
        _charged = out[out['points'] > 0]
        print('FORENSIC DATA-GAP PENALTY: %d of %d name(s) charged (%.2f points, worst %.2f '
              '= %.4f AggScore) because their Beneish assessment could not be made. THIS IS '
              'THE ONE PLACE THE FORENSIC LAYER ENTERS THE SCORE -- the M-score itself is '
              'still display-only; what is charged is its ABSENCE (CEO, 2026-08-16: "we should '
              'not be rewarding lack of data"). %d name(s) with the same gap were EXEMPT '
              'because the model is inapplicable to them (bank / insurer / REIT).'
              % (len(_charged), len(out), float(out['points'].sum()),
                 float(out['points'].max()) if len(out) else 0.0,
                 -0.01 * float(out['points'].max()) if len(out) else 0.0, n_exempt),
              flush=True)
    return out


def calcBeneishM(resdic, symblist, freq_map=None, verbose=True):
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
    #  O-13 guards on the other five indices: out-of-domain PERIODS (refused before the TTM
    #  sum) and non-positive BASES (refused after it).  Counted the same way and for the same
    #  reason -- an invisible refusal reads as a name that never had the data.
    n_period = {'daysSalesOutstanding': 0, 'grossProfitMargin': 0,
                'sellingGeneralAndAdministrativeExpenses': 0}
    n_base = {'SGI': 0, 'SGAI': 0, 'LVGI': 0}
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
        #  DOMAIN, PER PERIOD (O-13).  A DSO of zero is not a collection period -- see the
        #  BENEISH_PERIOD_DOMAIN block.  Refused BEFORE the TTM sum, so a window holding one
        #  such period abstains whole instead of summing a non-measurement into the base.
        _dso, _np = _domain_period_input(tempcdx_df['daysSalesOutstanding'],
                                         'daysSalesOutstanding')
        n_period['daysSalesOutstanding'] += _np
        dsoTTM = invrollsumTTM(_dso, _rpy)
        tmpmdf['DSRI'] = _yoyCurOverPrior(dsoTTM, _rpy)     # DSO_t / DSO_{t-1}

        #  DOMAIN, PER PERIOD (O-13).  GMI = GM_{t-1}/GM_t only reads as "margins declined"
        #  while both legs are positive fractions; outside (0,1) the index inverts or is
        #  arithmetically impossible.  See the BENEISH_PERIOD_DOMAIN block for BLDP.
        _gm, _np = _domain_period_input(tempcdx_df['grossProfitMargin'], 'grossProfitMargin')
        n_period['grossProfitMargin'] += _np
        gmiTTM = invrollsumTTM(_gm, _rpy)
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

        #  DOMAIN, ON THE BASE (O-13).  SGI's base is a CURRENCY LEVEL, so no unit-free floor
        #  exists for it and none is applied -- a 21,800x real revenue increase must still
        #  score (GXAI).  What IS refusable is the arithmetic domain: trailing-year sales of
        #  zero or less is not a sales level to index against.  See BENEISH_PERIOD_DOMAIN.
        sgiTTM, _nb = _positive_base(invrollsumTTM(tempcdx_df['revenue'], _rpy))
        n_base['SGI'] += _nb
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

        #  DOMAIN, PER PERIOD AND ON THE BASE (O-13).  A reported SG&A of exactly zero is the
        #  line not being reported, not a company that spends nothing on selling and admin;
        #  and SGA/Sales <= 0 is not an expense intensity.
        _sga, _np = _domain_period_input(
            tempcdx_df['sellingGeneralAndAdministrativeExpenses'],
            'sellingGeneralAndAdministrativeExpenses')
        n_period['sellingGeneralAndAdministrativeExpenses'] += _np
        sgaTTM = invrollsumTTM(_sga, _rpy)
        #  STRUCTURALLY UNREACHABLE TODAY, AND KEPT ANYWAY (review L-3).  `sgaTTM` sums periods
        #  already restricted to (0, inf) and `sgiTTM` is already forced positive, so this ratio
        #  is positive-or-NaN by construction and the counter reads 0 on every run.  It is
        #  defence-in-depth against a future widening of the SG&A period domain, not a live
        #  guard -- the run log says so rather than advertising a refusal that cannot happen.
        sgaiTTM, _nb = _positive_base(sgaTTM/sgiTTM)
        n_base['SGAI'] += _nb
        tmpmdf['SGAI'] = _yoyCurOverPrior(sgaiTTM, _rpy)    # (SGA/Sales)_t / (SGA/Sales)_{t-1}

        ltdTTM = invrollsumTTM(tempcdx_df['longTermDebt'], _rpy)
        clTTM = invrollsumTTM(tempcdx_df['totalCurrentLiabilities'], _rpy)
        #  DOMAIN, ON THE BASE (O-13), AND DELIBERATELY NO FLOOR.  Liabilities cannot be
        #  negative and totalAssets is already guarded positive, so a leverage share <= 0 is
        #  impossible; but the panel shows NO degenerate population above zero for this base,
        #  so no magnitude floor is imposed.  See BENEISH_PERIOD_DOMAIN for the density.
        lvgiTTM, _nb = _positive_base((ltdTTM+clTTM)/taTTM)
        n_base['LVGI'] += _nb
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
    #  `verbose=False` ONLY for the second, internal call this run makes -- the forensic-gap
    #  penalty computes the same M over the pre-head(100) pool -- so the run log carries ONE
    #  set of guard counts, the one describing the names the CEO is shown.  It is not a way to
    #  turn the disclosure off: `detectManipulationWrapper` never passes it.
    if not verbose:
        return mdf, SLmeanMscore, problemlist
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
        #  SECOND LINE, NOT FOLDED INTO THE FIRST.  These refusals are a DIFFERENT instrument
        #  from the AQI/DEPI floors -- they act on the PERIOD before the trailing-year sum, and
        #  on the arithmetic domain of a base rather than on its magnitude -- and an operator
        #  who cannot tell them apart cannot argue with either.
        print('BENEISH M-SCORE (O-13 domain guards): refused %d DSO period(s) (not in %s), %d '
              'gross-margin period(s) (not in %s) and %d SG&A period(s) (not in %s) BEFORE the '
              'trailing-year sum, so a window holding one abstains whole; and %d SGI, %d SGAI '
              '(structurally unreachable while the SG&A period domain holds -- defence-in-depth, '
              'not a live guard), %d LVGI base(s) as non-positive. NO magnitude floor is '
              'applied to any of the '
              'five: SGI is a currency level, and DSRI/LVGI show no degenerate population '
              'above zero on this panel.'
              % (n_period['daysSalesOutstanding'],
                 BENEISH_PERIOD_DOMAIN['daysSalesOutstanding'],
                 n_period['grossProfitMargin'], BENEISH_PERIOD_DOMAIN['grossProfitMargin'],
                 n_period['sellingGeneralAndAdministrativeExpenses'],
                 BENEISH_PERIOD_DOMAIN['sellingGeneralAndAdministrativeExpenses'],
                 n_base['SGI'], n_base['SGAI'], n_base['LVGI']), flush=True)
    except Exception as _e:
        print('WARNING: Beneish M-score summary skipped (%s)' % _e, flush=True)

    return mdf, SLmeanMscore, problemlist

def invrollsumTTM(Svec, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """Trailing-twelve-month sum on NEWEST-FIRST data: `rpy` rows, not a fixed 4.

    A semi-annual filer's two rows already cover 12 months; summing four of them built a
    TWENTY-FOUR-month "TTM" and made every ratio built on it (Beneish TATA/SGI/AQI/LVGI,
    the Montier flags, Sloan accruals) ~2x its peers."""
    return (Svec.iloc[::-1].rolling(int(rpy)).sum()).iloc[::-1]
