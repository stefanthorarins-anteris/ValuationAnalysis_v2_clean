import pandas as pd
import numpy as np
import warnings

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
        tmpmdf['AQI'] = _yoyCurOverPrior(aqiTTM, _rpy)      # AQI_t / AQI_{t-1}

        sgiTTM = invrollsumTTM(tempcdx_df['revenue'], _rpy)
        tmpmdf['SGI'] = _yoyCurOverPrior(sgiTTM, _rpy)      # Sales_t / Sales_{t-1}

        #z = x / (x + y) = 1 / ((x + y) / (x)) = 1 / (1 + (y / x))
        # Ef w = y / x, þá: z = 1 / (1 + w). x = depreciationAndAmortization, y = PP&Enet
        ddaTTM = invrollsumTTM(tempcdx_df['depreciationAndAmortization'], _rpy)
        w = ppenTTM/ddaTTM
        depiTTM = 1/(1+w)                                   # depreciation rate = Dep/(Dep+PPE)
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
        mscore = symb_mscore['M_Score'].head(rp.scale_window(4, _rpy)).mean()
        SLmeanMscore.loc[SLmeanMscore['source']==symbol, 'M_Score_mean'] = mscore

        if np.isnan(mscore) or np.isinf(mscore):
            problemlist.append(symbol)
        elif mscore > 0:
            problemlist.append(symbol)

    return mdf, SLmeanMscore, problemlist

def invrollsumTTM(Svec, rpy=rp.DEFAULT_ROWS_PER_YEAR):
    """Trailing-twelve-month sum on NEWEST-FIRST data: `rpy` rows, not a fixed 4.

    A semi-annual filer's two rows already cover 12 months; summing four of them built a
    TWENTY-FOUR-month "TTM" and made every ratio built on it (Beneish TATA/SGI/AQI/LVGI,
    the Montier flags, Sloan accruals) ~2x its peers."""
    return (Svec.iloc[::-1].rolling(int(rpy)).sum()).iloc[::-1]
