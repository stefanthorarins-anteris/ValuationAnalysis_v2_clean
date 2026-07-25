import pandas as pd
import numpy as np
import warnings

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
C_FLAG_CUTOFF = 4        # surface for review when C_Score_mean >= 4

# The six Montier red-flag columns, in the order they are counted.
C_FLAG_COLS = ['NICFOdiv', 'DSOinc', 'DSIinc', 'OCARinc', 'DAPPdec', 'TAgr']

def detectManipulationWrapper(resdic):
    symblist = list(resdic['postRank']['source'])
    mscore_df, SLmeanMscore, problemlist_Mscore = calcBeneishM(resdic,symblist)

    cscore_df, SLmeanCscore, problemlist_Cscore = calcMontierC(resdic, symblist)

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


def _yoyCurOverPrior(ttm):
    """current / prior-year, on NEWEST-FIRST data.

    Row k is the current quarter; row k+4 is the SAME quarter one year earlier
    (4 rows older). Beneish DSRI/AQI/SGI/SGAI/LVGI are all defined current/prior."""
    return ttm / ttm.shift(-4)


def _yoyPriorOverCur(ttm):
    """prior-year / current, on NEWEST-FIRST data.

    Beneish GMI and DEPI are defined prior/current (a DECLINE in gross margin or
    in the depreciation rate pushes the index ABOVE 1.0, the suspicious side)."""
    return ttm.shift(-4) / ttm


def calcMontierC(resdic,symblist):
    cdx_df = resdic['cdx_df']
    SLmeanCscore = pd.DataFrame(columns=['source', 'C_Score_mean'])
    SLmeanCscore['source'] = symblist
    cdf = pd.DataFrame(columns=['date', 'symbol', 'NICFOdiv','DSOinc','DSIinc','OCARinc','DAPPdec','TAgr','C_Score'])
    problemlist = []
    for symbol in symblist:
        tmpcdf = pd.DataFrame(columns=['date', 'symbol', 'NICFOdiv','DSOinc','DSIinc','OCARinc','DAPPdec','TAgr','C_Score'])
        # C-score if NICFO > 0, DSOinc > 0 ...
        # Newest-first so diff(-4)/pct_change(-4) read current-minus-prior (a YoY
        # INCREASE is the suspicious side for every Montier flag) and head(...)
        # summarizes the MOST RECENT quarters.
        tempcdx_df = _toNewestFirst(cdx_df[cdx_df['source'] == symbol])

        # MISSING DATA MUST NOT FIRE A RED FLAG (audit H7 fix, 2026-07-19).
        # Every flag below used to end in `.fillna(99999)`, i.e. a period whose YoY
        # change could not be COMPUTED was scored as a maximal INCREASE -- the
        # suspicious side -- so absent data manufactured forensic red flags.  It was
        # also inconsistent: DAPPdec negates AFTER the fill (-99999), so for that one
        # flag missing data could never fire while for the other five it always did.
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
        cfoTTM = invrollsumTTM(tempcdx_df['netCashProvidedByOperatingActivities'])
        niTTM = invrollsumTTM(tempcdx_df['netIncome'])
        NICFO = (niTTM - cfoTTM)/cfoTTM.abs()
        tmpcdf['NICFOdiv'] = NICFO.diff(periods=-4)

        dsoTTM = invrollsumTTM(tempcdx_df['daysSalesOutstanding'])
        tmpcdf['DSOinc'] = dsoTTM.diff(periods=-4)

        dsiTTM = invrollsumTTM(tempcdx_df['daysOfInventoryOutstanding'])
        tmpcdf['DSIinc'] = dsiTTM.diff(periods=-4)

        ocarTTM = invrollsumTTM(tempcdx_df['otherCurrentAssets'])/invrollsumTTM(tempcdx_df['revenue'])
        tmpcdf['OCARinc'] = ocarTTM.diff(periods=-4)

        ddaTTM = invrollsumTTM(tempcdx_df['depreciationAndAmortization'])
        capex = tempcdx_df['capexPerShare']*tempcdx_df['weightedAverageShsOut']
        nppeTTM = invrollsumTTM(tempcdx_df['propertyPlantEquipmentNet'])
        adTTM = ddaTTM.iloc[::-1].cumsum().iloc[::-1]
        gppeTTM = nppeTTM - capex + adTTM
        dappTTM = ddaTTM/gppeTTM
        tmpcdf['DAPPdec'] = -dappTTM.diff(periods=-4)

        taTTM = invrollsumTTM(tempcdx_df['totalAssets'])
        tmpcdf['TAgr'] = taTTM.pct_change(-4, fill_method=None) - 0.1

        # Count over the SIX flag columns explicitly (not `tmpcdf > 0` over every
        # column) so the score can never pick up a stray column; NaN > 0 is False, so a
        # non-computable flag simply does not fire.
        tmpcdf['C_Score'] = (tmpcdf[C_FLAG_COLS].apply(pd.to_numeric, errors='coerce')
                             > 0).sum(axis=1)

        tmpcdf['date'] = tempcdx_df['date']
        tmpcdf['symbol'] = tempcdx_df['source']

        symb_cscore = tmpcdf[0:len(tmpcdf)-4]

        cdf = pd.concat([cdf, symb_cscore])
        # head(2): newest-first, so the 2 MOST RECENT quarters (current condition).
        cscore = symb_cscore['C_Score'].head(2).mean()
        SLmeanCscore.loc[SLmeanCscore['source']==symbol, 'C_Score_mean'] = cscore

        if np.isnan(cscore) or np.isinf(cscore):
            problemlist.append(symbol)
        elif cscore >= C_FLAG_CUTOFF:      # ONE cutoff (was a stricter `> 4` here)
            problemlist.append(symbol)

    return cdf, SLmeanCscore, problemlist

def calcBeneishM(resdic,symblist):
    cdx_df = resdic['cdx_df']
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
        dsoTTM = invrollsumTTM(tempcdx_df['daysSalesOutstanding'])
        salesTTM = invrollsumTTM(tempcdx_df['revenue'])
        tmpmdf['DSRI'] = _yoyCurOverPrior(dsoTTM)           # DSO_t / DSO_{t-1}

        gmiTTM = invrollsumTTM(tempcdx_df['grossProfitMargin'])
        tmpmdf['GMI'] = _yoyPriorOverCur(gmiTTM)            # GM_{t-1} / GM_t  (margin decline -> >1)

        tcaTTM = invrollsumTTM(tempcdx_df['totalCurrentAssets'])
        ppenTTM = invrollsumTTM(tempcdx_df['propertyPlantEquipmentNet'])
        taTTM = invrollsumTTM(tempcdx_df['totalAssets'])
        aqiTTM = 1- (tcaTTM + ppenTTM)/taTTM
        tmpmdf['AQI'] = _yoyCurOverPrior(aqiTTM)            # AQI_t / AQI_{t-1}

        sgiTTM = invrollsumTTM(tempcdx_df['revenue'])
        tmpmdf['SGI'] = _yoyCurOverPrior(sgiTTM)            # Sales_t / Sales_{t-1}

        #z = x / (x + y) = 1 / ((x + y) / (x)) = 1 / (1 + (y / x))
        # Ef w = y / x, þá: z = 1 / (1 + w). x = depreciationAndAmortization, y = PP&Enet
        ddaTTM = invrollsumTTM(tempcdx_df['depreciationAndAmortization'])
        w = ppenTTM/ddaTTM
        depiTTM = 1/(1+w)                                   # depreciation rate = Dep/(Dep+PPE)
        tmpmdf['DEPI'] = _yoyPriorOverCur(depiTTM)          # rate_{t-1} / rate_t  (rate decline -> >1), full 4Q YoY

        sgaTTM = invrollsumTTM(tempcdx_df['sellingGeneralAndAdministrativeExpenses'])
        sgaiTTM = sgaTTM/sgiTTM
        tmpmdf['SGAI'] = _yoyCurOverPrior(sgaiTTM)          # (SGA/Sales)_t / (SGA/Sales)_{t-1}

        ltdTTM = invrollsumTTM(tempcdx_df['longTermDebt'])
        clTTM = invrollsumTTM(tempcdx_df['totalCurrentLiabilities'])
        lvgiTTM = (ltdTTM+clTTM)/taTTM
        tmpmdf['LVGI'] = _yoyCurOverPrior(lvgiTTM)          # Leverage_t / Leverage_{t-1}

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
        niTTM = invrollsumTTM(tempcdx_df['netIncome'])
        cfoTTM = invrollsumTTM(tempcdx_df['netCashProvidedByOperatingActivities'])
        # totalAssets <= 0 is impossible (614 such rows exist in the panel, audit M-3);
        # as a LEVEL denominator it would produce +-inf rather than the ~harmless small
        # number a 4-quarter sum used to hide, so it is coerced to NaN = "not computable".
        taLevel = pd.to_numeric(tempcdx_df['totalAssets'], errors='coerce')
        taLevel = taLevel.where(taLevel > 0)
        tmpmdf['TATA'] = (niTTM - cfoTTM)/taLevel

        # +1.78 folds the -1.78 manipulator cutoff into the stored score, so the
        # stored M>0 is exactly the standard Beneish M>-1.78. Preserved verbatim;
        # it now folds correctly-directed components.
        tmpmdf['M_Score'] = - 4.84 + 0.92*tmpmdf.DSRI + 0.528*tmpmdf.GMI + 0.404*tmpmdf.AQI + 0.892*tmpmdf.SGI +\
                            0.115*tmpmdf.DEPI - 0.172*tmpmdf.SGAI + 4.679*tmpmdf.TATA - 0.327*tmpmdf.LVGI + 1.78
        tmpmdf['date'] = tempcdx_df['date']
        tmpmdf['symbol'] = tempcdx_df['source']
        symb_mscore = tmpmdf[0:len(tmpmdf)-4]

        mdf = pd.concat([mdf,symb_mscore])
        # head(4): newest-first, so the 4 MOST RECENT quarters (current condition).
        mscore = symb_mscore['M_Score'].head(4).mean()
        SLmeanMscore.loc[SLmeanMscore['source']==symbol, 'M_Score_mean'] = mscore

        if np.isnan(mscore) or np.isinf(mscore):
            problemlist.append(symbol)
        elif mscore > 0:
            problemlist.append(symbol)

    return mdf, SLmeanMscore, problemlist

def invrollsumTTM(Svec):
    irsTTM = (Svec.iloc[::-1].rolling(4).sum()).iloc[::-1]

    return irsTTM
