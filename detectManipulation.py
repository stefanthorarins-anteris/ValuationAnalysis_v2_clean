import pandas as pd
import numpy as np
import warnings

# Suppress FutureWarning about DataFrame concatenation with empty/all-NA entries
warnings.filterwarnings('ignore', message='.*concatenation with empty or all-NA entries.*')

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

        cfoTTM = invrollsumTTM(tempcdx_df['netCashProvidedByOperatingActivities'])
        niTTM = invrollsumTTM(tempcdx_df['netIncome'])
        NICFO = (niTTM - cfoTTM)/cfoTTM.abs()
        tmpcdf['NICFOdiv'] = NICFO.diff(periods=-4).fillna(99999)

        dsoTTM = invrollsumTTM(tempcdx_df['daysSalesOutstanding'])
        tmpcdf['DSOinc'] = dsoTTM.diff(periods=-4).fillna(99999)

        dsiTTM = invrollsumTTM(tempcdx_df['daysOfInventoryOutstanding'])
        tmpcdf['DSIinc'] = dsiTTM.diff(periods=-4).fillna(99999)

        ocarTTM = invrollsumTTM(tempcdx_df['otherCurrentAssets'])/invrollsumTTM(tempcdx_df['revenue'])
        tmpcdf['OCARinc'] = ocarTTM.diff(periods=-4).fillna(99999)

        ddaTTM = invrollsumTTM(tempcdx_df['depreciationAndAmortization'])
        capex = tempcdx_df['capexPerShare']*tempcdx_df['weightedAverageShsOut']
        nppeTTM = invrollsumTTM(tempcdx_df['propertyPlantEquipmentNet'])
        adTTM = ddaTTM.iloc[::-1].cumsum().iloc[::-1]
        gppeTTM = nppeTTM - capex + adTTM
        dappTTM = ddaTTM/gppeTTM
        tmpcdf['DAPPdec'] = -dappTTM.diff(periods=-4).fillna(99999)

        taTTM = invrollsumTTM(tempcdx_df['totalAssets'])
        tmpcdf['TAgr'] = taTTM.pct_change(-4, fill_method=None).fillna(99999) - 0.1

        tmpcdf['C_Score'] = (tmpcdf > 0).sum(axis=1)

        tmpcdf['date'] = tempcdx_df['date']
        tmpcdf['symbol'] = tempcdx_df['source']

        symb_cscore = tmpcdf[0:len(tmpcdf)-4]

        cdf = pd.concat([cdf, symb_cscore])
        # head(2): newest-first, so the 2 MOST RECENT quarters (current condition).
        cscore = symb_cscore['C_Score'].head(2).mean()
        SLmeanCscore.loc[SLmeanCscore['source']==symbol, 'C_Score_mean'] = cscore

        if np.isnan(cscore) or np.isinf(cscore):
            problemlist.append(symbol)
        elif cscore > 4:
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
        dsoTTM = invrollsumTTM(tempcdx_df['daysSalesOutstanding'])
        salesTTM = invrollsumTTM(tempcdx_df['revenue'])
        dsriTTM = dsoTTM/salesTTM
        tmpmdf['DSRI'] = _yoyCurOverPrior(dsriTTM)          # (DSO/Sales)_t / (DSO/Sales)_{t-1}

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

        niTTM = invrollsumTTM(tempcdx_df['netIncome'])
        cfoTTM = invrollsumTTM(tempcdx_df['netCashProvidedByOperatingActivities'])
        cffTTM = invrollsumTTM(tempcdx_df['netCashUsedProvidedByFinancingActivities'])
        # TATA is a current-period LEVEL (higher accruals = more suspicious), so it
        # carries no YoY orientation to invert. NOTE: the extra -cffTTM term is a
        # local modeling choice, NOT published Beneish (published TATA = (NI-CFO)/TA);
        # left unchanged here because it is not an orientation/recency defect.
        tmpmdf['TATA'] = (niTTM - cfoTTM - cffTTM)/taTTM

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
