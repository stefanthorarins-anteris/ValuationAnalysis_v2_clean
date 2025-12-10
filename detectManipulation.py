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

def calcMontierC(resdic,symblist):
    cdx_df = resdic['cdx_df']
    SLmeanCscore = pd.DataFrame(columns=['source', 'C_Score_mean'])
    SLmeanCscore['source'] = symblist
    cdf = pd.DataFrame(columns=['date', 'symbol', 'NICFOdiv','DSOinc','DSIinc','OCARinc','DAPPdec','TAgr','C_Score'])
    problemlist = []
    for symbol in symblist:
        tmpcdf = pd.DataFrame(columns=['date', 'symbol', 'NICFOdiv','DSOinc','DSIinc','OCARinc','DAPPdec','TAgr','C_Score'])
        # C-score if NICFO > 0, DSOinc > 0 ...
        tempcdx_df = cdx_df[cdx_df['source'] == symbol]

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
        tempcdx_df = cdx_df[cdx_df['source'] == symbol]
        dsoTTM = invrollsumTTM(tempcdx_df['daysSalesOutstanding'])
        salesTTM = invrollsumTTM(tempcdx_df['revenue'])
        dsriTTM = dsoTTM/salesTTM
        tmpmdf['DSRI'] = dsriTTM.pct_change(-4, fill_method=None) + 1

        gmiTTM = invrollsumTTM(tempcdx_df['grossProfitMargin'])
        tmpmdf['GMI'] = gmiTTM.pct_change(-4, fill_method=None) + 1

        tcaTTM = invrollsumTTM(tempcdx_df['totalCurrentAssets'])
        ppenTTM = invrollsumTTM(tempcdx_df['propertyPlantEquipmentNet'])
        taTTM = invrollsumTTM(tempcdx_df['totalAssets'])
        aqiTTM = 1- (tcaTTM + ppenTTM)/taTTM
        tmpmdf['AQI'] = aqiTTM.pct_change(-4, fill_method=None) + 1

        sgiTTM = invrollsumTTM(tempcdx_df['revenue'])
        tmpmdf['SGI'] = sgiTTM.pct_change(-4, fill_method=None) + 1

        #z = x / (x + y) = 1 / ((x + y) / (x)) = 1 / (1 + (y / x))
        # Ef w = y / x, þá: z = 1 / (1 + w). x = depreciationAndAmortization, y = PP&Enet
        ddaTTM = invrollsumTTM(tempcdx_df['depreciationAndAmortization'])
        w = ppenTTM/ddaTTM
        depiTTM = 1/(1+w)
        tmpmdf['DEPI'] = depiTTM.shift(-1)/depiTTM

        sgaTTM = invrollsumTTM(tempcdx_df['sellingGeneralAndAdministrativeExpenses'])
        sgaiTTM = sgaTTM/sgiTTM
        tmpmdf['SGAI'] = sgaiTTM.pct_change(-4, fill_method=None) + 1

        ltdTTM = invrollsumTTM(tempcdx_df['longTermDebt'])
        clTTM = invrollsumTTM(tempcdx_df['totalCurrentLiabilities'])
        lvgiTTM = (ltdTTM+clTTM)/taTTM
        tmpmdf['LVGI'] = lvgiTTM.pct_change(-4, fill_method=None) + 1

        niTTM = invrollsumTTM(tempcdx_df['netIncome'])
        cfoTTM = invrollsumTTM(tempcdx_df['netCashProvidedByOperatingActivities'])
        cffTTM = invrollsumTTM(tempcdx_df['netCashUsedProvidedByFinancingActivities'])
        tmpmdf['TATA'] = (niTTM - cfoTTM - cffTTM)/taTTM

        tmpmdf['M_Score'] = - 4.84 + 0.92*tmpmdf.DSRI + 0.528*tmpmdf.GMI + 0.404*tmpmdf.AQI + 0.892*tmpmdf.SGI +\
                            0.115*tmpmdf.DEPI - 0.172*tmpmdf.SGAI + 4.679*tmpmdf.TATA - 0.327*tmpmdf.LVGI + 1.78
        tmpmdf['date'] = tempcdx_df['date']
        tmpmdf['symbol'] = tempcdx_df['source']
        symb_mscore = tmpmdf[0:len(tmpmdf)-4]

        mdf = pd.concat([mdf,symb_mscore])
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
