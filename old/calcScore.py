import pandas as pd
import numpy as np

## ÞETTA FALL ER SORG
def get_avs_yf(df,relAvsItems,itemAlts_dict):
    # ALLT Í RUGLI
    sms = pd.DataFrame()
    avs = pd.DataFrame()
    sms.columns = [relAvsItems]
    avs.columns = [relAvsItems]

    for accitem in relAvsItems:
        itemAlt = itemAlts_dict[accitem]
        for ticker in balbal
            df = df.loc[(df[itemAlt].notnull().all(axis=1)) & (df['Source'] != ticker)]
            df = df.drop(df[(df[itemAlt].isnull().any(axis=1)) | (df['Source'] == ticker)].index)

        perOfNull_df.columns = [relAvsItems]
        ## check null
        df_nancheck = df['Date',itemAlts,'Source']
        df_nan = df_nancheck[itemAlts].isnull().assign(date=df['date'], source=df['source'])

        test = df_nan.groupby(['date', 'source'])
        faillist = df_nan[keywords].all(axis=1).astype(int)
        cntr = 0
        mask = df.columns.isin(itemAlts)
        nrOfmatches = mask.sum()
        if nrOfmatches == 0:
            raise Exception(f"Can't find a colum associated with {accitem}")
        elif nrOfmatches == 1:
            # check for null and remove
            df_nonan, perOfNull_df = remnansFromdfAtCol(df, accitem)
            avs[accitem] = df_nonan.groupby('Date')[accitem].mean()
            sms[accitem] = df_nonan.groupby('Date')[accitem].sum()
        elif nrOfmathces > 1
            for i, value in enumerate(mask):
                if value:
                    # check for null and remove
                    df_nonan, perOfNull_tempdf = remNansFromDfAtCol(df, itemAlts[i])
                    perOfNull_df[accitem] += perOfNull_tempdf[itemAlts[i]]
                    avs[accitem] = df_nonan.groupby('Date')[accitem].mean()
                    sms[accitem] = df_nonan.groupby('Date')[accitem].sum()
    return avs, sms, perOfNull_df

def createBoMetricdf_fromtxt(bmlist)
    Bometric_df = pd.DataFrame()
    columnnames = open(bmlist,'r').read()

    with open(bmlist, 'r') as f:
        column_names = f.readline().strip().split(',')

    # Create empty DataFrame using extracted column names
    Bometric_df = pd.DataFrame(columns=column_names)

def createBoMetricdf_fromdicts(dict_base,dict_mean,dict_diff,dict_special,BoScore_df):
    collist = []
    for key in dict_base:
        collist.append(key)
    for key in dict_diff:
        dstr = 'd' + key
        dstr = dstr[:1] + dstr[1:].capitalize()
        collist.append(dstr)
    for key in dict_mean:
        mstr = 'm' + key
        mstr = mstr[:1] + mstr[1:].capitalize()
        collist.append(mstr)
    for key in dict_special:
        collist.append(key)

    BoMetric_df = pd.DataFrame(columns=collist)
    return BoMetric_df

def remNansFromDfAtCol(df,col)
    nanbool = df[col].isnull()
    numbool = ~nanbool
    numOfNull = sum(bool(x) for x in numbool)
    perOfNull_df = pd.DataFrame()
    perOfNull_df.columns = [col]
    perOfNull_df[col] = numOfNull / len(df[col])
    df_nonan = df[accitem][numbool.values]

    return df_nonan,perOfNull_df



def add_to_boscoredf(tckr,score,boscoredf):
    if isinstance(tckr, str):
        if tckr in boscoredf['Tickers'].values:
            boscoredf.loc[boscoredf['Tickers'] == tckr, 'BoScore'] = boscoredf.loc[boscoredf[
                                                                                       'Tickers'] == tckr, 'BoScore'] + score
        else:
            tmpdf = pd.DataFrame()
            tmpdf['Tickers'] = [tckr]
            tmpdf['BoScore'] = [score]
            boscoredf = pd.concat([boscoredf, tmpdf])
            boscoredf.reset_index(drop=True, inplace=True)

    return boscoredf

def PercentageDifferenceOfDifference(x,y):
    d1 = x[0]-x[1]
    d2 = y[0]-y[1]
    pdod = d1/d2-1
    return pdod

def aveRatioDiff(x,y,n):
    s1 = sum(x[i] / y[i] for i in range(0, n))
    s2 = sum(x[i] / y[i] for i in range(n+1,2n))
    return s1 - s2

def lastHigherThanAve(x,n):
    a = sum(x[1:n])
    return x[1] - m

def aveRatioFromMean(x,y,m,n):
    s1 = sum(x[i] / y[i] for i in range(0, n)
    return s1 - m

def aveFromMean(x,m,n):
    s1 sum(x[:n])
    return s1 - m

def calc_STIER_score(df,tckr,boscoredf,priceHist, nrShares, score, avs, ds='yf'):
    weight = 1
    if ds == 'yf':
        relIndicators = ['Net Income', 'Total Assets', 'Operating Cash Flow']
    elif ds == 'fmp':
        relIndicators = ['Net Income', 'Total Assets', 'Operating Cash Flow']
    else:
        raise Exception('double bruh')

    ninc = df[relIndicators[0]]
    ass = df[relIndicators[1]]
    RoA = ninc[0]/ass[0]
    if RoA > 0:
        score = score + weight*1

    RoA2 = ninc[1]/ass[1]
    if RoA - RoA2 > 0:
        score = score + weight*1

    CFO = df[relIndicators[2]]
    if CFO[0] > 0:
        score = score + weight

    boscoredf = add_to_boscoredf(tckr, score, boscoredf)

    return boscoredf


def calc_ATIER_score(df, tckr, boscoredf, priceHist, nrShares, score, avs, ds='yf'):
    weight = 0.75
    if ds == 'yf':
        relIndicators = ['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity', 'Share Issued', 'Gross Profit']
    elif ds == 'fmp':
        relIndicators = ['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity', 'Share Issued', 'Gross Profit']
    else:
        raise Exception('double bruh')
    sEQ = df[relIndicators[3]]
    totA = df[relIndicators[0]]
    totL = sEQ - totA

    nrShares = df[relIndicators[4]]
    mC = nrShares[0]*priceHist[0]
    btom = sEQ[0]/mC

    if btom > avs['btom']:
        score = score + weight

    gm = df[relIndicators[5]]
    gtoa = gm[0]/totA[0]

    if gtoa > avs['gtoa']:
        score = score + weight

    boscoredf = add_to_boscoredf(tckr,score,boscoredf)

    return boscoredf


def calc_BTIER_score(df, tckr, boscoredf, priceHist, nrShares, score, avs, ds='yf'):
    weight = 0.5
    if ds == 'yf':
        relIndicators = ['Inventory', 'Operating Revenue', 'Total Assets', 'Operating Cash Flow']
    elif ds == 'fmp':
        relIndicators = ['Total Assets', 'Total Liabilities']
    else:
        raise Exception('double bruh')
    I = df[relIndicators[0]]
    Sls = df[relIndicators[1]]
    Einv = 0.5*(I[1] + I[2])
    Esls = 0.5*(Sls[1] + Sls[2])
    if Sls[0] - Esls > I[0] - Einv:
        score = score + weight

    totA = df[relIndicators[0]]
    if Sls[0]/totA[0] > Sls[1]/totA[1]:
        score = score + weight

    CFO = df[relIndicators[4]]
    mC = nrShares[0] * priceHist[0]
    ctom = CFO[0]/mC

    if ctom > avs['ctom']:
        score = score + weight

    boscoredf = add_to_boscoredf(tckr, score, boscoredf)

    return boscoredf

def calc_CTIER_score(df, tckr, boscoredf, priceHist, nrShares, score, avs, ds='yf'):
    weight = 0.3
    if ds == 'yf':
        # no employees info??
        relIndicators = ['Current Assets', 'Current Liabilities', 'Net Income', 'Stockholders Equity', 'Gross Profit', 'Operating Revenue', 'Tax Provision', 'Pretax income', 'Share Issued', 'Net PPE Purchase And Sale', 'Depreciation And Amortization', 'Operating Cash Flow']
    elif ds == 'fmp':
        relIndicators = ['Current ratio', 'Net Income', 'Stockholders Equity', 'Gross Profit', 'Operating Revenue', 'Net PPE Purchase And Sale', 'Depreciation And Amortization']
    else:
        raise Exception('double bruh')

    cA = df[relIndicators[0]]
    cL = df[relIndicators[1]]
    if cA[0]/cL[0] - cA[1]/cL[1] > 0:
        score = score + weight

    ninc = df[relIndicators[2]]
    Eq = df[relIndicators[3]]
    dRoE = ninc[0]/Eq[0] - ninc[1]/Eq[1]
    if dRoE > 0:
        score = score + weight

    if Eq[0] - Eq[1] > 0:
        score = score + weight

    GM = df[relIndicators[4]]
    ORev = df[relIndicators[5]]
    if GM[0]/ORev[0] - GM[1]/ORev[1]:
        score = score + weight

    ## ElogD(SALES/EMPLOYEES) HERE

    Tax = df[relIndicators[5]]
    ninc_pt = df[relIndicators[6]]
    effTax = Tax[0]/ninc_pt[0] - Tax[1]/ninc_pt[1]
    if effTax < 0:
        score = score + weight

    if ORev[0] > 0.5*(ORev[1] + ORev[2]):
        score = score + weight

    mC0 = nrShares[0]*priceHist[0]
    mC1 = nrShares[0]*priceHist[1]
    stomc = 0Rev[0]/mC0

    if stomc > avs[stomc]:
        score = score + weight

    if CFO[0]/mC0 - CFO[1]/mC1 > 0
        score = score + weight

    if ninc[0]/priceHist[0] > avs[etop]
        score = score + weight
    boscoredf = add_to_boscoredf(tckr, score, boscoredf)

    return boscoredf
