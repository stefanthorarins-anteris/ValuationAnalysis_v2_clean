import pandas as pd
import numpy as np

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

def calc_compRatio(df,strUp,strDn,metstr,n):
    res = pd.DataFrame()
    if strDn == 'Identity':
        res[metstr] = df[strUp]
    else:
        res[metstr] = df[strUp] / df[strDn]

    res = res.iloc[::-1]
    res[metstr] = res[metstr].rolling(n).mean()
    res = res.iloc[::-1]

    return res.tolist()

def calc_diff(df,metstr,n):
    res = pd.DataFrame()

    dstr = "d" + metstr[0].upper() + metstr[1:]
    res[dstr] = df[metstr] - df[metstr].shift(-1)
    res = res.iloc[::-1]
    res[dstr] = res[dstr].rolling(n).mean()
    res = res.iloc[::-1]

    return res

def calc_special(df,metstr,n):
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
    if metstr == 'PEG':
        res[metstr] = np.where(df['priceEarningsToGrowthRatio'] != 0, 1 / df['priceEarningsToGrowthRatio'], 0) - 1
    #elif str == 'CFOlessEarnings':
    #    res[metstr] = df['netCashProvidedByOperatingActivities'] - df['netIncome']
    #    res = res.iloc[::-1]
    #    res[metstr] = res[metstr].rolling(n).mean()
    #    res = res.iloc[::-1]
    elif metstr == 'returnOnEquity':
        #res[metstr] = df['netIncome']/df['totalStockholdersEquity'] - 0.12/4
        res[metstr] = df['returnOnEquity'] - 0.12/4
    elif metstr == 'EPStoEPSmean':
        eps = df['netIncome']/df['weightedAverageShsOut']
        epsmean = eps.mean()
        res[metstr] = eps - epsmean
    elif metstr == 'capitalExpenditureCoverageRatio':
        tempce2cr = df[metstr]
        ce2cr = -tempce2cr.fillna(0)
        res[metstr] = ce2cr - 2

    return res