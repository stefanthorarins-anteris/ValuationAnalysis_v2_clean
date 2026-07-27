import pandas as pd
import numpy as np

import reporting_period as rp

# `rpy` = this source's rows per year (4 quarterly / 2 semi-annual, reporting_period).
# It scales the moving-average window so it spans the same CALENDAR time, and it is the
# divisor for any ANNUAL rate expressed per period.  rpy defaults to 4 -> unchanged.

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

def calc_special(df,metstr,n,rpy=rp.DEFAULT_ROWS_PER_YEAR):
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
        res[metstr] = np.where(df['priceEarningsToGrowthRatio'] != 0, 1 / df['priceEarningsToGrowthRatio'], 0) - 1
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
        res[metstr] = df['returnOnEquity'] - 0.12/float(rpy)
    elif metstr == 'EPStoEPSmean':
        eps = df['netIncome']/df['weightedAverageShsOut']
        epsmean = eps.mean()
        res[metstr] = eps - epsmean
    elif metstr == 'capitalExpenditureCoverageRatio':
        tempce2cr = df[metstr]
        ce2cr = -tempce2cr.fillna(0)
        res[metstr] = ce2cr - 2

    return res