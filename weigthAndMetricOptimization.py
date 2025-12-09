import pandas as pd
import createDicts as cdic
import calcScore as csf
import gains
from datetime import datetime


def main():
    ## PART 1
    # either load fundamentaldata or call getfundamentaldata
    # get gains for all tickers for the over the last N years
    # regress the metric fundamentals on them.
    bmfn = 'Bometric_dic-fmp_stock_US1_EU1_all_2023-02-16_len5314_manelim3692_fails3587.pickle'
    xreg = loadBoMetricForRegression(bmfn)
    symblist = bmdf_reg['source'].unique()
    sorteddatest = sorted(xreg['date'].unique())
    lastdate_xreg = sorteddatest[len(sorteddatest)-8]
    lastdate_yreg = sorteddatest[len(sorteddatest)-4]
    firstdate_yreg = datetime.strptime('2018-10-01','%Y-%m-%d')
    y_gains = gains.calculateGainWrapper(symblist,firstdate_yreg,lastdate_yreg)
    #Here we regress x_reg on y_gains.

    ## Part 2
    # after choosing the metrics to filter out the top X
    # we regress our ranking metrics on the gains of both all and the top X specifically

def loadBoMetricForRegression(bmfn):
    xdic = pd.read_pickle(bmfn)
    bmdf = xdic['BoMetric_df']
    xreg = pd.DataFrame(columns=bmdf.columns)
    numericols = list(set(bmdf.columns) - set(['source','date']))
    #bmdf_inv = bmdf.iloc[::-1]
    #xreg_inv = bmdf_inv.groupby('source')[numericols].rolling(window=4).mean().reset_index(level=0, drop=True).reset_index()
    #xreg = xreg_inv.iloc[::-1]in bmdf['source']:
    cntr = 0
    fullen = len(bmdf['source'].unique())
    for symbol in bmdf['source'].unique():
        #print(symbol)
        tempdf = bmdf[bmdf['source'] == symbol].iloc[::-1]
        newdf_inv = tempdf[numericols].rolling(window=4).mean()
        newdf = newdf_inv.iloc[::-1]
        xreg = pd.concat([xreg,newdf])

        cntr = cntr +1
        if cntr%500 == 0:
            rat = cntr/fullen
            print(f'{rat} finished')

    xreg[['date','source']] = bmdf[['date','source']]

    return xreg


def scoreForReg(bmdf,lastdate):
    meandic = csf.getAves2(bmdf)
    bm_ave = meandic['BoMetric_ave']
    bm_da = meandic['BoMetric_dateAve']
    d1, d2, d3, d4, d5 = cdic.getBaseMeanDiffUnitySpecialDicts()
    fulld = {**d1, **d2, **d3, **d4, **d5}
    xregcols = ['date']
    xregcols.append(fulld.keys())
    xreg = pd.DataFrame(columns=xregcols)
    n = 8
    for date in bmdf['date']:
        scoreNow = bmdf.loc[bmdf[bmdf['date'] <= date and bmdf['date'] <= date - n]]
        new_row = csf.simpleScore_fromDict(scoreNow, bm_ave, bm_da, n)
        xreg = xreg.append(new_row, ignore_index=True)

    return xreg

def calculateGainWrapper(symblist,firstdate,lastdate,fn='NA'):
    yreg = pd.DataFrame(columns=['source','gain'])
    if not fn == 'NA'
        for symbol in symblist:

        calculateGain(symbol, buydate, currentdate, histprices, histdivs):
        currentprices = getPrice(symbol, histprices, currentdate)
        buyprice = getPrice(symbol, histprices, buydate)
        divpersharetot = getDividends(symbol, histdivs, buydate, currentdate)
        gain = (currentprices - buyprice + divpersharetot) / buyprice
    else:
        load fn