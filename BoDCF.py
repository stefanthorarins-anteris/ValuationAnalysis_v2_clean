import pandas as pd
import requests

def BoDCF(resdic,mos=0.66,baseurl,api_key):
    symblist = resdic['postRank']['source']
    dcf = pd.DataFrame(columns=['symbol','buyprice','sellprice'])
    #dcf_df = pd.DataFrame(columns = ['symbol','mos','fcf', 'revenue', 'taxes', 'ebit', 'grossMargin', 'capex','dcf','buyprice','sellprice'])
    dcf_hist_df = pd.DataFrame(columns = ['symbol','fcf', 'revenue', 'taxes', 'ebit', 'grossMargin', 'capex'])
    dhd_bp_df = dcf_hist_df
    marketrisk = getMarketRisk()
    if len(symblist) > 100:
        wacctype = 'long'
    else:
        wacctype = 'short'

    cdx_df = resdic['cdx_df']
    for symb in symblist:
        dcf_hist_df = fillHistDCF(cdx_df,symb,dhd_bp_df)
        WACC = getWACC(symb,dcf_hist_df,'long',baseurl,api_key)
        buyprice,sellprice = getDCF(dcf_hist_df,mos,WACC)
        tempdcflist = [symb,buyprice,sellprice]
        dfc.loc[dfc.index.max() + 1] = tempdcflist

def getDCF(df,mos=0.66,WACC=0.12):
    meanFCFgr = df.fcf.pct_change(-4).mean()
    meanRevgr = df.revenue.pct_change(-4).mean()
    meanGMgr = df.grossMargin.pct_change(-4).mean()
    startgr = 0.5*(meanFCFgr + meanRevgr + meanGMgr)
    finalgr = 0.025
    n = 10
    d = (finalgr - startgr) / (n - 1)
    grlist = [startgr + (i - 1) * d for i in range(1, n + 1)]
    pvfcf_fw = []
    fcf_last = df.freeCashFlow.head(4).sum()
    for i in range(1,n):
        fcf_next = fcf_last*grlist[i]/WACC**i
        fcf_fw.append(fcf_next)
        fcf_last = fcf_next

    TG = fcf_last*(1+grlist[n-1])/(WACC-finalgr)
    sellprice = (sum(pvfcf_fw) + TG)
    buyprice = sellprice*mos

    return dcf

def fillHistDCF(cdx_df,symb,df):
    tempcdx = cdx_df[cdx_df['source'] == symb]
    df['fcf'] = tempcdx.freeCashFlow
    df['revenue'] = tempcdx.revenue
    df['taxes'] = tempcdx.effectiveTaxRate
    df['ebit'] = tempcdx.operatingIncome
    df['grossMargin'] = tempcdx.grossMargin
    df['capex'] = tempcdx.capex
    #df['shares'] = tempcdx.weightedAverageShsOutDil
    df['shares'] = tempcdx.weightedAverageShsOut

    return df

def getWACC(symb,df,baseurl,api_key,wacctype='long'):
    mr = 0.12
    br = 0.04
    if wacctype == 'long':
        WACC = 0.12
    else:
        pr = requests.get(f'{baseurl}v3/profile/{symb}?apikey={api_key}').json()
        beta = pr[0]['beta']
        x = df.debtEquityRatio
        y = 1/x
        we = 1/(x+1)
        wd = 1/(y+1)
        RoC = (df['interestExpense']/df['operatingIncome']).mean()
        tax = df['effectiveTaxRate'].mean()
        WACC = we.mean()*(br + beta*(mr-br)) + wd.mean()*RoC*(1-tax)
    return WACC

def getMarketRisk():

    return mr