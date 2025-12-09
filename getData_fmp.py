import pandas as pd
import requests
import calcMetrics as cm
import json
from tqdm import tqdm
import numpy as np
import warnings
import createDicts as cdic
import getData_gen as gdg
import failTests as ft
import utils as utils
from datetime import datetime


def get_fundamentals_fmp(Tickers_df, cdx_df, BoMetric_df, baseurl,
                         api_key,compyear, n=1, nrTaT=-1, startindex=0,period='quarter',limit=44):
    print('Fetching financial data from FMP and calculating relevant metrics.')
    if not isinstance(Tickers_df, pd.DataFrame):
        raise Exception('provide a DataFrame')
    if period == 'quarter' and limit < 16:
        raise Exception('Number of periods, if periods are quarters, must be larger than 16')
    tickersfailed = []
    lenfail = []
    datefail = []
    pricefail = []
    pricefailESN = []
    emptyfail = []
    hasCurrentYear = []
    if nrTaT < 0 and startindex == 0:
        pbar = tqdm(total=len(Tickers_df))
    elif nrTaT < 0 and startindex > 0:
        pbar = tqdm(total=len(Tickers_df)-startindex)
    else:
        total = min(nrTaT,len(Tickers_df)-startindex)
        pbar = tqdm(total=total)
    cntr = 0
    Tickers_df = Tickers_df.iloc[startindex: ,:]
    preReq_dict, BoMetric_Calc_dict, BoMetric_base_dict, BoMetric_mean_dict, BoMetric_diff_dict, \
        BoMetric_unity_dict, BoMetric_special_dict = cdic.getDicts()
    for row in Tickers_df.itertuples():
        cntr = cntr + 1
        ticker = row.symbol

        km, fr, inc, bs, cf, tickersfailed, lenfail, datefail, emptyfail = getFsData_fmp(ticker, period, limit,baseurl,
                                                                                         api_key, compyear, tickersfailed,
                                                                                         lenfail, datefail,emptyfail)
        if not (isinstance(km, int) and km == -37707):
            #fs = inc.merge(bs, on='date', how='inner').merge(cf, on='date', how='inner').merge(km, on='date', how='inner').merge(fr, on='date', how='inner')
            #tempfund, tempMetric_df, tempMetric_sum, temptckr_count = initTempMets(BoMetric_df.columns,
            #                                                                       BoMetric_sum.columns,
            #                                                                       BoTckr_count.columns, cdx_df.columns,
            #                                                                       bs['date'], ticker)
            tempfund, tempMetric_df = initTempMets(BoMetric_df.columns, cdx_df.columns,
                                                                                   bs['date'], ticker)

            tempfund, hcy = fillPreReqdf(tempfund, preReq_dict, bs, inc, cf, km, fr)
            tempMetric_df = utils.setDatesToQuarterly(tempMetric_df)
            if hcy == 1:
                hasCurrentYear.append(ticker)

            if not gdg.checkIfValidFS(tempfund):
                tickersfailed.append(ticker)
                pricefail.append(ticker)
                pricefailESN.append(row.exchangeShortName)
            else:
                tempdf = pd.DataFrame()
                tempdf['date'] = tempfund['date']
                # need to lag denominator for Assets, Investment and such [determined before t]

                ratioOpCalcDicts = {**BoMetric_base_dict, **BoMetric_mean_dict, **BoMetric_unity_dict, **BoMetric_diff_dict}
                for key in ratioOpCalcDicts:
                    restr = key
                    strUp = ratioOpCalcDicts[key]['Upper']
                    strDn = ratioOpCalcDicts[key]['Lower']
                    tf = cm.calc_simpleRatio(tempfund, strUp, strDn)
                    if key in BoMetric_base_dict:
                        tempMetric_df[restr] = tf
                    if key in BoMetric_mean_dict:
                        mrestr = "m" + restr[0].upper() + restr[1:]
                        tempMetric_df[mrestr] = tf
                    if key in BoMetric_unity_dict:
                        urestr = "u" + restr[0].upper() + restr[1:]
                        tempMetric_df[urestr] = tf
                    if key in BoMetric_diff_dict:
                        tempdf['forDiff'] = tf
                        tf = cm.calc_diff(tempdf,'forDiff',n)
                        drestr = "d" + restr[0].upper() + restr[1:]
                        tempMetric_df[drestr] = tf

                for key1 in BoMetric_special_dict.keys():
                    tf = cm.calc_special(tempfund, key1, n)
                    tempMetric_df[key1] = tf

                tempMetric_df_trimmed = tempMetric_df.drop(tempMetric_df.tail(4).index)
                #for i in tempMetric_df_trimmed.columns:
                #    if i == 'date':
                #        tempMetric_sum[i] = tempfund[i]
                #    else:
                #        tf = tempMetric_df_trimmed[i]
                #        tf_bool = pd.to_numeric(tf, errors='coerce')
                #        if not any(pd.isnull(tf_bool)):
                #            tempMetric_sum[i] = tf
                #            temptckr_count[i] = temptckr_count[i] + 1

                # align schemas (preserve all columns) before concatenation to avoid losing columns
                cols_union = BoMetric_df.columns.union(tempMetric_df_trimmed.columns)
                BoMetric_df = BoMetric_df.reindex(columns=cols_union)
                tempMetric_df_trimmed = tempMetric_df_trimmed.reindex(columns=cols_union)
                # perform concat while suppressing the specific FutureWarning about
                # concatenation with empty / all-NA entries (make this local only)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=FutureWarning,
                        message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated"
                    )
                    BoMetric_df = pd.concat([BoMetric_df, tempMetric_df_trimmed], ignore_index=True)
                #if BoMetric_sum.empty:
                #    BoMetric_sum.set_index('date')
                #BoMetric_sum = BoMetric_sum.add(tempMetric_sum.set_index('date'), fill_value=0)
                #BoTckr_count = BoTckr_count + temptckr_count
                #BoMetric_mean = BoMetric_sum/BoTckr_count
                # align schemas for cdx as well
                cols_union_cdx = cdx_df.columns.union(tempfund.columns)
                cdx_df = cdx_df.reindex(columns=cols_union_cdx)
                tempfund = tempfund.reindex(columns=cols_union_cdx)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=FutureWarning,
                        message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated"
                    )
                    cdx_df = pd.concat([cdx_df, tempfund], ignore_index=True)
                if row.symbol == 'CF':
                    print('This is tempMetric_df_trimmed from CF')
                    print(tempMetric_df_trimmed)

        if nrTaT > 0 and cntr == nrTaT:
            break
        elif len(tickersfailed) > (cntr + 1)*20:
            break
        pbar.update(n=1)
    pbar.close()

    #BoMetric_df = utils.setDatesToQuarterly(BoMetric_df)
    BoMetric_df, cdx_df = gdg.fixAfterGetData(BoMetric_df, cdx_df)
    #resfunddic = {'BoMetric_df':BoMetric_df, 'BoMetric_sum':BoMetric_sum, 'BoTckr_count' : BoTckr_count,
    #              'cdx_df': cdx_df, 'tickersfailed': tickersfailed, 'lenfail': lenfail,
    #              'datefail': datefail, 'emptyfail': emptyfail, 'cind': cntr}
    resfunddic = {'BoMetric_df':BoMetric_df,
                  'cdx_df': cdx_df, 'tickersfailed': tickersfailed, 'lenfail': lenfail, 'pricefail': pricefail,
                  'datefail': datefail, 'emptyfail': emptyfail, 'cind': cntr, 'hasCurrentYear': hasCurrentYear}
    return resfunddic

def fillPreReqdf(tempfund,preReq_dict,bs,inc,cf,km,fr):
    hcybool = False
    for key1 in preReq_dict:
        for i in preReq_dict[key1]:
            if key1 == 'bs':
                tempfund[i] = bs[i]
            elif key1 == 'inc':
                tempfund[i] = inc[i]
            elif key1 == 'cf':
                tempfund[i] = cf[i]
            elif key1 == 'km':
                tempfund[i] = km[i]
            elif key1 == 'fr':
                tempfund[i] = fr[i]
            else:
                #tempfund['shares'] = inc['revenue'] / km['revenuePerShare']
                tempfund['price'] = fr['priceEarningsRatio'] * (inc['netIncome'] / inc['weightedAverageShsOut'])
    tempfund = utils.setDatesToQuarterly(tempfund)
    if tempfund['date'].iloc[0].year == datetime.today().year:
        hcybool = True

    return tempfund, hcybool

def getFsData_fmp(ticker, period, limit, baseurl, api_key,compyear, tickersfailed, lenfail,datefail,emptyfail):

    failcodes = list(range(400, 600))
    failbool, whyfail, outdic = ft.testForAPIFaults_fmp(failcodes,compyear,ticker,period,limit,baseurl,api_key)
    if failbool:
        tickersfailed.append(ticker)
        if whyfail == 'datefail':
            datefail.append(ticker)
        elif whyfail == 'lenfail':
            lenfail.append(ticker)
        elif whyfail == 'emptyfail':
            emptyfail.append(ticker)
        km, fr, inc, bs, cf = -37707, -1, -1, -1, -1
    else:
        km = outdic['km'] #pd.DataFrame.from_dict(resp_km.json())
        fr = outdic['fr'] #pd.DataFrame.from_dict(resp_fr.json())
        inc = outdic['inc'] #pd.DataFrame.from_dict(resp_inc.json())
        bs = outdic['bs'] #pd.DataFrame.from_dict(resp_bs.json())
        cf = outdic['cf'] #pd.DataFrame.from_dict(resp_cf.json())

    return km, fr, inc, bs, cf, tickersfailed, lenfail, datefail, emptyfail

def symbchRestock(tckrs_df,baseurl,period,limit,api_key,compyear,timdir='old2new'):
    symbch_df = pd.DataFrame(requests.get(f'https://financialmodelingprep.com/api/v4/symbol_change?apikey={api_key}').json())
    if timdir == 'new2old':
        int = list(set(symbch_df.newSymbol) & set(tckrs_df.symbol))
    else:
        int = list(set(symbch_df.oldSymbol) & set(tckrs_df.symbol))

    succbool_lvl2 = []
    failcodes = list(range(400, 600))
    failers = []
    failers_lvl2 = []
    succnotadded = []
    succ_lvl2 = []
    failstosucc = []

    print(f'Starting symbol restock: {timdir}')
    pbar = tqdm(total=len(int))
    for ticker in int:
        failbool_lvl2_agg = False
        resp_km = requests.get(f'{baseurl}/key-metrics/{ticker}?period={period}&limit={limit}&apikey={api_key}')
        resp_fr = requests.get(f'{baseurl}/ratios/{ticker}?period={period}&limit={limit}&apikey={api_key}')
        resp_inc = requests.get(f'{baseurl}/income-statement/{ticker}?period={period}&limit={limit}&apikey={api_key}')
        resp_bs = requests.get(f'{baseurl}/balance-sheet-statement/{ticker}?period={period}&limit={limit}&apikey={api_key}')
        resp_cf = requests.get(f'{baseurl}/cash-flow-statement/{ticker}?period={period}&limit={limit}&apikey={api_key}')
        respstatcodes = [resp_km.status_code, resp_fr.status_code, resp_inc.status_code, resp_bs.status_code,
                         resp_cf.status_code]
        failbool, whyfail = ft.testForAPIFaults(respstatcodes, failcodes,compyear, resp_km, resp_fr, resp_inc, resp_bs, resp_cf)
        if failbool:
            failers.append(ticker)
            if timdir == 'new2old':
                nt_df = symbch_df[symbch_df['newSymbol'] == ticker]['oldSymbol']
            else:
                nt_df = symbch_df[symbch_df['oldSymbol'] == ticker]['newSymbol']
            for i in range(0, len(nt_df)):
                ticker_next = nt_df.iloc[i]
                resp_km = requests.get(f'{baseurl}/key-metrics/{ticker_next}?period={period}&limit={16}&apikey={api_key}')
                resp_fr = requests.get(f'{baseurl}/ratios/{ticker_next}?period={period}&limit={16}&apikey={api_key}')
                resp_inc = requests.get(f'{baseurl}/income-statement/{ticker_next}?period={period}&limit={16}&apikey={api_key}')
                resp_bs = requests.get(f'{baseurl}/balance-sheet-statement/{ticker_next}?period={period}&limit={16}&apikey={api_key}')
                resp_cf = requests.get(f'{baseurl}/cash-flow-statement/{ticker_next}?period={period}&limit={16}&apikey={api_key}')
                respstatcodes = [resp_km.status_code, resp_fr.status_code, resp_inc.status_code, resp_bs.status_code,resp_cf.status_code]
                failbool_lvl2, whyfail = ft.testForAPIFaults(respstatcodes, failcodes,compyear, resp_km, resp_fr, resp_inc, resp_bs,resp_cf)
                if failbool_lvl2:
                    failbool_lvl2_agg = True
                    failers_lvl2.append(ticker_next)
                elif ticker_next not in list(tckrs_df['symbol']):
                    tckrs_df.loc[tckrs_df['symbol'] == ticker, 'symbol'] = ticker_next
                    failstosucc.append(ticker_next)
                else:
                    succnotadded.append(ticker_next)
            if failbool_lvl2_agg == False:
                succ_lvl2.append(ticker)
        pbar.update(n=1)
    pbar.close()
    fullfail = list(set(failers + failers_lvl2))
    tckrs_df_new = tckrs_df[~tckrs_df['symbol'].isin(fullfail)].reset_index(drop=True)

    if len(failers) > 0:
        pcfixed = len(succ_lvl2)/len(failers)
        pcnotadded = len(succnotadded)/len(failers)

    return tckrs_df_new, failers, failers_lvl2, succ_lvl2, succnotadded, failstosucc, pcfixed, pcnotadded

def initTempMets(dfcols,cdxcols,datevec,ticker):
    tempMetric_df = pd.DataFrame(columns=dfcols)
    #tempMetric_sum = pd.DataFrame(columns=sumcols)
    tempfund = pd.DataFrame(columns=cdxcols)
    #temptckr_count = pd.DataFrame({string: [0] for string in countcols})
    tempfund['date'] = datevec
    tempfund['source'] = ticker
    tempMetric_df['date'] = datevec
    tempMetric_df['source'] = ticker
    #tempMetric_sum['date'] = datevec

    #return tempfund, tempMetric_df, tempMetric_sum, temptckr_count
    return tempfund, tempMetric_df