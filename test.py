 ## test if two dicst t he same
for key in xdic.keys():
    if isinstance(xdic[key], pd.DataFrame):
        if not isinstance(resdic[key], pd.DataFrame):
            print(f'here is a problem {key}')
        else:
            xdictemp = xdic[key].apply(pd.to_numeric, errors='coerce', axis=1)
            resdictemp = resdic[key].apply(pd.to_numeric, errors='coerce', axis=1)
            compdf = xdictemp - resdictemp
            compdf = compdf.dropna(axis=1)
            if (compdf != 0).any().any():
                print(f'here is a problem with nonzero elements {key}')
## test stock screener
tempss = requests.get(f'https://financialmodelingprep.com/api/v3/stock-screener?sector=Technology&apikey={api_key}').json()
tempssbeta = requests.get(f'https://financialmodelingprep.com/api/v3/stock-screener?betaLowerThan=0.2&apikey={api_key}').json()
## MAIN TEST
import sys
import pandas as pd
import getData_gen as gdg
import requests
import createDicts as cdic
import numpy as np
import calcMetrics as cm
import calcScore as csf
import getData_fmp as gdf
from datetime import datetime
import failTests as ft
from tqdm import tqdm
#import postBoScore as pbs
import os
import csv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#tickerfilter = 'stock_US1'
tickerfilter = 'stock_NA1'
#tickerfilter = 'stock_WW1_TV'
#tickerfilter = 'stock_NA1_EU1'
rT_bool = 0
ds_t = 'fmp'
ds_f = 'fmp'

api_key_fname = 'fmpAPIkey.txt'
baseurl = "https://financialmodelingprep.com/api/v3"
## conf done. Make everything compatible
if ds_t == 'fmp':
    rT_bool = 0
lastindex_fn = 'lastIndexOfRead_' + tickerfilter + '.txt'
## Get list of symbols and last index of ticker read
# assing API key
api_key = open(api_key_fname,'r').read()
#get the starting index for getting data for fundamentals
startindex = gdg.get_lastIndexRead(lastindex_fn)
# get Tickers
mcap = -1
period = 'quarter'
limit = 6*4
nrTaT = -1
compyear = datetime.now().year - 1
n = 1

saveBoMetric = 1
saveBoResults = 1
loadBoMetric = 0
loadBoResults = 0
symbchRestock = 0
loadBoMetricfname = 'BoMetric_dic-2023-01-27_stock_NA1_9546_fails4731.pickle'
loadBoResultsfname = 'BoResults_dic-2023-01-27_stock_NA1_9546.pickle'
manelimtick_fname_toget = 'manualEliminationTickersList_stock_NA1_2023-01-24.csv'

with open(manelimtick_fname_toget, 'r') as file:
    reader = csv.reader(file)
    templist = list(reader)
    manual_elims_tickers = templist[0]
#manual_elims_tickers = []

Tickers_df = gdg.get_tickers(ds_t, api_key, manual_elims_tickers, tickerfilter, rT_bool)
if symbchRestock == 1 and loaddata == 0:
    Tickers_df, failers_o2n, failers_lvl2_o2n, succ_lvl2_o2n, succnotadded_o2n, failstosucc_o2n, pcfixed_o2n, pcnotadded_o2n = gdf.symbchRestock(Tickers_df,baseurl,period,limit,api_key,compyear,'old2new')

    manual_elims_tickers = list(set(manual_elims_tickers + failers_o2n + failers_lvl2_o2n))
    manelimtick_fname = 'manualEliminationTickersList_' + tickerfilter + '_' + datetime.now().strftime('%Y-%m-%d') + ".csv"
    with open(manelimtick_fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(manual_elims_tickers)

BoMetric_df, BoMetric_sum, BoTckr_count, cdx_df = gdf.initBoMetric_fromDict(baseurl,api_key)
if not cdic.dictCheckValid():
    raise Exception('Something wrong with the dictionary setup')

if not loadBoMetric:
    BoMetric_df, BoMetric_sum, BoTckr_count, cdx_df, tickersfailed, lenfail, datefail, cind = gdf.get_fundamentals_fmp(Tickers_df, cdx_df, BoMetric_df, BoTckr_count, BoMetric_sum, baseurl, api_key, compyear, n, nrTaT, startindex,period,limit)
    BoMetric_df, cdx_df = gdg.fixAfterGetData(BoMetric_df, cdx_df)
    BoMetric_ave, BoMetric_dateAve, colslost_ave = csf.getAves(BoMetric_df)
    gdg.write_lastIndexRead(lastindex_fn, cind)
else:
    savedata = 0
    data = pd.read_pickle(loadBoMetricfname)
    BoMetric_df = data['BoMetric_df']
    BoTckr_count = data['BoTckr_count']
    tickersfailed = data['tickersfailed']
    cdx_df =data['cdx_df']
    cind = data['cind']
    lenfail = data['lenfail']
    tickerfilter = data['tickerfilter']

if saveBoMetric:
    fidag = datetime.today().strftime('%Y-%m-%d')
    fname_bmdf = f'BoMetric_dic-{fidag}_{tickerfilter}_{len(Tickers_df)}_fails{len(tickersfailed)}.pickle'
    datadic = {'BoMetric_df': BoMetric_df, 'BoTckr_count': BoTckr_count, 'tickersfailed': tickersfailed, 'cdx_df': cdx_df, 'lenfail': lenfail, 'cind': cind, 'tickerfilter': tickerfilter}
    pd.to_pickle(datadic, fname_bmdf)

if not loadBoResults:
    ## Calc BoScore
    bsfl = csf.simpleScore_fromDict(BoMetric_df, BoMetric_ave, BoMetric_dateAve,16)
    BoScore_df = bsfl

    bsdf_top20 = BoScore_df['source'].head(20)
    nl = '\n'
    fidag = datetime.today().strftime('%Y-%m-%d')
    if nrTaT < 0:
        fnstocks = len(Tickers_df)
    else:
        fnstocks = nrTaT

    fname_t20 = f'top20-{fidag}_{tickerfilter}_{fnstocks}.txt'
    with open(f'{fname_t20}', 'w') as f:
        f.write(f'Top 20 is: {nl}{nl.join(list(bsdf_top20))}')

    # Calc Ranking
    BoS_dftop100 = BoScore_df.head(100)
    BoM_dftop100 = BoMetric_df[BoMetric_df['source'].isin(list(BoS_dftop100.source))].reset_index(drop=True)
    cdx_dftop100 = cdx_df[cdx_df['source'].isin(list(BoS_dftop100.source))].reset_index(drop=True)

    finalBoRank_df, BoRank_df, BoROR_df, BoRS_df, BoAggCorr = pbs.postBoScoreRanking(BoM_dftop100,BoS_dftop100,cdx_dftop100,baseurl,api_key,'quarter',9)
else:
    saveBoResults = 0
    data = pd.read_pickle(loadBoResultsfname)
    postScoreMetric_df = data['postScoreMetric_df']
    BoRank_df = data['BoRank_df']
    BoROR_df = data['BoROR_df']
    BoRS_df =data['BoRS_df']
    Tickers_df = data['Tickers_df']
    BoScore_df = data['BoScore_df']
if saveBoResults:
    fidag = datetime.today().strftime('%Y-%m-%d')
    fname_brdf = f'BoResults_dic-{fidag}_{tickerfilter}_{len(Tickers_df)}.pickle'
    datadic = {'finalBoRank_df': finalBoRank_df, 'BoRank_df': BoRank_df, 'BoROR_df': BoROR_df, 'BoRS_df': BoRS_df, 'BoScore_df':BoScore_df, 'Tickers_df': Tickers_df}
    pd.to_pickle(datadic, fname_brdf)


flbr_descript = Tickers_df[Tickers_df['symbol'].isin(list(finalBoRank_df['source']))]
fname_AggScoretop40 = f'AggScoreTop40-{fidag}_{tickerfilter}.csv'

pbs.writeBoAggToCSV(finalBoRank_df, BoRank_df, BoROR_df, BoRS_df, BoAggCorr,fname_AggScoretop40,fname_spreadSheettop20)
n_pres = 20
years = 10
fname_spreadSheet = f'PresentationTop{n_pres}-{fidag}_{tickerfilter}.xlsx'
pbs.createPresentation(finalBoRank_df,baseurl,fname,n_pres,years)

