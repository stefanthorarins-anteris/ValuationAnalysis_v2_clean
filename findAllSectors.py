import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime

def findAllSectorsViaProfile(baseurl,api_key):

    resp_assets = requests.get(f'{baseurl}v3/available-traded/list?apikey={api_key}')
    #resp_assets = requests.get(f'{baseurl}v3/stock/list?apikey={api_key}')
    ass_df = pd.DataFrame(resp_assets.json())
    sectordic = {}
    pbar = tqdm(total=len(ass_df['symbol'].unique()))
    for symbol in ass_df['symbol'].unique():
        resp_profile = requests.get(f'{baseurl}v3/profile/{symbol}?apikey={api_key}')
        symb_sector = resp_profile.json()[0]['sector']

        if symb_sector not in sectordic:
            sectordic[symb_sector] = []

        sectordic[symb_sector].append(symbol)
        pbar.update(n=1)
    pbar.close()

    fidag = datetime.today().strftime('%Y-%m-%d')
    sectordicfn = f'sectorsdic_fmp_{fidag}.pickle'

    newsectordic = {}
    dicmaps = {'Financial': 'Financial Services', 'Industrial Goods': 'Industrials', 'Biotechnology': 'Healthcare',
               'Pharmaceuticals': 'Healthcare', 'Retail': 'Consumer Cyclical', 'Banking': 'Financial Services'}
    for key in sectordic.keys():
        if key in dicmaps:
            newkey = dicmaps[key]
        elif key in ['','N/A',None]:
            newkey = 'Unspecified'
        elif len(sectordic[key]) < 10:
            newkey = 'Unspecified'
        else:
            newkey = key

        if newkey not in newsectordic:
            newsectordic[newkey] = sectordic[key]
        else:
            newsectordic[newkey] = newsectordic[newkey] + sectordic[key]

    pd.to_pickle(newsectordic, sectordicfn)

    return newsectordic

## USELESS I THINK
def findAllSectorsViaScreener(baseurl,api_key):

    stocks_seen = []
    sectordic = {}
    exchanges = ['nyse', 'nasdaq', 'amex', 'euronext', 'tsx']
    betaLowerThan = [0.5    ,2.1,1000]
    betaMoreThan = [-1000   ,1.1,2.1]
    priceLowerThan = [1,  7.5,  12,  70, 150,   1000000]
    priceMoreThan = [0,   1,    7.5 , 12,   70,   150]
    isEtf = False
    isActivelyTrading = True
    baseMegaCap = 100000000000
    marketCapMoreThan = [0                  ,baseMegaCap/1000,baseMegaCap/100,baseMegaCap/10,baseMegaCap,
                         baseMegaCap*10]
    marketCapLowerThan = [baseMegaCap/1000 ,baseMegaCap/100 ,baseMegaCap/10 ,baseMegaCap  ,baseMegaCap*10,
                         100*baseMegaCap]

    T = len(exchanges)*len(betaLowerThan)**2*len(priceMoreThan)**2*len(marketCapLowerThan)**2
    tqdm(total=len(ass_df['symbol'].unique()))
    for ex in exchanges:
        for i in range(0,len(betaLowerThan)-1):
            blt = betaLowerThan[i]
            bmt = betaMoreThan[i]
            for j in range(0,len(marketCapLowerThan)-1):
                mcmt = marketCapMoreThan[j]
                mclt = marketCapLowerThan[j]
                for k in range(0,len(priceMoreThan)-1):
                    plt = priceLowerThan[k]
                    pmt = priceMoreThan[k]

                    reqstr = f"{baseurl}v3/stock-screener?marketCapLowerThan={mclt}&marketCapMoreThan={mcmt}&" \
                              f"betaLowerThan={blt}&betaMoreThan={bmt}&" \
                              f"priceLowerThan={plt}&priceMoreThan={pmt}&exchange={ex}&" \
                              f"isEtf=true&isActivelyTrading=true&apikey={api_key}"
                    resp_screen = requests.get(reqstr).json()
                    if len(resp_screen) > 0:
                        sdf = pd.DataFrame(resp_screen)
                        for symbol in sdf['symbol']:
                            stocks_seen.append(symbol)
                            symb_sector = sdf[sdf['symbol'] == symbol].sector
                            if symb_sector not in sectordic:
                                sectordic[symb_sector] = []
                                sectordic[symb_sector].append(symbol)

