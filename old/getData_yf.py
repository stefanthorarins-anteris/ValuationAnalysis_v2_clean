import yfinance as yf
import pandas as pd
from tqdm import tqdm
import numpy as np

def get_fundamentals_yf(tckrnames_df, nrTaT=-1, startIndex=0,fundElsToAccItems,itemAlts_dict):
    if not isinstance(tckrnames_df, pd.DataFrame):
        raise Exception('provide a DataFrame')
    nrFails = 0
    tickersFailed = []
    pbar = tqdm(total=len(tckrnames_df))
    df_master_TaT = pd.DataFrame()
    tickerlist = []
    cntr = 0
    tckrnames_df = tckrnames_df.iloc[startIndex:]
    for row in tckrnames_df.itertuples():
        cc = yf.Ticker(row.Ticker)
        # If something fails here, break and update nrFails and failsIndex
        incs = cc.financials
        if len(incs.columns) > 2:
            bs = cc.balancesheet
            cf = cc.cashflow
            fs = pd.concat([incs,bs,cf])

            data = fs.T
            data['Source'] = row.Ticker
            data = data.reset_index()
            data.columns = ['Date', *data.columns[1:]]

            tempmaster = pd.concat([df_master_TaT,data])
            tempmaster = tempmaster.reset_index(drop=True)
            tempmaster.columns = ['Date', *tempmaster.columns[1:]]
            df_master_TaT = tempmaster

            ## calculate sums for each year to later get averages
            for basekey in fundElsToAccItems:
                for calcdicts in basekey:
                    ## check null and delete null
                    #something like
                    data_nancheck = data['Date', itemAlts, 'Source']
                    data_nan = data_nancheck[itemAlts].isnull().assign(date=df['date'], source=df['source'])

                    test = df_nan.groupby(['date', 'source'])
                    faillist = df_nan[keywords].all(axis=1).astype(int)

                    ## calc sum
                    #something like
                    avs[accitem] = data_nonan.groupby('Date')[accitem].mean()
                    sms[accitem] = data_nonan.groupby('Date')[accitem].sum()


        else:
            nrFails = nrFails + 1
            tickersFailed += [row.Ticker]

        cntr += 1
        if cntr == nrTaT:
            break
        pbar.update(n=1)

    return df_master_TaT, nrFails, tickersFailed


def get_compGen_yf(tckr,startDaily,endDaily,startQuarterly,endQuarterly,startYearly,endYearly):
    cc = yf.Ticker(tckr)
    phis = cc.history(start='startDaily', end='endDaily')
    phis = phis.iloc[::-1]
    phis.reset_index()
    priceHist = phis['High']

    relInfoDaily = priceHist

    nrShares = cc.shares
    Rearnings = cc.earnings
    marketCap = cc.info['Market Cap']
    ftEmpl = cc.info['fullTimeEmployees']

    relInfoYearly = pd.concat([nrShares,Rearnings])

    return relInfoDaily,relInfoYearly

