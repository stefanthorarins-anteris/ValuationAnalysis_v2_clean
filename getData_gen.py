import pandas as pd
import requests
import warnings
import json
import numpy as np
import os
import time


def safe_get(url, params=None, headers=None, timeout=10, retries=3, backoff=1):
    """Perform a GET request with basic retry/backoff, timeout and JSON parsing.

    Returns parsed JSON on success, or None on repeated failures.
    """
    attempt = 0
    while attempt < retries:
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            try:
                return resp.json()
            except ValueError:
                warnings.warn(f"Response from {url} was not valid JSON")
                return None
        except requests.RequestException as e:
            attempt += 1
            if attempt >= retries:
                warnings.warn(f"Failed to GET {url} after {retries} attempts: {e}")
                return None
            # simple exponential backoff
            sleep_for = backoff * (2 ** (attempt - 1))
            time.sleep(sleep_for)
    return None

def get_tickers(ds, baseurl, api_key, manual_elim=None, tfilt='stock_NA1',sfilt='all', mcapf=-1,fn=''):
    df = -1
    if ds == 'fromFile':
        # read tickers from CSV file; ensure returned dataframe is assigned to `df`
        df = pd.read_csv(fn)

    elif ds == 'fmp':
        # use safe_get to fetch API endpoints with retries and timeouts
        resp_stockAT_cmp_json = safe_get(f'{baseurl}v3/available-traded/list?apikey={api_key}')
        resp_tckr_json = safe_get(f'{baseurl}v3/financial-statement-symbol-lists?apikey={api_key}')
        resp_stockAT_cmp_df = pd.DataFrame(resp_stockAT_cmp_json) if resp_stockAT_cmp_json else pd.DataFrame()
        #resp_stock_cmp_df = pd.DataFrame(resp_stock_cmp.json())
        resp_tckr_df = pd.DataFrame(resp_tckr_json) if resp_tckr_json else pd.DataFrame()
        resp_tckr_df.columns = ['symbol']

        #tickers_df = resp_stock_cmp_df.merge(resp_tckr_df, on='symbol', how='inner', indicator=True)
        #tickers_df.drop('_merge', axis=1, inplace=True)

        maskAT = resp_stockAT_cmp_df['symbol'].isin(resp_tckr_df['symbol'])
        #mask = resp_stock_cmp_df['symbol'].isin(resp_tckr_df['symbol'])
        tickersAT_df = resp_stockAT_cmp_df[maskAT].drop_duplicates(subset='symbol').reset_index(drop=True)
        #tickers_df = resp_stock_cmp_df[mask].drop_duplicates(subset='symbol').reset_index(drop=True).copy()
        #tickers_df = resp_tckr_df

        df = tickerfilterWrapper(tickersAT_df, tfilt, sfilt, mcapf, baseurl, api_key)

    else:
        raise Exception('Not a valid tickers source')

    if manual_elim is None:
        manual_elim = []

    df = df[~df['symbol'].isin(manual_elim)].reset_index(drop=True)

    return df

def tickerfilterWrapper(tickdf,tfilt,sfilt,mcapf,baseurl,api_key):
    df = tickdf
    if tfilt == 'stock_US1':
        tickers_df_stock = filter_tickers(tickdf, 'type', 'stock', mcapf, api_key)
        tickers_df_stock_US1 = filter_tickers(tickers_df_stock, 'exchangeShortName', ['NYSE', 'NASDAQ'], mcapf, api_key)
        df = tickers_df_stock_US1
    if tfilt == 'stock_NA1':
        tickers_df_stock = filter_tickers(tickdf, 'type', 'stock', mcapf, api_key)
        tickers_df_stock_NA1 = filter_tickers(tickers_df_stock, 'exchangeShortName', ['NYSE', 'NASDAQ', 'TSX'], mcapf,
                                              api_key)
        df = tickers_df_stock_NA1
    if tfilt == 'stock_WW1_TV':
        tickers_df_stock = filter_tickers(tickdf, 'type', 'stock', mcapf, api_key)
        tickers_df_stock_WW1_TV = filter_tickers(tickers_df_stock, 'exchangeShortName',
                                                 ['NYSE', 'NASDAQ', 'EURONEXT', 'LSE', 'XETRA'], mcapf, api_key)
        df = tickers_df_stock_WW1_TV
    elif tfilt == 'stock_NA1_EU1':
        tickers_df_stock = filter_tickers(tickdf, 'type', 'stock', mcapf, api_key)
        tickers_df_stock_NA1_EU1 = filter_tickers(tickers_df_stock, 'exchangeShortName',
                                                   ['NYSE', 'NASDAQ', 'EURONEXT', 'LSE', 'TSX', 'XETRA', 'STO', 'OSE',
                                                    'ICE'], mcapf, api_key)
        df = tickers_df_stock_NA1_EU1
    elif tfilt == 'stock_US1_EU1':
        tickers_df_stock = filter_tickers(tickdf, 'type', 'stock', mcapf, api_key)
        tickers_df_stock_US1_EU1 = filter_tickers(tickers_df_stock, 'exchangeShortName',
                                                   ['NYSE', 'NASDAQ', 'EURONEXT', 'LSE', 'XETRA', 'STO', 'OSE',
                                                    'ICE'], mcapf, api_key)
        df = tickers_df_stock_US1_EU1
    elif tfilt == 'stock_US1_EU2':
        tickers_df_stock = filter_tickers(tickdf, 'type', 'stock', mcapf, api_key)
        tickers_df_stock_US1_EU2 = filter_tickers(tickers_df_stock, 'exchangeShortName',
                                                   ['NYSE', 'NASDAQ', 'EURONEXT'], mcapf, api_key)
        df = tickers_df_stock_US1_EU2

    if sfilt != 'all':
        df = filterBySector(df, sfilt)
    delist_json = safe_get(f'{baseurl}/v3/delisted-companies?page=0&apikey={api_key}')
    delist_df = pd.DataFrame(delist_json) if delist_json else pd.DataFrame()
    delist = list(delist_df['symbol']) if 'symbol' in delist_df.columns else []
    # record delisted tickers to a file for auditing
    try:
        import csv
        fidag = pd.Timestamp.today().strftime('%Y-%m-%d')
        with open(f'delisted_tickers_{fidag}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(delist)
    except Exception:
        # non-fatal: if we can't write, continue
        pass
    df = df[~df['symbol'].isin(delist)].reset_index(drop=True)

    return df

def filterBySector(df,sfilt):
    sectordic = pd.read_pickle('sectorsdic_fmp.pickle')

    dfsfilt = df[df['symbol'].isin(sectordic[sfilt])]

    return dfsfilt

def filter_tickers(ticker_df, colname, cond, mcap, api_key):
    # maybe check if colname is a string and is found in ticker_df
    # maybe check if cond is of type which is valid as a condition
    # start with the full dataframe; apply condition filters if provided
    ntdf = ticker_df.copy()

    if cond and isinstance(cond, str):
        ntdf = ticker_df[ticker_df[colname] == cond]
    elif cond and isinstance(cond, list) and all(isinstance(elem, str) for elem in cond):
        mask = np.zeros(len(ticker_df), dtype=bool)
        for c in cond:
            mask = mask | (ticker_df[colname].values == c)

        ntdf = ticker_df[mask]

    if mcap > 0:
        # use the already-filtered dataframe size when deciding whether to do per-symbol checks
        if len(ntdf) < 3000:
            # iterate over the filtered set and drop those below mcap
            to_drop = []
            for row in ntdf.itertuples():
                symb = row.symbol
                tempjson = safe_get(f'https://financialmodelingprep.com/api/v3/profile/{symb}?apikey={api_key}')
                if tempjson and len(tempjson) > 0:
                    mcap_inst = tempjson[0].get('mktCap')
                    if mcap_inst is None or mcap > mcap_inst:
                        to_drop.append(row.symbol)
                else:
                    # missing profile — treat as fail and drop
                    to_drop.append(row.symbol)

            if to_drop:
                ntdf = ntdf[~ntdf['symbol'].isin(to_drop)]
        else:
            warnings.warn("To many tickers left after other filters to do a market cap screening. It would take forever")
    ntdf.reset_index(drop=True, inplace=True)

    return ntdf

def checkIfValidFS(fs):
    retbool = True
    if any(fs['price'][0:10].isna()):
        retbool = False

    return retbool

def fixAfterGetData(BoMetric_df, cdx_df):
    BoMetric_df = BoMetric_df.dropna(subset=['source'])
    tempfix = BoMetric_df.reset_index(drop=False)
    tempfix2 = tempfix.drop(['index'], axis=1)
    BoMetric_df = tempfix2

    cdx_df = cdx_df.dropna(subset=['source'])
    tempfix = cdx_df.reset_index(drop=False)
    tempfix2 = tempfix.drop(['index'], axis=1)
    cdx_df = tempfix2

    BoMetric_df = forceNumOnDf(BoMetric_df)
    cdx_df = forceNumOnDf(cdx_df)

    return BoMetric_df, cdx_df

def forceNumOnDf(df):
    # Safely coerce each non-identifier column to numeric where possible.
    dftemp = df.copy()
    # Columns to preserve (identifiers / non-numeric)
    preserve = set()
    if 'date' in dftemp.columns:
        preserve.add('date')
    if 'source' in dftemp.columns:
        preserve.add('source')

    for col in dftemp.columns:
        if col in preserve:
            # skip coercion for identifier columns
            continue
        try:
            # try vectorized coercion for the column; invalid entries become NaN
            coerced = pd.to_numeric(dftemp[col], errors='coerce')
            dftemp[col] = coerced
        except Exception:
            # fallback: try coercing element-wise to be extra defensive
            try:
                dftemp[col] = dftemp[col].apply(lambda x: pd.to_numeric(x, errors='coerce'))
            except Exception:
                # leave the column as-is if it cannot be coerced
                dftemp[col] = dftemp[col]

    dftemp.replace([np.inf, -np.inf], np.nan, inplace=True)
    return dftemp