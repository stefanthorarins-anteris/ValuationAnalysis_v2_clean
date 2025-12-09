import pandas as pd
import requests
from datetime import datetime
import getData_gen as gdg


def getHistPrices(symbol, api_key, baseurl):
    """Return a DataFrame of historical prices for `symbol` with columns ['date','price','symbol'].
    Uses FMP endpoint `historical-price-full/{symbol}` and parses 'close' as price.
    """
    resp = gdg.safe_get(f"{baseurl}v3/historical-price-full/{symbol}?apikey={api_key}")
    rows = []
    if resp:
        hist = resp.get('historical') if isinstance(resp, dict) else resp
        if hist:
            for r in hist:
                try:
                    rows.append({'date': pd.to_datetime(r.get('date')), 'price': r.get('close', None), 'symbol': symbol})
                except Exception:
                    continue
    return pd.DataFrame(rows)


def getHistDivs(symbol, api_key, baseurl):
    """Return a DataFrame of historical dividends for `symbol` with columns ['date','dividend','symbol'].
    Uses `historical-price-full/stock_dividend/{symbol}`. Returns empty DataFrame on failure.
    """
    resp = gdg.safe_get(f"{baseurl}v3/historical-price-full/stock_dividend/{symbol}?apikey={api_key}")
    rows = []
    if resp:
        # response may be {'symbol':..., 'historical': [ ... ]}
        if isinstance(resp, dict) and 'historical' in resp:
            hist = resp.get('historical', [])
        elif isinstance(resp, list):
            hist = resp
        else:
            hist = []

        for r in hist:
            try:
                d = pd.to_datetime(r.get('date'))
                dv = r.get('adjDividend', r.get('dividend', 0))
                dv = float(dv) if dv is not None else 0.0
                rows.append({'date': d, 'dividend': dv, 'symbol': symbol})
            except Exception:
                continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('date').reset_index(drop=True)
    return df


def getPrice(symbol, histprice, pricedate):
    """Return the price (float) for `symbol` nearest to `pricedate` (on or before nearest available).
    Returns NaN if no price found.
    """
    if histprice is None or histprice.empty:
        return float('nan')
    df = histprice[histprice['symbol'] == symbol]
    if df.empty:
        return float('nan')
    # ensure datetime
    try:
        pricedate = pd.to_datetime(pricedate)
    except Exception:
        pass
    # choose nearest date (prefer exact match or nearest earlier date)
    df = df.sort_values('date')
    # prefer on-or-before pricedate; if none, take earliest after
    earlier = df[df['date'] <= pricedate]
    if not earlier.empty:
        return float(earlier.iloc[-1]['price'])
    later = df[df['date'] > pricedate]
    if not later.empty:
        return float(later.iloc[0]['price'])
    return float('nan')


def getDividends(symbol, histdivs_df, start_date, end_date):
    """Sum dividends for `symbol` with payment date in (start_date, end_date] (exclusive start, inclusive end).
    `histdivs_df` may be a DataFrame for the symbol or a combined DataFrame — function will filter by symbol if necessary.
    """
    if histdivs_df is None or histdivs_df.empty:
        return 0.0
    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    except Exception:
        return 0.0

    df = histdivs_df
    if 'symbol' in df.columns:
        df = df[df['symbol'] == symbol]
    df = df[(df['date'] > start_date) & (df['date'] <= end_date)]
    return float(df['dividend'].sum())


def nearestDate(dates, pivot):
    return min(dates, key=lambda x: abs(x - pivot))